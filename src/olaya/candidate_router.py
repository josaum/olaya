"""Fast Top-K Candidate Router & Subgraph Extractor.

Enables scaling to massive ontologies (1,000+ to 100,000+ classes) by dynamically
selecting the active decision subgraph that fits into the decision model's context window.
Incorporates lexical synonym pooling and n-gram matching principles from mycelia-align.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from olaya.ontology import Concept, Ontology


def tokenize_normalize(text: str) -> List[str]:
    """Normalize text, split CamelCase, and extract lowercased alphanumeric tokens."""
    # Split CamelCase: "DatabaseOutage" -> "Database Outage"
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", s.lower())
    return [t for t in tokens if len(t) > 1]


class CandidateRouter:
    """Indexes an ontology and extracts the active decision subgraph for a given state."""

    def __init__(self, ontology: Ontology, max_candidates: int = 12):
        self.ontology = ontology
        self.max_candidates = max_candidates
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        self.concept_tokens: Dict[str, Set[str]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Index all concepts, labels, synonyms, and descriptions."""
        for curie, concept in self.ontology.concepts.items():
            pool_text = [concept.label, concept.description] + concept.synonyms
            full_text = " ".join(pool_text)
            tokens = set(tokenize_normalize(full_text))
            self.concept_tokens[curie] = tokens
            for token in tokens:
                self.inverted_index[token].add(curie)

    def route_candidates(
        self,
        state_text: str,
        root_curie: str | None = None,
        top_k: int | None = None,
    ) -> List[Tuple[Concept, float]]:
        """Score concepts against input state text and return top candidates."""
        k = top_k or self.max_candidates
        # Sorted so that float accumulation order (and therefore tie-breaking) is reproducible.
        query_tokens = sorted(set(tokenize_normalize(state_text)))
        query_set = set(query_tokens)

        # Filter candidate pool if constrained by root
        allowed_concepts: Set[str] = (
            self.ontology.get_descendants(root_curie) if root_curie else set(self.ontology.concepts.keys())
        )

        scores: Dict[str, float] = defaultdict(float)

        for token in query_tokens:
            matching_curies = self.inverted_index.get(token, set()).intersection(allowed_concepts)
            # Inverse document frequency weighting
            idf = 1.0 / (1.0 + len(matching_curies))
            for curie in sorted(matching_curies):
                scores[curie] += idf * 2.0

        # Add Jaccard overlap on token sets (only concepts sharing a token can score)
        for curie in sorted(scores):
            concept_toks = self.concept_tokens.get(curie, set())
            intersection = len(query_set.intersection(concept_toks))
            if intersection > 0:
                jaccard = intersection / len(query_set.union(concept_toks))
                scores[curie] += jaccard * 5.0

        # Sort descending by score, ties broken by CURIE for determinism
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        top_results = ranked[:k]

        if not top_results:
            # Completely novel terminology: fall back to the children of the root (or of the
            # ontology's single top-level concept), else the top-level concepts themselves.
            return [(self.ontology.concepts[c], 0.1) for c in self._fallback_curies(root_curie)[:k]]

        return [(self.ontology.concepts[curie], score) for curie, score in top_results]

    def _fallback_curies(self, root_curie: str | None) -> List[str]:
        if root_curie is not None:
            anchor = root_curie if root_curie in self.ontology.concepts else None
        else:
            top_level = sorted(c for c, concept in self.ontology.concepts.items() if not concept.parents)
            anchor = top_level[0] if len(top_level) == 1 else None
            if anchor is None:
                return top_level
        if anchor is None:
            return []
        children = sorted(self.ontology.concepts[anchor].children)
        return children or [anchor]

    def extract_decision_subgraph(
        self,
        state_text: str,
        root_curie: str | None = None,
        max_candidates: int | None = None,
    ) -> Ontology:
        """Extract a self-contained active decision subgraph tailored to the input state.

        Includes the top routed candidates and all of their ancestors.
        """
        limit = max_candidates or self.max_candidates
        top_candidates = self.route_candidates(state_text, root_curie=root_curie, top_k=limit)
        curies = [c.curie for c, _ in top_candidates]
        return self.ontology.extract_subgraph(curies, include_parents=True)
