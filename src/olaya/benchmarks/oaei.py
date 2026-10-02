"""OAEI (Ontology Alignment Evaluation Initiative) Benchmark and Formal Validation Suite.

Provides OWL ontology parsing, OAEI reference alignment ingestion, and
formal validation of semantic alignment and axiomatic consistency for O-Laya.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olaya.axioms import (
    enforce_disjointness_exclusion,
    enforce_subsumption_monotonicity,
    project_hierarchical_distribution,
    verify_axioms,
)
from olaya.candidate_router import CandidateRouter
from olaya.ontology import Concept, Ontology

# XML Namespaces used in OWL and OAEI Alignment RDF
NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
NS_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
NS_OWL = "http://www.w3.org/2002/07/owl#"
NS_ALIGN = "http://knowledgeweb.semanticweb.org/heterogeneity/alignment"


def clean_uri(uri: str | None) -> str:
    """Extract local name / fragment from URI."""
    if not uri:
        return ""
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.split("/")[-1]


def camel_to_words(name: str) -> str:
    """Convert CamelCase or snake_case identifier to human-readable words."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", s)
    return s.replace("_", " ").replace("-", " ").strip()


class OWLParser:
    """Parser for OWL/RDF ontologies into O-Laya Ontology objects."""

    @staticmethod
    def parse(owl_file: str | Path) -> Ontology:
        tree = ET.parse(owl_file)
        root = tree.getroot()
        onto_name = Path(owl_file).stem
        ontology = Ontology(name=onto_name)

        classes_dict: dict[str, Concept] = {}
        parents_map: dict[str, set[str]] = {}
        disjoints_map: dict[str, set[str]] = {}

        for elem in root.findall(f".//{{{NS_OWL}}}Class"):
            cid = elem.attrib.get(f"{{{NS_RDF}}}ID") or elem.attrib.get(f"{{{NS_RDF}}}about")
            if not cid:
                continue
            curie = clean_uri(cid)
            if not curie or curie.startswith("_") or curie == "Thing":
                continue

            # Label: rdfs:label or camel_to_words(curie)
            label_elem = elem.find(f"{{{NS_RDFS}}}label")
            label = label_elem.text.strip() if label_elem is not None and label_elem.text else camel_to_words(curie)

            # Comment / Description: rdfs:comment
            comment_elem = elem.find(f"{{{NS_RDFS}}}comment")
            comment = comment_elem.text.strip() if comment_elem is not None and comment_elem.text else ""

            # Synonyms: generate variations
            synonyms = []
            if label.lower() != curie.lower():
                synonyms.append(curie)
            words = label.split()
            if len(words) > 1:
                synonyms.append(" ".join(reversed(words)))

            # SubClassOf
            if curie not in parents_map:
                parents_map[curie] = set()
            for sc in elem.findall(f"{{{NS_RDFS}}}subClassOf"):
                res = sc.attrib.get(f"{{{NS_RDF}}}resource")
                if res:
                    p = clean_uri(res)
                    if p and p != "Thing" and p != curie:
                        parents_map[curie].add(p)
                nested = sc.find(f"{{{NS_OWL}}}Class")
                if nested is not None:
                    p = clean_uri(nested.attrib.get(f"{{{NS_RDF}}}ID") or nested.attrib.get(f"{{{NS_RDF}}}about"))
                    if p and p != "Thing" and p != curie:
                        parents_map[curie].add(p)

            # DisjointWith
            if curie not in disjoints_map:
                disjoints_map[curie] = set()
            for dj in elem.findall(f"{{{NS_OWL}}}disjointWith"):
                res = dj.attrib.get(f"{{{NS_RDF}}}resource")
                if res:
                    d = clean_uri(res)
                    if d and d != curie:
                        disjoints_map[curie].add(d)
                nested = dj.find(f"{{{NS_OWL}}}Class")
                if nested is not None:
                    d = clean_uri(nested.attrib.get(f"{{{NS_RDF}}}ID") or nested.attrib.get(f"{{{NS_RDF}}}about"))
                    if d and d != curie:
                        disjoints_map[curie].add(d)

            if curie not in classes_dict:
                classes_dict[curie] = Concept(
                    curie=curie,
                    label=label,
                    description=comment or f"{label} concept in {onto_name} ontology",
                    synonyms=synonyms,
                )
            else:
                # Merge comment/label if found later
                if comment and not classes_dict[curie].description:
                    classes_dict[curie].description = comment

        # Add concepts to ontology
        for c in classes_dict.values():
            ontology.add_concept(c)

        # Wire subsumption relations (only between existing concepts)
        for curie, parents in parents_map.items():
            if curie not in ontology.concepts:
                continue
            for p in parents:
                if p in ontology.concepts:
                    ontology.add_subsumption(curie, p)

        # Wire disjointness relations (only between existing concepts)
        for curie, disjoints in disjoints_map.items():
            if curie not in ontology.concepts:
                continue
            for d in disjoints:
                if d in ontology.concepts:
                    ontology.add_disjointness(curie, d)

        return ontology


@dataclass
class OAEIAlignment:
    """An individual alignment correspondence between two ontologies."""

    entity1: str
    entity2: str
    relation: str = "="
    measure: float = 1.0


class OAEIAlignmentParser:
    """Parser for OAEI RDF Alignment format."""

    @staticmethod
    def parse(rdf_file: str | Path) -> list[OAEIAlignment]:
        tree = ET.parse(rdf_file)
        root = tree.getroot()
        alignments: list[OAEIAlignment] = []

        for cell in root.iter(f"{{{NS_ALIGN}}}Cell"):
            e1 = cell.find(f"{{{NS_ALIGN}}}entity1")
            e2 = cell.find(f"{{{NS_ALIGN}}}entity2")
            rel = cell.find(f"{{{NS_ALIGN}}}relation")
            meas = cell.find(f"{{{NS_ALIGN}}}measure")

            if e1 is not None and e2 is not None:
                u1 = clean_uri(e1.attrib.get(f"{{{NS_RDF}}}resource"))
                u2 = clean_uri(e2.attrib.get(f"{{{NS_RDF}}}resource"))
                r = rel.text.strip() if rel is not None and rel.text else "="
                m = float(meas.text.strip()) if meas is not None and meas.text else 1.0
                if u1 and u2:
                    alignments.append(OAEIAlignment(entity1=u1, entity2=u2, relation=r, measure=m))

        return alignments


@dataclass
class OAEIEvalResult:
    """Results of OAEI evaluation on a pair of ontologies."""

    pair_name: str
    total_references: int
    evaluated_classes: int
    hits_at_1: int
    hits_at_3: int
    hits_at_5: int
    mrr: float
    raw_monotonicity_violations: int
    raw_disjoint_collisions: int
    post_monotonicity_violations: int
    post_disjoint_collisions: int
    top_matches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def acc_at_1(self) -> float:
        return self.hits_at_1 / max(1, self.evaluated_classes)

    @property
    def recall_at_3(self) -> float:
        return self.hits_at_3 / max(1, self.evaluated_classes)

    @property
    def recall_at_5(self) -> float:
        return self.hits_at_5 / max(1, self.evaluated_classes)


class OAEIBenchmark:
    """Formal validation benchmark on OAEI tracks."""

    def __init__(self, ontologies_dir: str | Path, alignments_dir: str | Path):
        self.ontologies_dir = Path(ontologies_dir)
        self.alignments_dir = Path(alignments_dir)
        self._parsed_ontologies: dict[str, Ontology] = {}

    def get_ontology(self, name: str) -> Ontology:
        name_lower = name.lower()
        if name_lower not in self._parsed_ontologies:
            matches = [
                f for f in self.ontologies_dir.glob("*.owl")
                if f.stem.lower() == name_lower
            ]
            if not matches:
                matches = [
                    f for f in self.ontologies_dir.glob("*.owl")
                    if name_lower in f.stem.lower()
                ]
            if not matches:
                raise FileNotFoundError(f"Ontology file for {name} not found in {self.ontologies_dir}")
            self._parsed_ontologies[name_lower] = OWLParser.parse(matches[0])
        return self._parsed_ontologies[name_lower]

    def evaluate_pair(self, source_name: str, target_name: str, alignment_file: str | Path) -> OAEIEvalResult:
        source_onto = self.get_ontology(source_name)
        target_onto = self.get_ontology(target_name)
        alignments = OAEIAlignmentParser.parse(alignment_file)

        # Filter only alignments between classes present in both ontologies
        valid_alignments = [
            a for a in alignments
            if a.entity1 in source_onto.concepts and a.entity2 in target_onto.concepts
        ]

        if not valid_alignments:
            return OAEIEvalResult(
                pair_name=f"{source_name}-{target_name}",
                total_references=len(alignments),
                evaluated_classes=0,
                hits_at_1=0,
                hits_at_3=0,
                hits_at_5=0,
                mrr=0.0,
                raw_monotonicity_violations=0,
                raw_disjoint_collisions=0,
                post_monotonicity_violations=0,
                post_disjoint_collisions=0,
            )

        all_target_curies = list(target_onto.concepts.keys())
        router = CandidateRouter(target_onto, max_candidates=len(all_target_curies))

        hits_1 = 0
        hits_3 = 0
        hits_5 = 0
        reciprocal_ranks = []
        raw_mono_viols = 0
        raw_disj_colls = 0
        post_mono_viols = 0
        post_disj_colls = 0
        top_matches = []

        for align in valid_alignments:
            src_concept = source_onto.concepts[align.entity1]
            ground_truth_target = align.entity2

            # Formulate query text representing source concept in context
            query_parts = [src_concept.label]
            if src_concept.description and src_concept.description != f"{src_concept.label} concept in {source_onto.name} ontology":
                query_parts.append(src_concept.description)
            for p in src_concept.parents:
                if p in source_onto.concepts:
                    query_parts.append(f"type {source_onto.concepts[p].label}")
            query_text = ". ".join(query_parts)

            # Score all candidates in target ontology using router/lexical matching
            routed = router.route_candidates(query_text, top_k=len(all_target_curies))
            ranked_curies = [c.curie for c, _ in routed]
            # Fallback if any missing
            seen = set(ranked_curies)
            for tc in all_target_curies:
                if tc not in seen:
                    ranked_curies.append(tc)

            # Rank of ground truth target
            rank = ranked_curies.index(ground_truth_target) + 1 if ground_truth_target in ranked_curies else 999
            reciprocal_ranks.append(1.0 / rank)

            if rank == 1:
                hits_1 += 1
            if rank <= 3:
                hits_3 += 1
            if rank <= 5:
                hits_5 += 1

            # Check axiomatic consistency on top-5 predicted candidate set
            top_k_curies = ranked_curies[:5]
            # Dummy raw softmax probabilities based on rank
            raw_scores = [1.0 / (i + 1) for i in range(len(top_k_curies))]
            total = sum(raw_scores)
            raw_probs = {c: s / total for c, s in zip(top_k_curies, raw_scores)}

            # Check raw violations
            for c, p_c in raw_probs.items():
                for parent in target_onto.concepts[c].parents:
                    if parent in raw_probs and raw_probs[parent] < p_c - 1e-5:
                        raw_mono_viols += 1
            for a, b in target_onto.concepts.items():
                if a in raw_probs:
                    for dj in b.disjoint_with:
                        if dj in raw_probs and dj > a:
                            if max(raw_probs[a], raw_probs[dj]) >= 0.4 and min(raw_probs[a], raw_probs[dj]) > 0.05:
                                raw_disj_colls += 1

            # Apply Leio's axiomatic projection operator
            mono_probs = enforce_subsumption_monotonicity(raw_probs, target_onto)
            post_probs = enforce_disjointness_exclusion(mono_probs, target_onto, threshold=0.4)
            # Re-normalize if non-zero
            sum_p = sum(post_probs.values())
            if sum_p > 0:
                post_probs = {c: p / sum_p for c, p in post_probs.items()}

            for c, p_c in post_probs.items():
                for parent in target_onto.concepts[c].parents:
                    if parent in post_probs and post_probs[parent] < p_c - 1e-5:
                        post_mono_viols += 1
            for a, b in target_onto.concepts.items():
                if a in post_probs:
                    for dj in b.disjoint_with:
                        if dj in post_probs and dj > a:
                            if max(post_probs[a], post_probs[dj]) >= 0.4 and min(post_probs[a], post_probs[dj]) > 0.05:
                                post_disj_colls += 1

            top_matches.append({
                "source": align.entity1,
                "target_expected": ground_truth_target,
                "target_predicted": ranked_curies[0],
                "rank": rank,
                "hit": rank == 1,
            })

        mrr = sum(reciprocal_ranks) / max(1, len(reciprocal_ranks))

        return OAEIEvalResult(
            pair_name=f"{source_name}-{target_name}",
            total_references=len(alignments),
            evaluated_classes=len(valid_alignments),
            hits_at_1=hits_1,
            hits_at_3=hits_3,
            hits_at_5=hits_5,
            mrr=mrr,
            raw_monotonicity_violations=raw_mono_viols,
            raw_disjoint_collisions=raw_disj_colls,
            post_monotonicity_violations=post_mono_viols,
            post_disjoint_collisions=post_disj_colls,
            top_matches=top_matches,
        )

    def evaluate_all_available_pairs(self) -> list[OAEIEvalResult]:
        """Evaluate all pairwise reference alignments in the directory."""
        results = []
        for rdf_path in sorted(self.alignments_dir.glob("*.rdf")):
            stem = rdf_path.stem
            if "-" not in stem:
                continue
            src, tgt = stem.split("-", 1)
            try:
                res = self.evaluate_pair(src, tgt, rdf_path)
                if res.evaluated_classes > 0:
                    results.append(res)
            except Exception as e:
                print(f"Skipping pair {stem}: {e}")
        return results
