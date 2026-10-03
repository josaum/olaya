"""
Hard negative sampler for O-Laya training: mixes sibling, cousin, and random distractors.

Strategy:
- Siblings (same parent, different concept): hardest negatives — closely related in the hierarchy.
- Cousins (share grandparent but not parent): moderately hard — same broad domain, different branch.
- Random: easy baseline distractors drawn from the full pool.

Ancestors and descendants of the target are excluded from the distractor pool to prevent false negatives.
"""

from __future__ import annotations

import random

from olaya.ontology import Ontology


class HardNegativeSampler:
    """Distractor sampler that prioritises structurally near-miss concepts."""

    def __init__(
        self,
        ontology: Ontology,
        sibling_ratio: float = 0.5,
        cousin_ratio: float = 0.3,
    ) -> None:
        self.ontology = ontology
        self.sibling_ratio = sibling_ratio
        self.cousin_ratio = cousin_ratio

        # Precompute children map for fast sibling / cousin lookup
        self._children: dict[str, list[str]] = {}  # parent -> list of children
        for curie, concept in sorted(ontology.concepts.items()):
            for parent in sorted(concept.parents):
                self._children.setdefault(parent, []).append(curie)

    def get_siblings(self, curie: str) -> list[str]:
        """Return all sibling concepts (same parent, different concept)."""
        concept = self.ontology.concepts.get(curie)
        if not concept or not concept.parents:
            return []
        siblings: list[str] = []
        for parent in sorted(concept.parents):
            siblings.extend(c for c in self._children.get(parent, []) if c != curie)
        return sorted(set(siblings))

    def get_cousins(self, curie: str) -> list[str]:
        """Return cousin concepts (share a grandparent but not a parent)."""
        concept = self.ontology.concepts.get(curie)
        if not concept:
            return []
        cousins: list[str] = []
        for parent in sorted(concept.parents):
            parent_concept = self.ontology.concepts.get(parent)
            if not parent_concept:
                continue
            for grandparent in sorted(parent_concept.parents):
                for uncle in self._children.get(grandparent, []):
                    if uncle != parent:
                        cousins.extend(c for c in self._children.get(uncle, []) if c != curie)
        return sorted(set(cousins) - {curie})

    def sample_distractors(
        self,
        target_curie: str,
        k: int,
        rng: random.Random,
        exclude: set[str] | None = None,
    ) -> list[str]:
        """Sample *k* distractors: a mix of siblings, cousins, and random concepts.

        Ancestors and descendants of *target_curie* are always excluded so that
        distractors cannot be semantically correct answers.
        """
        exclude = (exclude or set()) | {target_curie}
        ancestors = self.ontology.get_ancestors(target_curie)
        descendants = self.ontology.get_descendants(target_curie)
        exclude = exclude | ancestors | descendants

        all_concepts = sorted(c for c in self.ontology.concepts if c not in exclude)
        if not all_concepts:
            return []

        n_siblings = max(1, int(k * self.sibling_ratio))
        n_cousins = max(0, int(k * self.cousin_ratio))
        n_random = k - n_siblings - n_cousins  # noqa: F841  (used implicitly via `needed`)

        siblings = [c for c in self.get_siblings(target_curie) if c not in exclude]
        cousins = [c for c in self.get_cousins(target_curie) if c not in exclude]
        random_pool = [c for c in all_concepts if c not in set(siblings) and c not in set(cousins)]

        chosen: list[str] = []
        chosen += rng.sample(siblings, min(n_siblings, len(siblings)))
        chosen += rng.sample(cousins, min(n_cousins, len(cousins)))

        # Fill remaining slots from random pool
        needed = k - len(chosen)
        chosen += rng.sample(random_pool, min(needed, len(random_pool)))

        # If still short, pad from full pool
        if len(chosen) < k:
            extra_pool = [c for c in all_concepts if c not in set(chosen)]
            chosen += rng.sample(extra_pool, min(k - len(chosen), len(extra_pool)))

        return chosen[:k]
