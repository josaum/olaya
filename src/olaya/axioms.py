"""Ontological consistency projections, subsumption monotonicity, and calibration."""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

import numpy as np

from olaya.ontology import Ontology


def temperature_by_cardinality(k: int) -> float:
    """Return calibrated temperature scaling based on candidate cardinality."""
    if k <= 2:
        return 0.85
    elif k <= 5:
        return 0.95
    elif k <= 10:
        return 1.10
    else:
        return 1.25


def normalized_entropy_confidence(probs: np.ndarray) -> float:
    """Calculate Jev-style normalized Shannon entropy confidence score.

    Confidence = 1 - (Entropy(P) / ln(K))
    Returns 1.0 for perfect certainty (Dirac delta), 0.0 for uniform uncertainty.
    """
    k = len(probs)
    if k <= 1:
        return 1.0

    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    p = p / np.sum(p)

    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(float(k))

    conf = 1.0 - (entropy / max_entropy)
    return float(np.clip(conf, 0.0, 1.0))


def enforce_subsumption_monotonicity(
    probs: Dict[str, float],
    subgraph: Ontology,
) -> Dict[str, float]:
    """Ensure that for every subsumption relation A subClassOf B:

    P(B) >= P(A).
    Uses a DAG relaxation sweep over topological generations.
    """
    adjusted = dict(probs)
    curies = list(adjusted.keys())

    # Build local DAG topological ordering (from leaves to parents)
    # If A is a child of B, P(B) must be at least P(A)
    # We iteratively lift parents if their children exceed them
    changed = True
    iterations = 0
    max_iterations = len(curies) * 2

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        for child_curie in curies:
            p_child = adjusted[child_curie]
            concept = subgraph.concepts.get(child_curie)
            if not concept:
                continue

            for parent_curie in concept.parents:
                if parent_curie in adjusted:
                    p_parent = adjusted[parent_curie]
                    if p_child > p_parent:
                        # Monotonicity violation: lift parent to match child
                        adjusted[parent_curie] = p_child
                        changed = True

    return adjusted


def enforce_disjointness_exclusion(
    probs: Dict[str, float],
    subgraph: Ontology,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Enforce that if concept A has high confidence (>= threshold),

    any concept B disjoint with A has its probability suppressed to zero.
    """
    adjusted = dict(probs)
    for c_a, p_a in list(adjusted.items()):
        if p_a >= threshold:
            concept_a = subgraph.concepts.get(c_a)
            if not concept_a:
                continue
            for c_b in concept_a.disjoint_with:
                if c_b in adjusted and c_b != c_a:
                    # Suppress disjoint peer
                    adjusted[c_b] = 0.0

    return adjusted


def _temperature_softmax(raw_logits: np.ndarray, k: int) -> np.ndarray:
    """Softmax over the first ``k`` logits with cardinality-calibrated temperature."""
    if k == 0:
        raise ValueError("Cannot calibrate a distribution over zero candidates.")
    logits = np.asarray(raw_logits, dtype=np.float64)[:k]
    scaled_logits = logits / temperature_by_cardinality(k)
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_logits / np.sum(exp_logits)


def hierarchical_marginals(candidate_probs: Dict[str, float], subgraph: Ontology) -> Dict[str, float]:
    """Lift a distribution over candidate concepts to marginals over every subgraph concept.

    P(concept) = sum of candidate mass at or below the concept. Because a child's descendant set is a
    subset of its parent's, P(parent) >= P(child) holds by construction.
    """
    marginals = {curie: 0.0 for curie in subgraph.concepts}
    for candidate, mass in candidate_probs.items():
        for ancestor in subgraph.get_ancestors(candidate):
            marginals[ancestor] += mass
    return marginals


def project_hierarchical_distribution(
    raw_logits: np.ndarray,
    candidate_curies: List[str],
    subgraph: Ontology,
    enforce_axioms: bool = True,
    disjoint_threshold: float = 0.6,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Calibrate candidate logits and project them onto the ontology.

    Returns ``(candidate_probs, marginals, confidence)``:
    - ``candidate_probs``: a proper distribution (sums to 1) over the candidate concepts.
    - ``marginals``: P(state is an instance of C) for every concept C in the subgraph; monotone
      under subsumption by construction.
    - ``confidence``: normalized entropy confidence of ``candidate_probs``.

    With ``enforce_axioms``, any concept whose marginal reaches ``disjoint_threshold`` (> 0.5, so at
    most one side of a disjoint pair can trigger) removes the mass of every candidate under its
    disjoint peers, and the candidate distribution is renormalized.
    """
    base_probs = _temperature_softmax(raw_logits, len(candidate_curies))
    candidate_probs = {curie: float(base_probs[i]) for i, curie in enumerate(candidate_curies)}

    if enforce_axioms:
        marginals = hierarchical_marginals(candidate_probs, subgraph)
        suppressed: Set[str] = set()
        for curie_a, p_a in marginals.items():
            if p_a < disjoint_threshold:
                continue
            under_a = subgraph.get_descendants(curie_a)
            for curie_b in subgraph.concepts[curie_a].disjoint_with:
                suppressed.update(subgraph.get_descendants(curie_b) - under_a)
        if suppressed:
            for curie in suppressed.intersection(candidate_probs):
                candidate_probs[curie] = 0.0
            total = sum(candidate_probs.values())
            if total > 0:
                candidate_probs = {c: p / total for c, p in candidate_probs.items()}

    marginals = hierarchical_marginals(candidate_probs, subgraph)
    conf = normalized_entropy_confidence(np.array(list(candidate_probs.values()), dtype=np.float64))
    return candidate_probs, marginals, conf


def calibrate_and_project_distribution(
    raw_logits: np.ndarray,
    candidate_curies: List[str],
    subgraph: Ontology,
    enforce_axioms: bool = True,
) -> Tuple[Dict[str, float], float]:
    """Apply temperature scaling, softmax, ontological projection, and compute confidence.

    Used for flat questions (choice, score, noul, subsumes). Monotonicity is enforced by lifting
    parents, so when the candidates include a hierarchy the returned values are no longer a simplex.
    Concept questions use :func:`project_hierarchical_distribution` instead.
    """
    base_probs = _temperature_softmax(raw_logits, len(candidate_curies))

    prob_map = {curie: float(base_probs[i]) for i, curie in enumerate(candidate_curies)}

    if enforce_axioms:
        # Enforce subsumption monotonicity
        prob_map = enforce_subsumption_monotonicity(prob_map, subgraph)
        # Enforce disjointness
        prob_map = enforce_disjointness_exclusion(prob_map, subgraph, threshold=0.6)

    # Compute normalized entropy confidence
    prob_values = np.array(list(prob_map.values()), dtype=np.float32)
    conf = normalized_entropy_confidence(prob_values)

    return prob_map, conf
