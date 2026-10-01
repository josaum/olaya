"""Ontological consistency projections, subsumption monotonicity, and calibration."""

from __future__ import annotations

import numpy as np

from olaya.ontology import Ontology

# Disjointness threshold shared by all runtimes (see spec/AXIOMS.md).
DISJOINT_THRESHOLD = 0.6


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
    probs: dict[str, float],
    subgraph: Ontology,
) -> dict[str, float]:
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
    probs: dict[str, float],
    subgraph: Ontology,
    threshold: float = 0.6,
) -> dict[str, float]:
    """Mutual exclusion for disjoint pairs (shared spec, see ``spec/AXIOMS.md``).

    Pairs (a, b) are visited in candidate order (a before b). If max(P(a), P(b)) >= threshold, the
    smaller side and all of its descendants are set to zero (keeping monotonicity); on a tie, b loses.
    """
    adjusted = dict(probs)
    for a, b in disjoint_pairs(list(adjusted), subgraph):
        p_a, p_b = adjusted[a], adjusted[b]
        if max(p_a, p_b) >= threshold:
            loser = b if p_a >= p_b else a
            for curie in subgraph.get_descendants(loser):
                if curie in adjusted:
                    adjusted[curie] = 0.0
    return adjusted


def disjoint_pairs(curies: list[str], subgraph: Ontology) -> list[tuple[str, str]]:
    """Disjoint pairs among ``curies`` as (earlier, later) in list order."""
    position = {c: i for i, c in enumerate(curies)}
    pairs = []
    for i, a in enumerate(curies):
        concept = subgraph.concepts.get(a)
        if concept is None:
            continue
        for b in sorted(concept.disjoint_with, key=lambda c: position.get(c, -1)):
            if position.get(b, -1) > i:
                pairs.append((a, b))
    return pairs


def verify_axioms(
    probs: dict[str, float],
    subgraph: Ontology,
    threshold: float = 0.6,
    tol: float = 1e-6,
) -> bool:
    """Check the guarantees the projection is meant to provide.

    - Monotonicity: P(parent) >= P(child) for every subsumption edge among ``probs``.
    - Exclusion: for every disjoint pair, either one side is zero or both are below ``threshold``.
    """
    for curie, p in probs.items():
        concept = subgraph.concepts.get(curie)
        if concept is None:
            continue
        for parent in concept.parents:
            if parent in probs and probs[parent] < p - tol:
                return False
    for a, b in disjoint_pairs(list(probs), subgraph):
        if min(probs[a], probs[b]) > tol and max(probs[a], probs[b]) >= threshold:
            return False
    return True


def _temperature_softmax(raw_logits: np.ndarray, k: int) -> np.ndarray:
    """Softmax over the first ``k`` logits with cardinality-calibrated temperature."""
    if k == 0:
        raise ValueError("Cannot calibrate a distribution over zero candidates.")
    logits = np.asarray(raw_logits, dtype=np.float64)[:k]
    scaled_logits = logits / temperature_by_cardinality(k)
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_logits / np.sum(exp_logits)


def hierarchical_marginals(candidate_probs: dict[str, float], subgraph: Ontology) -> dict[str, float]:
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
    candidate_curies: list[str],
    subgraph: Ontology,
    enforce_axioms: bool = True,
    disjoint_threshold: float = DISJOINT_THRESHOLD,
) -> tuple[dict[str, float], dict[str, float], float]:
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
        # Renormalizing can lift another concept over the threshold, so repeat to a fixpoint.
        # Each round only zeroes candidates, so this terminates within len(candidates) rounds.
        while True:
            marginals = hierarchical_marginals(candidate_probs, subgraph)
            suppressed: set[str] = set()
            for curie_a, p_a in marginals.items():
                if p_a < disjoint_threshold:
                    continue
                under_a = subgraph.get_descendants(curie_a)
                for curie_b in subgraph.concepts[curie_a].disjoint_with:
                    suppressed.update(subgraph.get_descendants(curie_b) - under_a)
            newly_zeroed = [c for c in suppressed.intersection(candidate_probs) if candidate_probs[c] > 0.0]
            if not newly_zeroed:
                break
            for curie in newly_zeroed:
                candidate_probs[curie] = 0.0
            total = sum(candidate_probs.values())
            if total <= 0:
                break
            candidate_probs = {c: p / total for c, p in candidate_probs.items()}

    marginals = hierarchical_marginals(candidate_probs, subgraph)
    conf = normalized_entropy_confidence(np.array(list(candidate_probs.values()), dtype=np.float64))
    return candidate_probs, marginals, conf


def calibrate_and_project_distribution(
    raw_logits: np.ndarray,
    candidate_curies: list[str],
    subgraph: Ontology,
    enforce_axioms: bool = True,
) -> tuple[dict[str, float], float]:
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
        prob_map = enforce_disjointness_exclusion(prob_map, subgraph, threshold=DISJOINT_THRESHOLD)

    # Compute normalized entropy confidence
    prob_values = np.array(list(prob_map.values()), dtype=np.float32)
    conf = normalized_entropy_confidence(prob_values)

    return prob_map, conf
