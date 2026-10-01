"""Unit tests for O-Laya (Ontology-Grounded Decision Model)."""

import pytest
import numpy as np
from olaya.ontology import Concept, Ontology, create_incident_ontology
from olaya.candidate_router import CandidateRouter
from olaya.axioms import (
    enforce_subsumption_monotonicity,
    enforce_disjointness_exclusion,
    normalized_entropy_confidence,
    calibrate_and_project_distribution,
)
from olaya.sequence import build_ontology_sequence


def test_ontology_dag():
    onto = create_incident_ontology()
    assert "INCIDENT:ROOT" in onto.concepts
    assert "INCIDENT:DB_DEADLOCK" in onto.concepts

    # Test subsumption
    ancestors = onto.get_ancestors("INCIDENT:DB_DEADLOCK")
    assert "INCIDENT:DATA" in ancestors
    assert "INCIDENT:ROOT" in ancestors
    assert onto.is_subsumed_by("INCIDENT:DB_DEADLOCK", "INCIDENT:ROOT")
    assert onto.is_subsumed_by("INCIDENT:DB_DEADLOCK", "INCIDENT:DATA")
    assert not onto.is_subsumed_by("INCIDENT:DATA", "INCIDENT:DB_DEADLOCK")

    # Test disjointness
    assert onto.are_disjoint("INCIDENT:SECURITY", "INCIDENT:BILLING")
    assert onto.are_disjoint("INCIDENT:DB_OUTAGE", "INCIDENT:INVOICE_MISCALC")
    assert not onto.are_disjoint("INCIDENT:DB_DEADLOCK", "INCIDENT:DATA")

    # Test lineage
    lineage = onto.get_lineage("INCIDENT:DB_DEADLOCK")
    assert lineage == ["INCIDENT:ROOT", "INCIDENT:DATA", "INCIDENT:DB_DEADLOCK"]


def test_candidate_router():
    onto = create_incident_ontology()
    router = CandidateRouter(onto, max_candidates=5)

    # Route database issue
    query = "Postgres connection pool exhausted and deadlock detected on write queries"
    candidates = router.route_candidates(query, top_k=3)
    curies = [c.curie for c, _ in candidates]

    assert any("DB" in c or "DATA" in c for c in curies)

    # Extract active decision subgraph
    subgraph = router.extract_decision_subgraph(query, max_candidates=4)
    assert len(subgraph.concepts) >= 2
    # Ensure parents are included in subgraph
    for c in list(subgraph.concepts.values()):
        for p in c.parents:
            assert p in subgraph.concepts


def test_axioms_and_calibration():
    onto = create_incident_ontology()

    # Test normalized entropy
    p_certain = np.array([1.0, 0.0, 0.0])
    p_uniform = np.array([1/3, 1/3, 1/3])
    assert normalized_entropy_confidence(p_certain) == pytest.approx(1.0, abs=1e-3)
    assert normalized_entropy_confidence(p_uniform) == pytest.approx(0.0, abs=1e-3)

    # Test subsumption monotonicity: P(Parent) >= P(Child)
    subgraph = onto.extract_subgraph(["INCIDENT:DB_DEADLOCK", "INCIDENT:DATA", "INCIDENT:ROOT"])
    # Suppose child accidentally got higher logit than parent
    raw_probs = {
        "INCIDENT:DB_DEADLOCK": 0.85,
        "INCIDENT:DATA": 0.30,
        "INCIDENT:ROOT": 0.20,
    }
    fixed = enforce_subsumption_monotonicity(raw_probs, subgraph)
    assert fixed["INCIDENT:DATA"] >= fixed["INCIDENT:DB_DEADLOCK"]
    assert fixed["INCIDENT:ROOT"] >= fixed["INCIDENT:DATA"]

    # Test disjointness exclusion
    subgraph_disjoint = onto.extract_subgraph(["INCIDENT:SECURITY", "INCIDENT:BILLING"])
    disjoint_probs = {
        "INCIDENT:SECURITY": 0.90,
        "INCIDENT:BILLING": 0.40,
    }
    gated = enforce_disjointness_exclusion(disjoint_probs, subgraph_disjoint, threshold=0.5)
    assert gated["INCIDENT:BILLING"] == 0.0


def test_olaya_system_one_e2e(olaya):
    state = {
        "service": "billing-worker",
        "log_level": "FATAL",
        "error": "Deadlock detected while updating customer subscription balances in PostgreSQL cluster.",
    }

    result = olaya.system_one(
        state=state,
        questions={
            "triage": {
                "type": "concept",
                "instructions": "Classify the operational incident into the appropriate ontology category.",
                "root": "INCIDENT:ROOT",
            },
            "is_data_issue": {
                "type": "subsumes",
                "concept": "INCIDENT:DATA",
                "instructions": "Is this a data or storage issue?",
            },
            "urgency": {
                "type": "score",
                "instructions": "Rate the operational disruption level.",
                "criteria": ["low", "medium", "urgent", "critical"],
            },
            "escalate_oncall": {
                "type": "noul",
                "instructions": "Should on-call engineering be paged immediately?",
            },
        },
    )

    assert result.model == "o-laya-incident-v1"
    assert "triage" in result.answers
    assert "is_data_issue" in result.answers
    assert "urgency" in result.answers
    assert "escalate_oncall" in result.answers

    # Verify concept answer structure
    ans = result.answers["triage"]
    assert ans.curie in olaya.ontology.concepts
    assert ans.curie != "INCIDENT:ROOT"
    assert len(ans.lineage) >= 2
    assert ans.curie in ans.candidate_probabilities
    assert sum(ans.candidate_probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    # The selected concept is never an ancestor of another competing candidate.
    for other in ans.candidate_probabilities:
        if other != ans.curie:
            assert not olaya.ontology.is_subsumed_by(other, ans.curie)
    # Marginals are monotone under subsumption.
    for curie, p in ans.probabilities.items():
        for parent in olaya.ontology.concepts[curie].parents:
            if parent in ans.probabilities:
                assert ans.probabilities[parent] >= p - 1e-4
    assert ans.confidence >= 0.0
    assert ans.axioms_verified is True

    # Verify usage
    assert result.usage["input_tokens"] > 0
    assert result.usage["output_tokens"] == 0

    print(f"\nExecution Latency: {result.latency_ms:.2f} ms")
    print(f"Predicted Concept: {ans.curie} ({ans.label})")
    print(f"Lineage: {' -> '.join(ans.lineage)}")
    print(f"Confidence: {ans.confidence:.4f}")
