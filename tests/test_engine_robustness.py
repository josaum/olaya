"""Regression tests for concept selection, determinism and input validation in the Python engine."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from olaya.axioms import project_hierarchical_distribution
from olaya.candidate_router import CandidateRouter
from olaya.ontology import create_incident_ontology
from olaya.sequence import build_ontology_sequence

STATE = {
    "service": "billing-worker",
    "error": "Deadlock detected while updating customer subscription balances in PostgreSQL cluster.",
}


def test_hierarchical_projection_is_monotone_and_normalized():
    onto = create_incident_ontology()
    candidates = ["INCIDENT:DB_DEADLOCK", "INCIDENT:DB_OUTAGE", "INCIDENT:PAYMENT_GATEWAY"]
    subgraph = onto.extract_subgraph(candidates)
    probs, marginals, conf = project_hierarchical_distribution(np.array([2.0, 1.0, 0.5]), candidates, subgraph)

    assert sum(probs.values()) == pytest.approx(1.0)
    assert marginals["INCIDENT:ROOT"] == pytest.approx(1.0)
    assert marginals["INCIDENT:DATA"] == pytest.approx(probs["INCIDENT:DB_DEADLOCK"] + probs["INCIDENT:DB_OUTAGE"])
    for curie, concept in subgraph.concepts.items():
        for parent in concept.parents:
            assert marginals[parent] >= marginals[curie]
    assert 0.0 <= conf <= 1.0


def test_hierarchical_projection_suppresses_disjoint_subtrees():
    onto = create_incident_ontology()
    # INCIDENT:INFRA is disjoint with INCIDENT:BILLING (parent of PAYMENT_GATEWAY).
    candidates = ["INCIDENT:INFRA", "INCIDENT:PAYMENT_GATEWAY", "INCIDENT:DB_DEADLOCK"]
    subgraph = onto.extract_subgraph(candidates)
    probs, marginals, _ = project_hierarchical_distribution(np.array([5.0, 1.0, 1.0]), candidates, subgraph)

    assert probs["INCIDENT:PAYMENT_GATEWAY"] == 0.0
    assert marginals["INCIDENT:BILLING"] == 0.0
    assert sum(probs.values()) == pytest.approx(1.0)


def test_projection_rejects_empty_candidates():
    onto = create_incident_ontology()
    with pytest.raises(ValueError, match="zero candidates"):
        project_hierarchical_distribution(np.array([]), [], onto)


def test_router_falls_back_without_root_on_novel_text():
    router = CandidateRouter(create_incident_ontology())
    curies = [c.curie for c, _ in router.route_candidates("zzz qqq")]
    # Single top-level concept (ROOT) -> its children.
    assert curies == sorted(create_incident_ontology().concepts["INCIDENT:ROOT"].children)


def test_sequence_keeps_one_marker_with_huge_instructions(tokenizer):
    onto = create_incident_ontology()
    concepts = [onto.concepts["INCIDENT:DB_DEADLOCK"], onto.concepts["INCIDENT:DB_OUTAGE"]]
    bundle = build_ontology_sequence(
        tokenizer_encode=lambda t: tokenizer.encode(t, add_special_tokens=False),
        special_ids={"cls": 1, "sep": 2, "mask": 3, "pad": 0},
        state="x",
        instructions="word " * 1000,
        active_concepts=concepts,
        ontology=onto,
    )
    assert len(bundle.marker_positions) >= 1
    assert bundle.candidate_curies[0] == "INCIDENT:DB_DEADLOCK"


def test_concept_without_root_on_novel_text(olaya):
    ans = olaya.system_one("zzz qqq", {"t": {"type": "concept"}}).answers["t"]
    assert ans.curie in olaya.ontology.concepts["INCIDENT:ROOT"].children


def test_concept_unknown_root_raises(olaya):
    with pytest.raises(ValueError, match="unknown root"):
        olaya.system_one(STATE, {"t": {"type": "concept", "root": "NOPE"}})


def test_subsumes_unknown_concept_raises(olaya):
    with pytest.raises(ValueError, match="unknown subsumes concept"):
        olaya.system_one(STATE, {"t": {"type": "subsumes", "concept": "NOPE"}})


def test_subsumes_does_not_duplicate_target(olaya):
    ans = olaya.system_one(STATE, {"t": {"type": "subsumes", "concept": "INCIDENT:DATA"}}).answers["t"]
    assert 0.0 <= ans.subsumes <= 1.0


def test_long_instructions_do_not_crash(olaya):
    ans = olaya.system_one(STATE, {"t": {"type": "noul", "instructions": "word " * 400}}).answers["t"]
    assert 0.0 <= ans.noul <= 1.0


@pytest.mark.parametrize("criteria", [["refund", "restart"], {"refund": "issue a refund", "restart": None}])
def test_choice_accepts_list_and_dict(olaya, criteria):
    ans = olaya.system_one(STATE, {"t": {"type": "choice", "criteria": criteria}}).answers["t"]
    assert ans.choice in ("refund", "restart")
    assert set(ans.probabilities) == {"refund", "restart"}


@pytest.mark.parametrize("criteria", [[], {}, None, "refund"])
def test_choice_rejects_bad_criteria(olaya, criteria):
    with pytest.raises(ValueError, match="choice"):
        olaya.system_one(STATE, {"t": {"type": "choice", "criteria": criteria}})


DETERMINISM_SCRIPT = """
import json, sys, torch
sys.path.insert(0, {tests_dir!r})
from conftest import load_local_tokenizer
from olaya import OLaya
torch.manual_seed(0)
o = OLaya(tokenizer=load_local_tokenizer(), device="cpu")
r = o.system_one({state!r}, {{
    "triage": {{"type": "concept", "root": "INCIDENT:ROOT"}},
    "data": {{"type": "subsumes", "concept": "INCIDENT:DATA"}},
}})
t = r.answers["triage"]
print(json.dumps([t.curie, t.candidate_probabilities, t.probabilities, r.answers["data"].subsumes]))
"""


def test_results_independent_of_hash_seed():
    script = DETERMINISM_SCRIPT.format(tests_dir=str(Path(__file__).parent), state=STATE)
    outputs = set()
    for seed in ("0", "1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
        outputs.add(proc.stdout.strip().splitlines()[-1])
    assert len(outputs) == 1, outputs
    curie = json.loads(outputs.pop())[0]
    assert curie != "INCIDENT:ROOT"
