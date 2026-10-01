"""The Python kernel must reproduce spec/axioms_golden.json (shared with the Rust and Node runtimes)."""

import json
from pathlib import Path

import numpy as np
import pytest

from olaya.axioms import DISJOINT_THRESHOLD, calibrate_and_project_distribution, verify_axioms
from olaya.ontology import Concept, Ontology

GOLDEN = json.loads((Path(__file__).resolve().parent.parent / "spec" / "axioms_golden.json").read_text())


def _slot_ontology(case) -> Ontology:
    onto = Ontology("golden")
    for i in range(len(case["logits"])):
        onto.add_concept(Concept(curie=str(i), label=str(i), description=""))
    for child, parent in case["parent_edges"]:
        onto.add_subsumption(str(child), str(parent))
    for a, b in case["disjoint_edges"]:
        onto.add_disjointness(str(a), str(b))
    return onto


def test_threshold_matches_spec():
    assert DISJOINT_THRESHOLD == GOLDEN["disjoint_threshold"]


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_kernel_matches_golden(case):
    onto = _slot_ontology(case)
    curies = [str(i) for i in range(len(case["logits"]))]
    probs, conf = calibrate_and_project_distribution(np.array(case["logits"]), curies, onto)
    tol = GOLDEN["tolerance"]
    assert [probs[c] for c in curies] == pytest.approx(case["expected_probs"], abs=tol)
    assert conf == pytest.approx(case["expected_confidence"], abs=tol)
    assert verify_axioms(probs, onto) == case["expected_axioms_verified"]
