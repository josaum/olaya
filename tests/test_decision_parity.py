"""The Python lexical router must agree with the Rust and Node engines on spec/decision_parity.json.

The Python engine's decisions come from a neural head, so the runtime-independent contract checked
here is the lexical candidate router: for every fixture state it must rank ``expected_curie`` first,
and without a tie. The Rust (olaya-core, olaya-wasm) and Node test suites check that their lexical
engines *select* the same CURIE.
"""

import json
from pathlib import Path

import pytest

from olaya.candidate_router import CandidateRouter
from olaya.ontology import Ontology

FIXTURE = json.loads((Path(__file__).resolve().parent.parent / "spec" / "decision_parity.json").read_text())


@pytest.fixture(scope="module")
def router() -> CandidateRouter:
    ontology = Ontology.from_json(FIXTURE["ontology"])
    assert len(ontology.concepts) == len(FIXTURE["ontology"]["concepts"])
    return CandidateRouter(ontology, max_candidates=FIXTURE["max_candidates"])


@pytest.mark.parametrize("case", FIXTURE["queries"], ids=lambda c: c["expected_curie"])
def test_router_ranks_expected_concept_first(router: CandidateRouter, case: dict) -> None:
    ranked = router.route_candidates(case["state"])
    assert ranked, case["state"]
    assert ranked[0][0].curie == case["expected_curie"], [(c.curie, round(s, 3)) for c, s in ranked[:3]]
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], f"tied top score for {case['state']!r}"
