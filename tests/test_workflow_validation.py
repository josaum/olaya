"""Compiler and runtime validation branches of olaya.workflow, table-driven over the shared fixtures."""

from __future__ import annotations

import copy

import pytest
from tests.test_workflow import (  # noqa: F401  (pytest fixtures)
    EX,
    FakeEngine,
    answer,
    decision_activity,
    intent_node,
    workflow_document,
    workflow_ontology,
)

from olaya.engine import ConceptAnswer, NoulAnswer, ScoreAnswer, SubsumesAnswer
from olaya.workflow import compile_workflow, run_workflow


def _decision(doc):
    return doc["steps"]["storage"]


def _question(doc):
    return _decision(doc)["decide"]["question"]


def _numeric(doc, primitive="noul"):
    node = _decision(doc)
    node["decide"]["question"] = {"type": primitive, "root": "ex:Storage"}
    node["accept"] = {}
    node["route"] = [{"lt": 0.5, "to": "ambient"}, {"gte": 0.5, "to": "cold"}]
    return node


COMPILE_ERRORS = {
    "version": (lambda d: d.update(version=2), "Unsupported workflow version"),
    "prefix_map": (lambda d: d.update(prefixes=["ex"]), "prefix"),
    "no_steps": (lambda d: d.update(steps={}), "1..128 named nodes"),
    "start": (lambda d: d.update(start="nowhere"), "Unknown start node"),
    "properties_not_map": (lambda d: d["steps"]["cold"]["emit"].update(properties=["x"]), "must be a map"),
    "property_value": (
        lambda d: d["steps"]["cold"]["emit"].update(properties={"ex:note": ["a", "list"]}),
        "literals or IRI references",
    ),
    "invalid_node": (lambda d: d["steps"].update(cold=["not", "a", "node"]), "Invalid workflow node"),
    "unknown_primitive": (lambda d: _question(d).update(type="magic"), "Unknown primitive"),
    "instructions": (lambda d: _question(d).update(instructions=5), "Instructions must be text"),
    "repeated_option": (lambda d: _question(d).update(options=["ex:Cold", "ex:Cold"]), "option"),
    "irrelevant_field": (lambda d: _question(d).update(concept="ex:Cold"), "Irrelevant primitive fields|binding"),
    "require_axioms": (lambda d: _decision(d)["accept"].update(require_axioms="yes"), "require_axioms"),
    "routes_not_map": (lambda d: _decision(d).update(route=[{"lt": 1, "to": "cold"}]), "class map"),
    "route_outside_options": (
        lambda d: _decision(d)["route"].update({"ex:Review": "review"}),
        "Choice route not present in options|route",
    ),
    "concept_without_root": (lambda d: _question(d).clear() or _question(d).update(type="concept"), "root|binding"),
    "subsumes_without_concept": (
        lambda d: _question(d).clear() or _question(d).update(type="subsumes"),
        "concept|binding",
    ),
    "concept_route_outside_root": (
        lambda d: (
            _decision(d)["decide"].update(question={"type": "concept", "root": "ex:Storage"}),
            _decision(d).update(route={"ex:Cold": "cold", "ex:Review": "review"}),
        ),
        "outside bound root|route",
    ),
    "numeric_route_map": (lambda d: _numeric(d).update(route={"ex:Cold": "cold"}), "interval lists"),
    "empty_interval": (lambda d: _numeric(d).update(route=[{"gte": 0.5, "lt": 0.5, "to": "cold"}]), "Empty route"),
    "overlapping": (
        lambda d: _numeric(d).update(route=[{"lt": 0.6, "to": "ambient"}, {"gte": 0.5, "to": "cold"}]),
        "Overlapping",
    ),
}


@pytest.mark.parametrize("case", sorted(COMPILE_ERRORS))
def test_compile_validation_errors(workflow_ontology, workflow_document, case):  # noqa: F811
    mutate, message = COMPILE_ERRORS[case]
    doc = copy.deepcopy(workflow_document)
    mutate(doc)
    with pytest.raises(ValueError, match=message):
        compile_workflow(doc, workflow_ontology)


def test_run_workflow_input_and_snapshot_guards(workflow_ontology, workflow_document):  # noqa: F811
    compiled = compile_workflow(workflow_document, workflow_ontology)
    with pytest.raises(ValueError, match="Expected compiled workflow"):
        run_workflow({"not": "compiled"}, FakeEngine(workflow_ontology), {})
    with pytest.raises(ValueError, match="64 KiB"):
        run_workflow(compiled, FakeEngine(workflow_ontology), ["not", "a", "map"])
    with pytest.raises(ValueError, match="64 KiB"):
        run_workflow(compiled, FakeEngine(workflow_ontology), {"shipment": "x" * 70000})

    class DriftingEngine(FakeEngine):
        def system_one(self, state, questions):
            self.ontology.concepts["ex:Cold"].description = "drifted mid-run"
            return super().system_one(state, questions)

    doc = copy.deepcopy(workflow_document)
    doc["steps"]["cold"] = copy.deepcopy(doc["steps"]["storage"])
    doc["steps"]["cold"]["route"] = {"ex:Cold": "review", "ex:Ambient": "ambient"}
    two_step = compile_workflow(doc, workflow_ontology)
    with pytest.raises(ValueError, match="drifted during workflow execution"):
        run_workflow(two_step, DriftingEngine(copy.deepcopy(workflow_ontology)), {"shipment": "x"})


def _run(ontology, document, result, mutate=None):
    doc = copy.deepcopy(document)
    if mutate:
        mutate(doc)
    trace = run_workflow(compile_workflow(doc, ontology), FakeEngine(ontology, result), {"shipment": "text"})
    return decision_activity(trace), intent_node(trace)


ERROR_ANSWERS = {
    "abstention_flag": (answer(), {"abstained": "yes"}),
    "axiom_flag": (answer(), {"axioms_verified": "true"}),
    "probability_slot": (answer(), {"probabilities": {1: 0.9, "ex:Ambient": 0.1}}),
    "selection_absent": (answer(), {"choice": "ex:Missing"}),
    "normalization": (answer(), {"probabilities": {"ex:Cold": 0.5, "ex:Ambient": 0.2}}),
    "not_a_mapping": ("just a string", None),
}


@pytest.mark.parametrize("case", sorted(ERROR_ANSWERS))
def test_malformed_backend_answers_route_to_error(workflow_ontology, workflow_document, case):  # noqa: F811
    base, overrides = ERROR_ANSWERS[case]
    if overrides is None:
        result = base
    else:
        from dataclasses import asdict

        result = {**asdict(base), "type": "choice", **overrides}
    activity, intent = _run(workflow_ontology, workflow_document, result)
    assert activity["wf:status"] == "error" and str(EX.Review) in intent["@type"]


def test_concept_score_and_subsumes_answer_validation(workflow_ontology, workflow_document):  # noqa: F811
    def concept_question(doc):
        _decision(doc)["decide"]["question"] = {"type": "concept", "root": "ex:Storage"}

    outside = ConceptAnswer(curie="ex:Review", candidate_probabilities={"ex:Review": 1.0}, confidence=0.9)
    assert _run(workflow_ontology, workflow_document, outside, concept_question)[0]["wf:status"] == "error"

    def score_question(doc):
        _numeric(doc, "score")
        _question(doc)["criteria"] = ["ex:Cold", "ex:Ambient"]

    unknown_slot = ScoreAnswer(score=0.9, probabilities={"0": 0.1, "7": 0.9}, confidence=0.9)
    assert _run(workflow_ontology, workflow_document, unknown_slot, score_question)[0]["wf:status"] == "error"
    incomplete = ScoreAnswer(score=1.0, probabilities={"1": 1.0}, confidence=0.9)
    assert _run(workflow_ontology, workflow_document, incomplete, score_question)[0]["wf:status"] == "error"

    def subsumes_question(doc):
        _numeric(doc)
        _decision(doc)["decide"]["question"] = {"type": "subsumes", "concept": "ex:Cold"}

    changed = SubsumesAnswer(concept="ex:Ambient", subsumes=0.9, confidence=0.9)
    assert _run(workflow_ontology, workflow_document, changed, subsumes_question)[0]["wf:status"] == "error"

    tied = NoulAnswer(noul=0.5)
    activity, intent = _run(workflow_ontology, workflow_document, tied, _numeric)
    assert activity["wf:status"] == "abstained" and str(EX.Review) in intent["@type"]

    def uncovered(doc):
        _numeric(doc).update(route=[{"gte": 0.9, "to": "cold"}, {"lt": 0.1, "to": "ambient"}])

    activity, _ = _run(workflow_ontology, workflow_document, NoulAnswer(noul=0.7), uncovered)
    assert activity["wf:status"] == "abstained"
