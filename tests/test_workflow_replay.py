"""Replay policy checks use retained traces, never a model or private input."""

import copy
import inspect

import pytest

from olaya.ontology import Concept, Ontology
from olaya.workflow import compile_workflow, run_workflow
from olaya.workflow_replay import replay_workflow


@pytest.fixture
def compiled():
    onto = Ontology(name="replay test")
    for curie in ("Workflow", "Decision", "Root", "Action", "Review"):
        onto.add_concept(Concept("ex:" + curie, curie))
    for curie in ("Cold", "Ambient"):
        onto.add_concept(Concept("ex:" + curie, curie, parents={"ex:Root"}))
    for curie in ("DispatchCold", "DispatchAmbient"):
        onto.add_concept(Concept("ex:" + curie, curie, parents={"ex:Action"}))
    document = {
        "version": 1,
        "id": "urn:test:workflow",
        "tbox": "urn:test:tbox",
        "type": "ex:Workflow",
        "prefixes": {"ex": "https://example.org/tbox#"},
        "start": "choose",
        "steps": {
            "choose": {
                "type": "ex:Decision",
                "decide": {
                    "state": "input.shipment",
                    "question": {"type": "choice", "root": "ex:Root", "options": ["ex:Cold", "ex:Ambient"]},
                },
                "accept": {"min_confidence": 0.7},
                "route": {"ex:Cold": "cold", "ex:Ambient": "ambient"},
                "on_abstain": "review",
                "on_error": "review",
            },
            "cold": {"emit": {"type": "ex:DispatchCold"}},
            "ambient": {"emit": {"type": "ex:DispatchAmbient"}},
            "review": {"emit": {"type": "ex:Review"}},
        },
    }
    return compile_workflow(document, onto)


class Stub:
    def __init__(self, compiled, answer):
        import json

        self.ontology = Ontology.from_json(json.loads(compiled.ontology_json))
        self.answer = answer

    def system_one(self, *, state, questions):
        if isinstance(self.answer, Exception):
            raise self.answer
        return {"model": "retained-model-label", "answers": {next(iter(questions)): self.answer}}


def choice(**changes):
    return {
        "type": "choice",
        "choice": "ex:Cold",
        "probabilities": {"ex:Cold": 0.9, "ex:Ambient": 0.1},
        "confidence": 0.8,
        **changes,
    }


def trace_for(compiled, answer):
    return run_workflow(
        compiled, Stub(compiled, answer), {"shipment": "PRIVATE STATE never emitted"}, run_id="urn:test:original"
    )


def run(trace):
    return next(row for row in trace["@graph"] if "wf:WorkflowRun" in row["@type"])


@pytest.mark.parametrize(
    "answer",
    [
        choice(),
        choice(confidence=0.2),
        choice(probabilities={"ex:Cold": 0.5, "ex:Ambient": 0.5}),
        {"type": "choice", "abstained": True},
        {"type": "choice", "evidence_status": "unmatched"},
        RuntimeError("PRIVATE ERROR"),
    ],
)
def test_replay_acceptance_abstention_and_failure_markers(compiled, answer, monkeypatch):
    trace = trace_for(compiled, answer)

    def forbidden(*args, **kwargs):
        raise AssertionError("Actual backend must never be invoked")

    monkeypatch.setattr(Stub, "system_one", forbidden)
    replay = replay_workflow(compiled, trace)
    assert run(replay)["wf:executionMode"] == "replay"
    assert run(replay)["wf:replayOf"] == {"@id": "urn:test:original"}
    assert run(replay)["@id"] != run(trace)["@id"]
    assert run(replay)["wf:modelIdentifier"] == run(trace)["wf:modelIdentifier"]
    original_intent = next(row for row in trace["@graph"] if "wf:ActionIntent" in row["@type"])
    replay_intent = next(row for row in replay["@graph"] if "wf:ActionIntent" in row["@type"])
    assert original_intent["@type"] == replay_intent["@type"]
    assert "PRIVATE" not in str(replay)
    assert list(inspect.signature(replay_workflow).parameters) == ["compiled", "trace"]


@pytest.mark.parametrize("primitive", ["score", "noul", "subsumes"])
def test_replay_numeric_primitives(compiled, primitive):
    import json

    doc = compiled.document
    question = (
        {"type": primitive, "root": "ex:Root"} if primitive != "subsumes" else {"type": primitive, "concept": "ex:Cold"}
    )
    answer = {"type": primitive, "confidence": 0.8, primitive: 0.1}
    if primitive == "score":
        question["criteria"] = ["ex:Cold", "ex:Ambient"]
        answer["probabilities"] = {"0": 0.9, "1": 0.1}
    if primitive == "subsumes":
        answer["concept"] = "ex:Cold"
    doc["steps"]["choose"]["decide"]["question"] = question
    doc["steps"]["choose"]["route"] = [{"lt": 0.5, "to": "cold"}, {"gte": 0.5, "to": "ambient"}]
    new = compile_workflow(doc, Ontology.from_json(json.loads(compiled.ontology_json)))
    trace = trace_for(new, answer)
    assert run(replay_workflow(new, trace))["wf:executionMode"] == "replay"


@pytest.mark.parametrize(
    "tamper", ["hash", "context", "next", "sequence", "duplicate", "truncate", "extra", "answer", "intent"]
)
def test_trace_tampering_is_rejected(compiled, tamper):
    trace = copy.deepcopy(trace_for(compiled, choice()))
    graph = trace["@graph"]
    activity = next(row for row in graph if "wf:DecisionActivity" in row["@type"])
    if tamper == "hash":
        next(row for row in graph if row["@id"] == "urn:test:tbox")["wf:sha256"] = "forged"
    elif tamper == "context":
        trace["@context"]["ex"] = "https://evil.example/"
    elif tamper == "next":
        activity["wf:nextStep"] = {"@id": "urn:test:workflow/steps/ambient"}
    elif tamper == "sequence":
        activity["wf:sequence"] = 2
    elif tamper == "duplicate":
        graph.append(copy.deepcopy(activity))
    elif tamper == "truncate":
        graph.remove(next(row for row in graph if "wf:ActionIntent" in row["@type"]))
    elif tamper == "extra":
        graph.append({"@id": "urn:test:extra", "@type": ["prov:Entity"]})
    elif tamper == "answer":
        next(row for row in graph if "wf:DecisionRecord" in row["@type"])["wf:confidence"] = 0.1
    else:
        next(row for row in graph if "wf:ActionIntent" in row["@type"])["@type"][-1] = (
            "https://example.org/tbox#DispatchAmbient"
        )
    with pytest.raises(ValueError):
        replay_workflow(compiled, trace)


def test_replay_of_replay_is_explicitly_rejected(compiled):
    replay = replay_workflow(compiled, trace_for(compiled, choice()))
    with pytest.raises(ValueError, match="original live"):
        replay_workflow(compiled, replay)


def test_multistep_order_and_missing_answer_are_rejected(compiled):
    import json

    doc = compiled.document
    doc["steps"]["second"] = copy.deepcopy(doc["steps"]["choose"])
    doc["steps"]["choose"]["route"]["ex:Cold"] = "second"
    extended = compile_workflow(doc, Ontology.from_json(json.loads(compiled.ontology_json)))
    trace = trace_for(extended, choice())
    replay_workflow(extended, trace)
    activities = [row for row in trace["@graph"] if "wf:DecisionActivity" in row["@type"]]
    assert len(activities) == 2
    reordered = copy.deepcopy(trace)
    indices = [i for i, row in enumerate(reordered["@graph"]) if "wf:DecisionActivity" in row["@type"]]
    a, b = indices
    reordered["@graph"][a], reordered["@graph"][b] = reordered["@graph"][b], reordered["@graph"][a]
    with pytest.raises(ValueError, match="order"):
        replay_workflow(extended, reordered)
    missing = copy.deepcopy(trace)
    missing["@graph"] = [row for row in missing["@graph"] if row["@id"] != "urn:test:original/steps/1/answer"]
    with pytest.raises(ValueError, match="Missing linked"):
        replay_workflow(extended, missing)


def test_concept_axiom_gate_is_recomputed(compiled):
    import json

    doc = compiled.document
    node = doc["steps"]["choose"]
    node["decide"]["question"] = {"type": "concept", "root": "ex:Root"}
    node["accept"]["require_axioms"] = True
    new = compile_workflow(doc, Ontology.from_json(json.loads(compiled.ontology_json)))
    answer = {
        "type": "concept",
        "curie": "ex:Cold",
        "candidate_probabilities": {"ex:Cold": 0.9, "ex:Ambient": 0.1},
        "confidence": 0.8,
        "axioms_verified": True,
    }
    replay_workflow(new, trace_for(new, answer))
    answer["axioms_verified"] = False
    trace = trace_for(new, answer)
    replay = replay_workflow(new, trace)
    assert (
        next(row for row in replay["@graph"] if "wf:DecisionActivity" in row["@type"])["wf:reason"]
        == "axioms_unverified"
    )
