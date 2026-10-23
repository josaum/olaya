"""More trace-tampering cases for olaya.workflow_replay, reusing the shared replay fixtures."""

from __future__ import annotations

import copy

import pytest
from tests.test_workflow_replay import choice, compiled, run, trace_for  # noqa: F401  (compiled is a fixture)

from olaya.workflow_replay import replay_workflow


def _rows(trace, kind):
    return [row for row in trace["@graph"] if kind in row["@type"]]


TAMPERS = {
    "extra_top_level_key": (lambda t: t.update(extra=1), "Expected emitted trace document"),
    "graph_not_list": (lambda t: t.update({"@graph": {"not": "a list"}}), "oversized"),
    "entity_not_dict": (lambda t: t["@graph"].__setitem__(0, "not an entity"), "Invalid graph entity"),
    "types_not_list": (lambda t: _rows(t, "wf:ActionIntent")[0].update({"@type": "wf:ActionIntent"}), "types|run"),
    "no_run": (
        lambda t: _rows(t, "wf:WorkflowRun")[0]["@type"].remove("wf:WorkflowRun"),
        "exactly one workflow run",
    ),
    "model_identifier": (lambda t: _rows(t, "wf:WorkflowRun")[0].pop("wf:modelIdentifier"), "model identifier"),
    "steps_list": (lambda t: _rows(t, "wf:WorkflowRun")[0].update({"wf:steps": {"@list": []}}), "ordered decision"),
    "unknown_step": (
        lambda t: _rows(t, "wf:DecisionActivity")[0].update({"wf:step": {"@id": "urn:test:workflow/steps/nope"}}),
        "unknown decision node",
    ),
    "record_link": (
        lambda t: _rows(t, "wf:DecisionRecord")[0].update({"prov:wasGeneratedBy": {"@id": "urn:test:elsewhere"}}),
        "decision record link",
    ),
    "bad_reference_shape": (
        lambda t: _rows(t, "wf:DecisionActivity")[0].update({"wf:step": "urn:test:not-a-ref"}),
        "IRI reference",
    ),
    "unknown_predicted_class": (
        lambda t: _rows(t, "wf:DecisionRecord")[0].update(
            {"wf:predictedType": {"@id": "https://example.org/tbox#Ghost"}}
        ),
        "unknown TBox class",
    ),
    "probability_slots_empty": (
        lambda t: _rows(t, "wf:DecisionRecord")[0].update({"wf:candidateProbabilities": []}),
        "probability slots",
    ),
    "probability_slot_entity": (
        lambda t: _rows(t, "wf:DecisionRecord")[0].update({"wf:candidateProbabilities": [{"@type": "x"}]}),
        "probability entity",
    ),
    "duplicate_slot": (
        lambda t: _rows(t, "wf:DecisionRecord")[0]["wf:candidateProbabilities"].append(
            copy.deepcopy(_rows(t, "wf:DecisionRecord")[0]["wf:candidateProbabilities"][0])
        ),
        "Duplicate probability slot",
    ),
}


@pytest.mark.parametrize("case", sorted(TAMPERS))
def test_more_trace_tampering_is_rejected(compiled, case):  # noqa: F811
    trace = copy.deepcopy(trace_for(compiled, choice()))
    mutate, message = TAMPERS[case]
    mutate(trace)
    with pytest.raises(ValueError, match=message):
        replay_workflow(compiled, trace)


def test_replay_input_guards(compiled):  # noqa: F811
    trace = trace_for(compiled, choice())
    with pytest.raises(ValueError, match="Expected compiled workflow"):
        replay_workflow({"not": "compiled"}, trace)
    with pytest.raises(ValueError, match="finite JSON"):
        replay_workflow(compiled, {"@context": {}, "@graph": [object()]})
    assert run(replay_workflow(compiled, trace))["wf:executionMode"] == "replay"
