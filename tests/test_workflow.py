"""Hermetic compiler, safe routing and actual JSON-LD graph contract tests."""

import copy
import json
from dataclasses import replace

import pytest
from rdflib import RDF, Graph, Literal, Namespace, URIRef

from olaya.engine import ChoiceAnswer, ConceptAnswer, NoulAnswer, OLaya, ScoreAnswer, SubsumesAnswer, SystemOneResult
from olaya.ontology import Concept, ObjectProperty, Ontology
from olaya.workflow import compile_workflow, run_workflow
from olaya.workflow_jsonld import WF

EX = Namespace("https://example.org/tbox#")
PROV = Namespace("http://www.w3.org/ns/prov#")
W = Namespace(WF)


@pytest.fixture
def workflow_ontology():
    ontology = Ontology("workflow-test-tbox")
    for name in (
        "Workflow",
        "Decision",
        "Storage",
        "Cold",
        "Ambient",
        "Action",
        "DispatchCold",
        "DispatchAmbient",
        "Review",
    ):
        ontology.add_concept(Concept("ex:" + name, name))
    ontology.add_subsumption("ex:Cold", "ex:Storage")
    ontology.add_subsumption("ex:Ambient", "ex:Storage")
    for name in ("DispatchCold", "DispatchAmbient", "Review"):
        ontology.add_subsumption("ex:" + name, "ex:Action")
    ontology.properties["ex:note"] = ObjectProperty("ex:note", domains={"ex:Action"})
    ontology.properties["ex:storageOnly"] = ObjectProperty("ex:storageOnly", domains={"ex:Storage"})
    return ontology


@pytest.fixture
def workflow_document():
    return {
        "version": 1,
        "id": "https://example.org/workflows/storage",
        "type": "ex:Workflow",
        "prefixes": {"ex": str(EX)},
        "tbox": "https://example.org/tbox",
        "start": "storage",
        "steps": {
            "storage": {
                "type": "ex:Decision",
                "decide": {
                    "state": "input.shipment",
                    "question": {
                        "type": "choice",
                        "root": "ex:Storage",
                        "options": ["ex:Cold", "ex:Ambient"],
                    },
                },
                "accept": {"min_confidence": 0.8, "min_margin": 0.15},
                "route": {"ex:Cold": "cold", "ex:Ambient": "ambient"},
                "on_abstain": "review",
                "on_error": "review",
            },
            "cold": {"emit": {"type": "ex:DispatchCold", "properties": {"ex:note": "cold dispatch intent"}}},
            "ambient": {"emit": {"type": "ex:DispatchAmbient", "properties": {}}},
            "review": {"emit": {"type": "ex:Review", "properties": {}}},
        },
    }


def answer(**changes):
    return replace(
        ChoiceAnswer(choice="ex:Cold", probabilities={"ex:Cold": 0.9, "ex:Ambient": 0.1}, confidence=0.9), **changes
    )


class FakeEngine:
    def __init__(self, ontology, result=None, error=None):
        self.ontology = ontology
        self.answer = result if result is not None else answer()
        self.error = error
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, copy.deepcopy(questions)))
        if self.error:
            raise self.error
        return SystemOneResult(
            model="test-only-fake", answers={key: self.answer for key in questions}, usage={}, latency_ms=0
        )


def decision_activity(trace):
    return next(node for node in trace["@graph"] if "wf:DecisionActivity" in node.get("@type", []))


def intent_node(trace):
    return next(node for node in trace["@graph"] if "wf:ActionIntent" in node.get("@type", []))


@pytest.mark.parametrize(
    "selected, probabilities, terminal",
    [
        ("ex:Cold", {"ex:Cold": 0.9, "ex:Ambient": 0.1}, EX.DispatchCold),
        ("ex:Ambient", {"ex:Cold": 0.1, "ex:Ambient": 0.9}, EX.DispatchAmbient),
    ],
)
def test_accepted_branch_calls_engine_and_expands_real_rdf(
    workflow_ontology, workflow_document, selected, probabilities, terminal
):
    engine = FakeEngine(workflow_ontology, answer(choice=selected, probabilities=probabilities))
    trace = run_workflow(
        compile_workflow(workflow_document, workflow_ontology),
        engine,
        {"shipment": "private shipment text"},
        run_id="urn:uuid:test-run",
    )
    assert len(engine.calls) == 1
    state, questions = engine.calls[0]
    assert state == "private shipment text"
    question = next(iter(questions.values()))
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"ex:Cold", "ex:Ambient"}
    assert "options" not in question
    assert "private shipment text" not in json.dumps(trace)
    intent = intent_node(trace)
    assert str(terminal) in intent["@type"]
    graph = Graph().parse(data=json.dumps(trace), format="json-ld")
    run = URIRef("urn:uuid:test-run")
    assert (run, RDF.type, W.WorkflowRun) in graph
    assert (run, RDF.type, EX.Workflow) in graph
    assert (run, PROV.used, URIRef(workflow_document["tbox"])) in graph
    assert (URIRef(intent["@id"]), RDF.type, terminal) in graph
    assert (URIRef(intent["@id"]), PROV.wasGeneratedBy, run) in graph
    record = next(graph.subjects(RDF.type, W.DecisionRecord))
    assert (record, W.predictedType, URIRef(str(EX) + selected.split(":")[1])) in graph
    # A decision links a TBox class; it never asserts the input is that class.
    assert not list(graph.subjects(RDF.type, EX.Cold))
    assert not list(graph.subjects(RDF.type, EX.Ambient))
    if selected == "ex:Cold":
        assert (URIRef(intent["@id"]), EX.note, Literal("cold dispatch intent")) in graph


@pytest.mark.parametrize(
    "result",
    [
        answer(confidence=0.1),
        {"type": "choice", "choice": "ex:Cold", "probabilities": {"ex:Cold": 0.9, "ex:Ambient": 0.1}},
        answer(probabilities={"ex:Cold": 0.5, "ex:Ambient": 0.5}),
        answer(probabilities={"ex:Cold": 0.55, "ex:Ambient": 0.45}),
    ],
)
def test_uncertain_answers_route_to_review(workflow_ontology, workflow_document, result):
    trace = run_workflow(
        compile_workflow(workflow_document, workflow_ontology),
        FakeEngine(workflow_ontology, result),
        {"shipment": "text"},
    )
    assert str(EX.Review) in intent_node(trace)["@type"]
    assert decision_activity(trace)["wf:status"] == "abstained"


@pytest.mark.parametrize(
    "result",
    [
        answer(confidence=float("nan")),
        answer(confidence=float("inf")),
        answer(confidence=-0.1),
        answer(probabilities={"ex:Cold": float("nan"), "ex:Ambient": 0.1}),
        answer(probabilities={"ex:Cold": 1.1, "ex:Ambient": -0.1}),
        answer(probabilities={"ex:Cold": 0.9}),
        answer(probabilities={"ex:Cold": 0.1, "ex:Ambient": 0.9}),
        answer(choice="ex:Review", probabilities={"ex:Review": 0.9, "ex:Cold": 0.1}),
        answer(type="concept"),
        {"type": "choice", "choice": "ex:Cold", "probabilities": [], "confidence": 0.9},
    ],
)
def test_malformed_answers_fail_soft_to_error_branch(workflow_ontology, workflow_document, result):
    trace = run_workflow(
        compile_workflow(workflow_document, workflow_ontology),
        FakeEngine(workflow_ontology, result),
        {"shipment": "text"},
    )
    assert str(EX.Review) in intent_node(trace)["@type"]
    assert decision_activity(trace)["wf:status"] == "error"
    json.dumps(trace, allow_nan=False)


def test_engine_error_and_missing_input_are_redacted(workflow_ontology, workflow_document):
    compiled = compile_workflow(workflow_document, workflow_ontology)
    engine = FakeEngine(workflow_ontology, error=RuntimeError("private credential details"))
    trace = run_workflow(compiled, engine, {"shipment": "private raw state"})
    assert str(EX.Review) in intent_node(trace)["@type"]
    assert "private" not in json.dumps(trace)
    engine = FakeEngine(workflow_ontology)
    missing = run_workflow(compiled, engine, {})
    assert str(EX.Review) in intent_node(missing)["@type"]
    assert not engine.calls


def test_compiled_snapshot_isolated_from_document_and_rejects_tbox_drift(workflow_ontology, workflow_document):
    compiled = compile_workflow(workflow_document, workflow_ontology)
    workflow_document["steps"]["cold"]["emit"]["type"] = "ex:Review"
    trace = run_workflow(compiled, FakeEngine(workflow_ontology), {"shipment": "text"})
    assert str(EX.DispatchCold) in intent_node(trace)["@type"]
    workflow_ontology.concepts["ex:Cold"].description = "modified after compile"
    with pytest.raises(ValueError, match="TBox"):
        run_workflow(compiled, FakeEngine(workflow_ontology), {"shipment": "text"})


def test_workflow_snapshot_integrity_is_checked(workflow_ontology, workflow_document):
    compiled = compile_workflow(workflow_document, workflow_ontology)
    tampered = replace(compiled, canonical_json=compiled.canonical_json + " ")
    with pytest.raises(ValueError, match="integrity"):
        run_workflow(tampered, FakeEngine(workflow_ontology), {"shipment": "text"})


@pytest.mark.parametrize("verified", [True, False])
def test_concept_answer_gate_preserves_tbox_links(workflow_ontology, workflow_document, verified):
    doc = copy.deepcopy(workflow_document)
    node = doc["steps"]["storage"]
    node["decide"]["question"] = {"type": "concept", "root": "ex:Storage"}
    node["accept"]["require_axioms"] = True
    result = ConceptAnswer(
        curie="ex:Cold",
        candidate_probabilities={"ex:Cold": 0.9, "ex:Ambient": 0.1},
        confidence=0.9,
        axioms_verified=verified,
    )
    trace = run_workflow(
        compile_workflow(doc, workflow_ontology), FakeEngine(workflow_ontology, result), {"shipment": "text"}
    )
    expected = EX.DispatchCold if verified else EX.Review
    assert str(expected) in intent_node(trace)["@type"]
    assert decision_activity(trace)["wf:status"] == ("accepted" if verified else "abstained")


@pytest.mark.parametrize(
    "primitive,result",
    [
        ("score", ScoreAnswer(score=0.9, probabilities={"0": 0.1, "1": 0.9}, confidence=0.9)),
        ("subsumes", SubsumesAnswer(concept="ex:Cold", subsumes=0.9, confidence=0.9)),
        ("noul", NoulAnswer(noul=0.9)),
    ],
)
def test_numeric_primitives_use_typed_interval_routes(workflow_ontology, workflow_document, primitive, result):
    doc = copy.deepcopy(workflow_document)
    node = doc["steps"]["storage"]
    question = {"type": primitive, "root": "ex:Storage"}
    if primitive == "score":
        question["criteria"] = ["ex:Cold", "ex:Ambient"]
    if primitive == "subsumes":
        question = {"type": primitive, "concept": "ex:Cold"}
    node["decide"]["question"] = question
    node["accept"] = {"min_margin": 0.15}
    node["route"] = [{"lt": 0.5, "to": "ambient"}, {"gte": 0.5, "to": "cold"}]
    engine = FakeEngine(workflow_ontology, result)
    trace = run_workflow(compile_workflow(doc, workflow_ontology), engine, {"shipment": "text"})
    assert str(EX.DispatchCold) in intent_node(trace)["@type"]
    assert decision_activity(trace)["wf:status"] == "accepted"
    if primitive == "score":
        assert next(iter(engine.calls[0][1].values()))["criteria"] == ["Cold", "Ambient"]
    Graph().parse(data=json.dumps(trace), format="json-ld")


@pytest.mark.parametrize(
    "case",
    [
        "unknown_class",
        "unknown_property",
        "property_domain",
        "incompatible_option",
        "cycle",
        "unreachable",
        "missing_route",
        "bad_state_ref",
        "reserved_prefix",
        "nonfinite_gate",
    ],
)
def test_compile_rejects_invalid_tbox_or_control_flow(workflow_ontology, workflow_document, case):
    doc = copy.deepcopy(workflow_document)
    decision = doc["steps"]["storage"]
    if case == "unknown_class":
        doc["steps"]["cold"]["emit"]["type"] = "ex:Missing"
    elif case == "unknown_property":
        doc["steps"]["cold"]["emit"]["properties"] = {"ex:Missing": "value"}
    elif case == "property_domain":
        doc["steps"]["cold"]["emit"]["properties"] = {"ex:storageOnly": "value"}
    elif case == "incompatible_option":
        decision["decide"]["question"]["options"] = ["ex:Cold", "ex:Review"]
    elif case == "cycle":
        decision["route"]["ex:Cold"] = "storage"
    elif case == "unreachable":
        doc["steps"]["unreachable"] = {"emit": {"type": "ex:Review"}}
    elif case == "missing_route":
        decision["route"]["ex:Cold"] = "missing"
    elif case == "bad_state_ref":
        decision["decide"]["state"] = "shipment"
    elif case == "reserved_prefix":
        doc["prefixes"]["prov"] = "https://evil.example/"
    elif case == "nonfinite_gate":
        decision["accept"]["min_confidence"] = float("nan")
    with pytest.raises(ValueError):
        compile_workflow(doc, workflow_ontology)


def test_real_untrained_engine_integration_uses_local_tokenizer(workflow_ontology, workflow_document, tokenizer):
    # Integration only: no weights loaded, no accuracy or calibration assertion.
    doc = copy.deepcopy(workflow_document)
    doc["steps"]["storage"]["accept"] = {"min_confidence": 1.0, "min_margin": 1.0}
    engine = OLaya(ontology=workflow_ontology, tokenizer=tokenizer, device="cpu", weights_path=False)
    trace = run_workflow(compile_workflow(doc, workflow_ontology), engine, {"shipment": "frozen seafood"})
    assert str(EX.Review) in intent_node(trace)["@type"]
    assert decision_activity(trace)["wf:status"] == "abstained"
    Graph().parse(data=json.dumps(trace), format="json-ld")


def test_score_cannot_route_against_its_distribution(workflow_ontology, workflow_document):
    node = workflow_document["steps"]["storage"]
    node["decide"]["question"] = {"type": "score", "root": "ex:Storage", "criteria": ["ex:Cold", "ex:Ambient"]}
    node["accept"] = {}
    node["route"] = [{"lt": 0.5, "to": "ambient"}, {"gte": 0.5, "to": "cold"}]
    engine = FakeEngine(workflow_ontology, ScoreAnswer(score=0.9, probabilities={"0": 0.9, "1": 0.1}, confidence=0.9))
    trace = run_workflow(compile_workflow(workflow_document, workflow_ontology), engine, {"shipment": "text"})
    assert decision_activity(trace)["wf:status"] == "error"
    assert str(EX.Review) in intent_node(trace)["@type"]


def test_compiler_requires_complete_tbox_prefix_bindings(workflow_ontology, workflow_document):
    workflow_ontology.add_concept(Concept("unknown:Child", "Child"))
    workflow_ontology.add_subsumption("unknown:Child", "ex:Storage")
    with pytest.raises(ValueError):
        compile_workflow(workflow_document, workflow_ontology)


def test_terminal_properties_cannot_inject_provenance(workflow_ontology, workflow_document):
    prop = str(PROV.wasGeneratedBy)
    workflow_ontology.add_property(ObjectProperty(prop))
    workflow_document["steps"]["cold"]["emit"]["properties"] = {prop: {"@id": "https://evil.example/activity"}}
    with pytest.raises(ValueError):
        compile_workflow(workflow_document, workflow_ontology)


def test_run_id_cannot_merge_with_workflow_definition(workflow_ontology, workflow_document):
    engine = FakeEngine(workflow_ontology)
    compiled = compile_workflow(workflow_document, workflow_ontology)
    with pytest.raises(ValueError):
        run_workflow(compiled, engine, {"shipment": "text"}, run_id=workflow_document["id"])
    assert not engine.calls


def test_full_iri_authoring_resolves_native_curie_classes(workflow_ontology, workflow_document):
    node = workflow_document["steps"]["storage"]
    node["decide"]["question"]["root"] = str(EX.Storage)
    node["decide"]["question"]["options"] = [str(EX.Cold), str(EX.Ambient)]
    node["route"] = {str(EX.Cold): "cold", str(EX.Ambient): "ambient"}
    workflow_document["steps"]["cold"]["emit"]["type"] = str(EX.DispatchCold)
    trace = run_workflow(
        compile_workflow(workflow_document, workflow_ontology), FakeEngine(workflow_ontology), {"shipment": "text"}
    )
    assert str(EX.DispatchCold) in intent_node(trace)["@type"]


@pytest.mark.parametrize("suffix", ["/intent", "/steps/0", "/steps/0/answer"])
def test_derived_execution_ids_do_not_merge_with_definitions(workflow_ontology, workflow_document, suffix):
    workflow_document["id"] = "urn:test:run" + suffix
    engine = FakeEngine(workflow_ontology)
    with pytest.raises(ValueError):
        run_workflow(
            compile_workflow(workflow_document, workflow_ontology), engine, {"shipment": "text"}, run_id="urn:test:run"
        )
    assert not engine.calls
