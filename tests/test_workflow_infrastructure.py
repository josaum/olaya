"""One TBox binds workflow intents, infrastructure, applications and relations."""

import copy
import json

import pytest
from rdflib import RDF, Graph, Namespace, URIRef

from olaya.ontology import Concept, ObjectProperty, Ontology
from olaya.workflow import compile_workflow, run_workflow
from olaya.workflow_jsonld import WF
from olaya.workflow_replay import replay_workflow

EX = Namespace("https://example.org/tbox#")
RES = Namespace("urn:test:resource:")


@pytest.fixture
def manifest():
    ontology = Ontology(name="workflow infrastructure applications")
    for name in ("Workflow", "Action", "Infrastructure", "Application"):
        ontology.add_concept(Concept("ex:" + name, name))
    ontology.add_concept(Concept("ex:Warehouse", "Warehouse", parents={"ex:Infrastructure"}))
    ontology.add_concept(Concept("ex:Planner", "Planner", parents={"ex:Application"}))
    for name, domain, ranges in (
        ("target", "Action", {"ex:Infrastructure"}),
        ("handledBy", "Action", {"ex:Application"}),
        ("runsOn", "Application", {"ex:Infrastructure"}),
        ("linkedTo", "Infrastructure", {"ex:Infrastructure"}),
        ("status", "Infrastructure", {"http://www.w3.org/2001/XMLSchema#string"}),
    ):
        ontology.add_property(ObjectProperty("ex:" + name, domains={"ex:" + domain}, ranges=ranges))
    document = {
        "version": 1,
        "id": "urn:test:workflow",
        "tbox": "urn:test:tbox",
        "type": "ex:Workflow",
        "prefixes": {"ex": str(EX), "res": str(RES)},
        "start": "finish",
        "resources": [
            {"id": "res:warehouse", "type": "ex:Warehouse", "properties": {"ex:status": "declared configuration"}},
            {"id": "res:planner", "type": "ex:Planner", "properties": {"ex:runsOn": {"@id": "res:warehouse"}}},
        ],
        "steps": {
            "finish": {
                "emit": {
                    "type": "ex:Action",
                    "properties": {"ex:target": {"@id": "res:warehouse"}, "ex:handledBy": {"@id": "res:planner"}},
                }
            }
        },
    }
    return ontology, document


class NoInference:
    def __init__(self, ontology):
        self.ontology = ontology

    def system_one(self, **kwargs):
        raise AssertionError("Terminal-only topology must not invoke inference")


def test_declared_infra_apps_and_intent_ranges_expand_as_rdf(manifest):
    ontology, document = manifest
    compiled = compile_workflow(document, ontology)
    assert compiled.document["resources"][0]["id"] == str(RES.warehouse)
    trace = run_workflow(compiled, NoInference(ontology), {}, run_id="urn:test:run")
    graph = Graph().parse(data=json.dumps(trace), format="json-ld")
    assert (RES.warehouse, RDF.type, EX.Warehouse) in graph
    assert (RES.planner, RDF.type, EX.Planner) in graph
    assert (RES.planner, EX.runsOn, RES.warehouse) in graph
    assert (URIRef("urn:test:run/intent"), EX.target, RES.warehouse) in graph
    assert (URIRef("urn:test:run/intent"), EX.handledBy, RES.planner) in graph
    resources = [row for row in trace["@graph"] if "wf:DeclaredResource" in row["@type"]]
    assert len(resources) == 2
    assert all(row["wf:resourceState"] == "declared" for row in resources)
    assert all(row["wf:declaredIn"] == {"@id": "urn:test:workflow"} for row in resources)
    assert replay_workflow(compiled, trace)


@pytest.mark.parametrize(
    "case",
    [
        "unknown_class",
        "duplicate_id",
        "collision",
        "unknown_property",
        "wrong_domain",
        "literal_range",
        "undeclared_ref",
        "wrong_range",
        "inline_type",
        "reserved_property",
        "unknown_resource_field",
        "too_many",
    ],
)
def test_invalid_resource_relations_fail_compile(manifest, case):
    ontology, doc = manifest
    doc = copy.deepcopy(doc)
    resource = doc["resources"][0]
    intent = doc["steps"]["finish"]["emit"]["properties"]
    if case == "unknown_class":
        resource["type"] = "ex:Missing"
    elif case == "duplicate_id":
        doc["resources"][1]["id"] = "res:warehouse"
    elif case == "collision":
        resource["id"] = doc["id"]
    elif case == "unknown_property":
        resource["properties"] = {"ex:Missing": "text"}
    elif case == "wrong_domain":
        resource["properties"] = {"ex:runsOn": {"@id": "res:warehouse"}}
    elif case == "literal_range":
        intent["ex:target"] = "warehouse"
    elif case == "undeclared_ref":
        intent["ex:target"] = {"@id": "res:unobserved"}
    elif case == "wrong_range":
        intent["ex:target"] = {"@id": "res:planner"}
    elif case == "inline_type":
        intent["ex:target"] = {"@id": "res:warehouse", "@type": "ex:Warehouse"}
    elif case == "reserved_property":
        ontology.add_property(ObjectProperty(WF + "status"))
        resource["properties"] = {WF + "status": "forged"}
    elif case == "unknown_resource_field":
        resource["deployment"] = "actual"
    else:
        doc["resources"] = [{"id": "res:n" + str(i), "type": "ex:Warehouse"} for i in range(129)]
    with pytest.raises(ValueError):
        compile_workflow(doc, ontology)


def test_resource_topology_cycles_are_allowed(manifest):
    ontology, doc = manifest
    doc["resources"].append(
        {"id": "res:warehouse2", "type": "ex:Warehouse", "properties": {"ex:linkedTo": {"@id": "res:warehouse"}}}
    )
    doc["resources"][0]["properties"]["ex:linkedTo"] = {"@id": "res:warehouse2"}
    compile_workflow(doc, ontology)


def test_all_known_class_ranges_must_hold(manifest):
    ontology, doc = manifest
    ontology.properties["ex:target"].ranges.add("ex:Application")
    with pytest.raises(ValueError, match="class range"):
        compile_workflow(doc, ontology)
    ontology.concepts["ex:Warehouse"].parents.add("ex:Application")
    compile_workflow(doc, ontology)


def test_resource_snapshot_and_replay_capacity(manifest):
    ontology, doc = manifest
    doc["resources"].extend({"id": "res:warehouse" + str(i), "type": "ex:Warehouse"} for i in range(20))
    compiled = compile_workflow(doc, ontology)
    doc["resources"][0]["type"] = "ex:Application"
    trace = run_workflow(compiled, NoInference(ontology), {})
    assert len(trace["@graph"]) > 3 * len(compiled.document["steps"]) + 4
    assert replay_workflow(compiled, trace)
    altered = copy.deepcopy(trace)
    next(row for row in altered["@graph"] if "wf:DeclaredResource" in row["@type"])["@type"][-1] = str(EX.Application)
    with pytest.raises(ValueError):
        replay_workflow(compiled, altered)


def test_property_class_range_aliases_cannot_bypass_validation(manifest):
    ontology, doc = manifest
    ontology.properties["ex:target"].ranges = {str(EX.Infrastructure)}
    ontology.properties["ex:target"].domains = {str(EX.Action)}
    compile_workflow(doc, ontology)
    doc["steps"]["finish"]["emit"]["properties"]["ex:target"] = {"@id": "res:planner"}
    with pytest.raises(ValueError, match="class range"):
        compile_workflow(doc, ontology)
