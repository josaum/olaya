import json
from pathlib import Path

import pytest
from rdflib import RDF, Graph, Namespace, URIRef

from olaya.ontology import Concept, Ontology
from olaya.workflow_jsonld import WF, build_trace, expand_iri, ontology_digest, validate_prefixes


def test_inline_context_expands_all_trace_links_and_declared_predicates():
    document = {
        "id": "https://example.org/workflow/storage",
        "type": "ex:Workflow",
        "tbox": "https://example.org/tbox",
        "prefixes": {"ex": "https://example.org/tbox#"},
        "steps": {
            "choose": {
                "type": "ex:Decision",
                "decide": {"question": {"type": "choice", "root": "ex:Storage", "options": ["ex:Cold", "ex:Ambient"]}},
            },
            "cold": {"emit": {"type": "ex:DispatchCold"}},
        },
    }
    records = [
        {
            "node": "choose",
            "type": "ex:Decision",
            "status": "accepted",
            "reason": "accepted",
            "next": "cold",
            "question": {"type": "choice", "root": "ex:Storage", "options": ["ex:Cold", "ex:Ambient"]},
            "answer": {"confidence": 0.9, "probabilities": {"ex:Cold": 0.9, "ex:Ambient": 0.1}},
            "selected_type": "ex:Cold",
        }
    ]
    trace = build_trace(
        document,
        "a" * 64,
        "b" * 64,
        records,
        {"node": "cold", "type": "ex:DispatchCold", "properties": {"ex:note": "Requested"}},
        "urn:olaya:run:test",
        "test-model",
    )
    graph = Graph().parse(data=json.dumps(trace), format="json-ld")
    wf, ex = Namespace(WF), Namespace("https://example.org/tbox#")
    assert (URIRef("urn:olaya:run:test/intent"), RDF.type, ex.DispatchCold) in graph
    assert (URIRef("urn:olaya:run:test/steps/0/answer"), wf.predictedType, ex.Cold) in graph
    assert (URIRef("urn:olaya:run:test/steps/0"), wf.scope, ex.Storage) in graph
    assert (URIRef("urn:olaya:run:test/intent"), wf.tbox, URIRef(document["tbox"])) in graph
    assert not list(graph.triples((None, RDF.type, ex.Cold)))
    vocabulary = Graph().parse(Path("spec/workflow-vocabulary.jsonld"), format="json-ld")
    for _, predicate, _ in graph:
        if str(predicate).startswith(WF):
            assert list(vocabulary.triples((predicate, RDF.type, None))), predicate


def test_ontology_hash_is_stable_across_insertion_order():
    first, second = Ontology("same"), Ontology("same")
    for name in ("ex:A", "ex:B"):
        first.add_concept(Concept(name, name))
    for name in ("ex:B", "ex:A"):
        second.add_concept(Concept(name, name))
    assert ontology_digest(first) == ontology_digest(second)
    second.concepts["ex:A"].description = "changed"
    assert ontology_digest(first) != ontology_digest(second)


@pytest.mark.parametrize(
    "prefixes", [{"wf": "https://evil.example/"}, {"ex": "relative/"}, {"@vocab": "https://example.org/"}]
)
def test_prefixes_cannot_drop_or_redefine_vocabulary(prefixes):
    with pytest.raises(ValueError):
        validate_prefixes(prefixes)


def test_unknown_compact_iris_rejected_instead_of_opaque_scheme():
    with pytest.raises(ValueError):
        expand_iri("unknown:Class", {"ex": "https://example.org/"})
