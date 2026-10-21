"""Bring-your-own ontology: olaya.ontology_io, olaya.domain_pack and OLaya.from_ontology / from_pack."""

from __future__ import annotations

import hashlib
import json

import pytest

from olaya.domain_pack import PACK_FORMAT, load_domain_pack
from olaya.ontology_io import OntologyFormatError, load_ontology, ontology_fingerprint

rdflib = pytest.importorskip("rdflib")

TTL = """
@prefix ex:   <https://example.org/shop#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

ex:Item a owl:Class ; rdfs:label "Item"@en .
ex:Food a owl:Class ; rdfs:subClassOf ex:Item ; rdfs:label "Food"@en, "Comida"@es ;
    rdfs:comment "Something edible.\\n   Kept cold or dry." .
ex:Tool a owl:Class ; rdfs:subClassOf ex:Item ; skos:prefLabel "Tool" ; skos:altLabel "Implement" .
ex:Milk a owl:Class ; rdfs:subClassOf ex:Food ; rdfs:label "Milk" .
ex:Hammer a owl:Class ; rdfs:subClassOf ex:Tool ; rdfs:label "Hammer" .
ex:Drill rdfs:subClassOf ex:Tool .
ex:PowerTool owl:equivalentClass [ owl:intersectionOf ( ex:Tool ex:Electric ) ] .
ex:Electric a owl:Class .
[] a owl:AllDisjointClasses ; owl:members ( ex:Food ex:Tool ) .
ex:Gadget owl:disjointUnionOf ( ex:Phone ex:Watch ) .
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_turtle_extraction(tmp_path):
    onto = load_ontology(_write(tmp_path, "shop.ttl", TTL))
    c = onto.concepts
    assert {"ex:Item", "ex:Food", "ex:Tool", "ex:Milk", "ex:Hammer", "ex:Drill", "ex:PowerTool"} <= set(c)
    assert c["ex:Food"].parents == {"ex:Item"}
    assert c["ex:Food"].label == "Food" and "Comida" in c["ex:Food"].synonyms
    assert c["ex:Food"].description == "Something edible. Kept cold or dry."
    assert c["ex:Tool"].label == "Tool" and c["ex:Tool"].synonyms == ["Implement"]
    assert c["ex:Drill"].label == "Drill"  # no label: local name
    assert c["ex:PowerTool"].parents == {"ex:Tool", "ex:Electric"}  # named intersection operands
    assert "ex:Tool" in c["ex:Food"].disjoint_with and "ex:Food" in c["ex:Tool"].disjoint_with
    assert c["ex:Phone"].parents == {"ex:Gadget"} and "ex:Watch" in c["ex:Phone"].disjoint_with
    assert c["ex:Milk"].metadata["iri"] == "https://example.org/shop#Milk"
    assert "ex:Milk" in c["ex:Food"].children
    assert onto.is_subsumed_by("ex:Milk", "ex:Item")


def test_local_curie_style_matches_rust_converter(tmp_path):
    onto = load_ontology(_write(tmp_path, "shop.ttl", TTL), curie_style="local")
    assert "Milk" in onto.concepts and onto.concepts["Milk"].parents == {"Food"}


def test_jsonld_and_fingerprint_are_format_independent(tmp_path):
    ttl = load_ontology(_write(tmp_path, "shop.ttl", TTL))
    graph = rdflib.Graph().parse(data=TTL, format="turtle")
    jsonld_path = _write(tmp_path, "shop.jsonld", graph.serialize(format="json-ld"))
    jl = load_ontology(jsonld_path)
    # JSON-LD carries no prefix bindings here, so compare in the Rust (local-name) CURIE scheme.
    a = load_ontology(_write(tmp_path, "a.ttl", TTL), curie_style="local")
    b = load_ontology(jsonld_path, curie_style="local")
    assert set(jl.concepts) and ontology_fingerprint(a) == ontology_fingerprint(b)
    # Round trip through O-Laya JSON preserves the fingerprint as well.
    rt = load_ontology(_write(tmp_path, "rt.json", ttl.to_json()))
    assert ontology_fingerprint(rt) == ontology_fingerprint(ttl)
    assert load_ontology(json.loads(ttl.to_json())).concepts.keys() == ttl.concepts.keys()


def test_directory_merge_and_errors(tmp_path):
    d = tmp_path / "closure"
    d.mkdir()
    _write(d, "a.ttl", TTL)
    _write(
        d,
        "b.ttl",
        "@prefix ex: <https://example.org/shop#> .\n@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\nex:Cheese rdfs:subClassOf ex:Food .\n",
    )
    merged = load_ontology(d)
    assert merged.concepts["ex:Cheese"].parents == {"ex:Food"}
    with pytest.raises(FileNotFoundError):
        load_ontology(tmp_path / "missing.ttl")
    with pytest.raises(OntologyFormatError):
        load_ontology(_write(tmp_path, "x.unknown", "nothing"))
    with pytest.raises(ValueError):
        load_ontology(d, curie_style="weird")


def test_curie_collisions_fall_back_to_iris(tmp_path):
    ttl = """
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<https://a.example/Thing> rdfs:subClassOf <https://a.example/Root> .
<https://b.example/Thing> rdfs:subClassOf <https://a.example/Root> .
"""
    onto = load_ontology(_write(tmp_path, "c.ttl", ttl), curie_style="local")
    assert {"https://a.example/Thing", "https://b.example/Thing"} <= set(onto.concepts)


def _pack(tmp_path, **overrides):
    onto_path = _write(tmp_path, "shop.ttl", TTL)
    manifest = {"format": PACK_FORMAT, "name": "shop", "ontology": "shop.ttl", "router": "hybrid", **overrides}
    _write(tmp_path, "pack.json", json.dumps(manifest))
    return onto_path


def test_domain_pack_manifest(tmp_path):
    onto_path = _pack(
        tmp_path, ontology_fingerprint="abc", encoder={"model": "m", "checkpoint": "ckpt.pt", "temperature": 0.1}
    )
    pack = load_domain_pack(tmp_path)
    assert pack.ontology == onto_path.resolve() or pack.ontology == onto_path
    assert pack.checkpoint == tmp_path.resolve() / "ckpt.pt"
    assert pack.temperature == 0.1 and pack.encoder_model == "m" and pack.ontology_fingerprint == "abc"
    _write(tmp_path, "pack.json", json.dumps({"format": "nope", "name": "x", "ontology": "shop.ttl"}))
    with pytest.raises(ValueError):
        load_domain_pack(tmp_path)
    _write(
        tmp_path,
        "pack.json",
        json.dumps({"format": PACK_FORMAT, "name": "x", "ontology": "shop.ttl", "router": "magic"}),
    )
    with pytest.raises(ValueError):
        load_domain_pack(tmp_path)


def test_engine_from_ontology_and_pack(tmp_path, tokenizer):
    pytest.importorskip("torch")
    from olaya.embeddings import HashEncoder
    from olaya.engine import OLaya

    onto_path = _pack(tmp_path)
    fingerprint = ontology_fingerprint(load_ontology(onto_path))
    engine = OLaya.from_ontology(
        onto_path, encoder=HashEncoder(dim=128), expected_fingerprint=fingerprint, tokenizer=tokenizer, device="cpu"
    )
    result = engine.system_one("cold milk carton", {"q": {"type": "concept", "root": "ex:Item"}})
    answer = result.answers["q"]
    assert answer.axioms_verified is True and answer.lineage[0] == "ex:Item"
    assert "+scorer:" in engine.model_id and "+candidates:routed" in engine.model_id

    with pytest.raises(ValueError, match="fingerprint"):
        OLaya.from_ontology(onto_path, encoder=HashEncoder(), expected_fingerprint="0" * 64, tokenizer=tokenizer)
    with pytest.raises(ValueError):
        OLaya.from_ontology(onto_path, encoder=HashEncoder(), router="lexical", tokenizer=tokenizer)
    bogus = _write(tmp_path, "bogus.pt", "not a checkpoint")
    with pytest.raises(ValueError, match="sha256"):
        OLaya.from_ontology(onto_path, checkpoint=bogus, checkpoint_sha256="0" * 64, tokenizer=tokenizer)
    assert hashlib.sha256(bogus.read_bytes()).hexdigest() != "0" * 64

    via_pack = OLaya.from_pack(tmp_path, encoder=HashEncoder(dim=128), tokenizer=tokenizer, device="cpu")
    assert (
        via_pack.system_one("hammer and nails", {"q": {"type": "concept", "root": "ex:Item"}})
        .answers["q"]
        .axioms_verified
    )


def test_caller_supplied_evidence_replaces_host_harvester():
    from olaya.ontology import Concept, Ontology
    from olaya.real_ontology_train import generate_real_ontology_observations

    onto = Ontology("ev")
    onto.add_concept(Concept("A:ROOT", "Root"))
    for i in range(9):
        onto.add_concept(Concept(f"A:{i}", f"Thing {i}", parents={"A:ROOT"}))
        onto.concepts["A:ROOT"].children.add(f"A:{i}")
    rows = generate_real_ontology_observations(onto, "generic", 50, evidence=[("A:1", "real observation one")])
    assert any(r.source_kind == "real_abox" and r.target_curie == "A:1" for r in rows)
    with pytest.warns(DeprecationWarning):
        legacy = generate_real_ontology_observations(onto, "generic", 50, use_real_abox=True)
    assert not any(r.source_kind == "real_abox" for r in legacy)
