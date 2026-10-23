"""Edge cases and error paths across ontology_io, domain_pack, real_ontology_train, calibration and engine."""

from __future__ import annotations

import json

import numpy as np
import pytest

from olaya.domain_pack import PACK_FORMAT, file_sha256, load_domain_pack
from olaya.ontology import Concept, Ontology
from olaya.ontology_io import OntologyFormatError, load_ontology, ontology_from_graph

rdflib = pytest.importorskip("rdflib")


# --- ontology_io ------------------------------------------------------------------------------


def test_load_ontology_passthrough_and_jsonld_dict(tmp_path):
    onto = Ontology("x")
    assert load_ontology(onto) is onto
    doc = {
        "@context": {"rdfs": "http://www.w3.org/2000/01/rdf-schema#", "ex": "https://example.org/z#"},
        "@graph": [
            {"@id": "ex:Root", "@type": "http://www.w3.org/2002/07/owl#Class", "rdfs:label": "Root"},
            {"@id": "ex:Leaf", "rdfs:subClassOf": {"@id": "ex:Root"}, "rdfs:label": "Leaf"},
        ],
    }
    from_dict = load_ontology(doc, name="zoo")
    assert from_dict.name == "zoo" and len(from_dict.concepts) == 2
    leaf = next(c for c in from_dict.concepts.values() if c.label == "Leaf")
    assert len(leaf.parents) == 1

    # A .json file that is JSON-LD (not O-Laya JSON) goes through rdflib.
    path = tmp_path / "zoo.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert {c.label for c in load_ontology(path).concepts.values()} == {"Root", "Leaf"}

    # O-Laya JSON keeps its own name unless one is given.
    olaya_json = tmp_path / "o.json"
    olaya_json.write_text(json.dumps({"name": "kept", "concepts": [{"curie": "A", "label": "A"}]}))
    assert load_ontology(olaya_json).name == "kept"
    assert load_ontology(olaya_json, name="renamed").name == "renamed"


def test_load_ontology_errors(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "notes.txt").write_text("not rdf")
    with pytest.raises(OntologyFormatError, match="no RDF files"):
        load_ontology(empty_dir)
    broken = tmp_path / "broken.ttl"
    broken.write_text("@prefix ex: <https://e.org/> . ex:A ex:b", encoding="utf-8")
    with pytest.raises(OntologyFormatError, match="failed to parse"):
        load_ontology(broken)


def test_ontology_title_names_and_iri_edge_cases(tmp_path):
    ttl = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<https://e.org/onto> a owl:Ontology ; rdfs:label "My  Shop Ontology" .
<https://e.org/A> a owl:Class .
<https://e.org/B> rdfs:subClassOf [ owl:intersectionOf ( <https://e.org/A> <https://e.org/C> ) ] .
<https://e.org/C> a owl:Class ; owl:disjointWith <https://e.org/A> .
<urn:x:Thing> a owl:Class .
<https://e.org/D> a owl:Class ; owl:disjointUnionOf <https://e.org/notalist> .
"""
    path = tmp_path / "titled.ttl"
    path.write_text(ttl, encoding="utf-8")
    onto = load_ontology(path, curie_style="local")
    assert onto.name == "My_Shop_Ontology"
    assert onto.concepts["B"].parents == {"A", "C"}  # anonymous superclass intersection
    assert "A" in onto.concepts["C"].disjoint_with
    assert any(c.startswith("urn:") or c == "Thing" for c in onto.concepts)  # non-http IRIs survive

    g = rdflib.Graph().parse(data=ttl, format="turtle")
    assert set(ontology_from_graph(g, curie_style="local").concepts) == set(onto.concepts)


# --- domain_pack ------------------------------------------------------------------------------


def test_domain_pack_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_domain_pack(tmp_path)
    (tmp_path / "pack.json").write_text(json.dumps({"format": PACK_FORMAT, "name": "x"}))
    with pytest.raises(ValueError, match="missing 'ontology'"):
        load_domain_pack(tmp_path / "pack.json")
    (tmp_path / "pack.json").write_text(json.dumps({"format": PACK_FORMAT, "name": "x", "ontology": "nope.ttl"}))
    with pytest.raises(FileNotFoundError, match="nope.ttl"):
        load_domain_pack(tmp_path)
    (tmp_path / "o.json").write_text(json.dumps({"name": "o", "concepts": [{"curie": "A", "label": "A"}]}))
    (tmp_path / "pack.json").write_text(
        json.dumps({"format": PACK_FORMAT, "name": "x", "ontology": str(tmp_path / "o.json"), "notes": "kept"})
    )
    pack = load_domain_pack(tmp_path)
    assert pack.ontology == tmp_path / "o.json" and pack.extra == {"notes": "kept"} and pack.checkpoint is None
    assert pack.engine_kwargs()["router"] == "hybrid"
    assert len(file_sha256(tmp_path / "o.json")) == 64


# --- real_ontology_train ----------------------------------------------------------------------


def _small_onto(n: int = 9) -> Ontology:
    onto = Ontology("small")
    onto.add_concept(Concept("A:ROOT", "Root"))
    for i in range(n):
        onto.add_concept(Concept(f"A:{i}", f"Thing {i}", synonyms=[f"item {i}"]))
        onto.add_subsumption(f"A:{i}", "A:ROOT")
    return onto


def test_load_ontology_file_paths(tmp_path):
    from olaya.real_ontology_train import load_ontology_file

    onto = _small_onto()
    assert load_ontology_file(onto) is onto
    with pytest.raises(FileNotFoundError):
        load_ontology_file(tmp_path / "missing.ttl")
    ttl = tmp_path / "x.ttl"
    ttl.write_text(
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n<https://e.org/OBO_1> rdfs:subClassOf <https://e.org/OBO_2> .\n"
    )
    loaded = load_ontology_file(ttl, name="obo")
    assert loaded.name == "obo" and loaded.concepts["OBO:1"].parents == {"OBO:2"}  # Rust CURIE scheme


def test_evidence_objects_and_legacy_samples(tokenizer):
    from olaya.real_ontology_train import (
        _specificity_depth,
        generate_real_ontology_observations,
        generate_real_ontology_samples,
    )

    class Fact:
        def __init__(self, curie, text):
            self.target_curie, self.text = curie, text

    onto = _small_onto()
    rows = generate_real_ontology_observations(onto, "generic", 40, evidence=[Fact("A:2", "a real thing two")])
    assert any(r.source_kind == "real_abox" for r in rows)

    special = {"cls": tokenizer.cls_token_id, "sep": tokenizer.sep_token_id, "mask": tokenizer.mask_token_id, "pad": 0}
    samples = generate_real_ontology_samples(
        onto,
        "generic",
        30,
        tokenizer_encode=lambda t: tokenizer.encode(t, add_special_tokens=False),
        special_ids=special,
        use_real_abox=False,
    )
    assert 0 < len(samples) <= 30
    assert _specificity_depth(onto, "A:3") == 1

    cyclic = _small_onto(2)
    cyclic.concepts["A:ROOT"].parents.add("A:0")
    with pytest.raises(ValueError, match="cycle"):
        _specificity_depth(cyclic, "A:0")
    dangling = _small_onto(2)
    dangling.concepts["A:0"].parents.add("A:GHOST")
    with pytest.raises(ValueError, match="absent"):
        _specificity_depth(dangling, "A:0")
    with pytest.raises(ValueError, match="no concepts with labels"):
        generate_real_ontology_observations(Ontology("empty"), "generic", 10)


def test_expand_observations_rejects_unknown_targets(tokenizer):
    from olaya.evaluation import SourceObservation
    from olaya.real_ontology_train import expand_observations

    row = SourceObservation("s", "g", "A:GHOST", "text", "synthetic_template")
    with pytest.raises(ValueError, match="absent from ontology"):
        expand_observations(
            _small_onto(), [row], "generic", lambda t: tokenizer.encode(t), {"cls": 1, "sep": 2, "mask": 3, "pad": 0}
        )


# --- calibration ------------------------------------------------------------------------------


def test_calibration_edge_cases():
    torch = pytest.importorskip("torch")
    from olaya.calibration import compute_ece, pad_and_cat_logits

    binary = compute_ece(torch.tensor([0.9, 0.2, 0.7, 0.4]), torch.tensor([1, 0, 0, 0]), num_bins=4)
    assert binary.accuracy == pytest.approx(0.75) and "ECE" in binary.summary()
    empty = compute_ece(np.zeros((0, 2)), np.zeros(0))
    assert empty.ece == 0.0 and empty.bin_counts == []

    assert pad_and_cat_logits([]).numel() == 0
    out = pad_and_cat_logits([torch.zeros(2, 3), torch.ones(1, 5)], pad_value=-9.0)
    assert out.shape == (3, 5) and float(out[0, 4]) == -9.0 and float(out[2, 4]) == 1.0


# --- engine error paths -----------------------------------------------------------------------


def test_engine_question_validation(tokenizer):
    pytest.importorskip("torch")
    from olaya.engine import OLaya, _validated_temperature

    engine = OLaya(tokenizer=tokenizer, device="cpu", weights_path=False)
    with pytest.raises(ValueError, match="At least one question"):
        engine.system_one("x", {})
    with pytest.raises(ValueError, match="Unknown question type"):
        engine.system_one("x", {"q": {"type": "nonsense"}})
    with pytest.raises(ValueError, match="rubric"):
        engine.system_one("x", {"q": {"type": "score", "criteria": []}})
    with pytest.raises(ValueError, match="non-numeric temperature"):
        _validated_temperature("hot", "ckpt.pt")


def test_engine_rejects_checkpoints_without_state_dict(tokenizer, tmp_path):
    torch = pytest.importorskip("torch")
    from olaya.engine import OLaya

    bad = tmp_path / "bad.pt"
    torch.save({"config": {"hidden_size": 256}, "model_state_dict": "not a dict"}, bad)
    with pytest.raises((ValueError, TypeError, AttributeError, RuntimeError)):
        OLaya(tokenizer=tokenizer, device="cpu", weights_path=bad)
