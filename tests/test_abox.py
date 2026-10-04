"""ABox (instance-level) OWL 2 RL rules, sameAs merging, property chains, consistency, serialisation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from olaya.ontology import Concept, Individual, ObjectProperty, Ontology
from olaya.owl2_rl import Inconsistency, OWL2RLReasoner, abox_consistency_report

DATA_DIR = Path(__file__).resolve().parents[1] / "src" / "olaya" / "data"


def _tbox(*classes: str, props: dict[str, ObjectProperty] | None = None) -> Ontology:
    onto = Ontology("abox-test")
    for c in classes:
        onto.add_concept(Concept(c, c))
    for p in (props or {}).values():
        onto.add_property(p)
    return onto


def _closure(onto: Ontology):
    return OWL2RLReasoner(onto).compute_closure()


# ---------------------------------------------------------------------------
# Class rules
# ---------------------------------------------------------------------------


def test_cax_sco_type_inheritance():
    onto = _tbox("A", "B", "C")
    onto.add_subsumption("A", "B")
    onto.add_subsumption("B", "C")
    onto.assert_type("x", "A")
    cl = _closure(onto)
    assert cl.types_of("x") == {"A", "B", "C"}
    assert cl.rule_firings["cax-sco"] >= 2
    assert onto.instances_of("C", cl) == {"x"}
    assert onto.instances_of("C") == {"x"}  # DAG-only fallback
    assert onto.instances_of("A") == {"x"}
    assert cl.is_consistent


def test_cax_eqc_equivalent_classes_share_instances():
    onto = _tbox("A", "B")
    onto.add_equivalence("A", "B")
    onto.assert_type("x", "A")
    onto.assert_type("y", "B")
    cl = _closure(onto)
    assert cl.types_of("x") == {"A", "B"}
    assert cl.types_of("y") == {"A", "B"}
    assert cl.rule_firings["cax-eqc1"] >= 1
    assert cl.rule_firings["cax-eqc2"] >= 1


def test_cls_int1_intersection_membership():
    onto = _tbox("C", "C1", "C2")
    onto.add_intersection("C", ["C1", "C2"])
    onto.assert_type("x", "C1")
    onto.assert_type("x", "C2")
    onto.assert_type("y", "C1")
    cl = _closure(onto)
    assert "C" in cl.types_of("x")
    assert "C" not in cl.types_of("y")
    assert cl.rule_firings["cls-int1"] == 1


def test_cax_dw_disjoint_classes_inconsistent():
    onto = _tbox("Male", "Female", "Boy")
    onto.add_disjointness("Male", "Female")
    onto.add_subsumption("Boy", "Male")
    onto.assert_type("pat", "Boy")
    onto.assert_type("pat", "Female")
    cl = _closure(onto)
    assert not cl.is_consistent
    report = abox_consistency_report(onto, cl)
    assert report == cl.inconsistencies
    rules = {inc.rule for inc in report}
    assert rules == {"cax-dw"}
    entities = {inc.entities for inc in report}
    assert ("pat", "Female", "Male") in entities  # inherited disjointness via scm-dco + cax-sco
    assert cl.rule_firings["cax-dw"] == len(report)
    assert all(isinstance(inc, Inconsistency) for inc in report)
    assert report[0].to_dict()["rule"] == "cax-dw"


def test_cls_nothing2_instance_of_unsatisfiable_class():
    onto = _tbox("A", "B", "Bad")
    onto.add_disjointness("A", "B")
    onto.add_subsumption("Bad", "A")
    onto.add_subsumption("Bad", "B")
    onto.assert_type("x", "Bad")
    cl = _closure(onto)
    assert "Bad" in cl.unsatisfiable_classes
    assert any(inc.rule == "cls-nothing2" and inc.entities == ("x", "Bad") for inc in cl.inconsistencies)


def test_consistent_abox_has_empty_report():
    onto = _tbox("A", "B")
    onto.add_disjointness("A", "B")
    onto.assert_type("x", "A")
    onto.assert_type("y", "B")
    assert abox_consistency_report(onto) == []
    assert _closure(onto).is_consistent


# ---------------------------------------------------------------------------
# Property rules
# ---------------------------------------------------------------------------


def test_prp_dom_rng():
    onto = _tbox("Person", "Paper", props={"writes": ObjectProperty("writes", domains={"Person"}, ranges={"Paper"})})
    onto.assert_property("alice", "writes", "p1")
    cl = _closure(onto)
    assert cl.types_of("alice") == {"Person"}
    assert cl.types_of("p1") == {"Paper"}
    assert cl.rule_firings["prp-dom"] == 1
    assert cl.rule_firings["prp-rng"] == 1
    assert onto.instances_of("Paper", cl) == {"p1"}


def test_prp_spo1_and_inherited_domain():
    onto = _tbox(
        "Person",
        props={
            "writes": ObjectProperty("writes", domains={"Person"}),
            "firstAuthorOf": ObjectProperty("firstAuthorOf", parents={"writes"}),
        },
    )
    onto.assert_property("alice", "firstAuthorOf", "p1")
    cl = _closure(onto)
    assert cl.has_assertion("alice", "writes", "p1")
    assert cl.rule_firings["prp-spo1"] == 1
    assert "Person" in cl.types_of("alice")


def test_prp_inv1_inv2():
    onto = _tbox(
        props={"hasParent": ObjectProperty("hasParent", inverses={"hasChild"}), "hasChild": ObjectProperty("hasChild")}
    )
    onto.assert_property("bob", "hasParent", "ann")
    onto.assert_property("carl", "hasChild", "dee")
    cl = _closure(onto)
    assert cl.has_assertion("ann", "hasChild", "bob")
    assert cl.has_assertion("dee", "hasParent", "carl")
    assert cl.rule_firings["prp-inv1"] == 1
    assert cl.rule_firings["prp-inv2"] == 1


def test_prp_symp():
    onto = _tbox(props={"knows": ObjectProperty("knows", is_symmetric=True)})
    onto.assert_property("a", "knows", "b")
    cl = _closure(onto)
    assert cl.has_assertion("b", "knows", "a")
    assert cl.rule_firings["prp-symp"] == 1


def test_prp_trp():
    onto = _tbox(props={"partOf": ObjectProperty("partOf", is_transitive=True)})
    onto.assert_property("a", "partOf", "b")
    onto.assert_property("b", "partOf", "c")
    onto.assert_property("c", "partOf", "d")
    cl = _closure(onto)
    assert cl.objects_of("a", "partOf") == {"b", "c", "d"}
    assert cl.rule_firings["prp-trp"] >= 3


def test_prp_spo2_property_chain():
    props = {p: ObjectProperty(p) for p in ("hasParent", "hasBrother", "hasUncle")}
    onto = _tbox(props=props)
    onto.add_property_chain("hasUncle", ["hasParent", "hasBrother"])
    onto.assert_property("kid", "hasParent", "mum")
    onto.assert_property("mum", "hasBrother", "unc")
    onto.assert_property("kid", "hasBrother", "sib")  # not a chain match
    cl = _closure(onto)
    assert cl.has_assertion("kid", "hasUncle", "unc")
    assert cl.objects_of("kid", "hasUncle") == {"unc"}
    assert cl.rule_firings["prp-spo2"] == 1
    with pytest.raises(ValueError):
        onto.add_property_chain("hasUncle", ["hasParent"])
    with pytest.raises(KeyError):
        onto.add_property_chain("hasUncle", ["hasParent", "nope"])


def test_prp_spo2_three_link_chain_with_transitivity_interplay():
    props = {p: ObjectProperty(p) for p in ("p", "q", "r", "s")}
    props["p"].is_transitive = True
    onto = _tbox(props=props)
    onto.add_property_chain("s", ["p", "q", "r"])
    onto.assert_property("a", "p", "b")
    onto.assert_property("b", "p", "c")  # a p c by prp-trp, then a s e via chain
    onto.assert_property("c", "q", "d")
    onto.assert_property("d", "r", "e")
    cl = _closure(onto)
    assert cl.has_assertion("a", "s", "e")
    assert cl.has_assertion("b", "s", "e")


def test_prp_fp_functional_sameas_merge():
    onto = _tbox("Person", props={"hasMother": ObjectProperty("hasMother", is_functional=True, ranges={"Person"})})
    onto.assert_property("kid", "hasMother", "m1")
    onto.assert_property("kid", "hasMother", "m2")
    onto.assert_type("m1", "Person")
    onto.assert_property("m2", "hasMother", "gran")
    cl = _closure(onto)
    assert cl.are_same("m1", "m2")
    assert cl.same_individuals("m1") == frozenset({"m1", "m2"})
    assert cl.rule_firings["prp-fp"] == 1
    # eq-rep-s: facts of m2 flow to m1 and vice versa
    assert cl.has_assertion("m1", "hasMother", "gran")
    assert "Person" in cl.types_of("m2")
    # eq-rep-o: kid hasMother both (already asserted) stays; gran range typing
    assert "Person" in cl.types_of("gran")
    assert cl.rule_firings["eq-rep-s"] >= 1
    assert cl.is_consistent


def test_prp_ifp_inverse_functional_sameas_merge_and_eq_rep_o():
    onto = _tbox(
        props={"hasSSN": ObjectProperty("hasSSN", is_inverse_functional=True), "likes": ObjectProperty("likes")}
    )
    onto.assert_property("p1", "hasSSN", "ssn42")
    onto.assert_property("p2", "hasSSN", "ssn42")
    onto.assert_property("fan", "likes", "p1")
    cl = _closure(onto)
    assert cl.are_same("p1", "p2")
    assert cl.rule_firings["prp-ifp"] == 1
    assert cl.has_assertion("fan", "likes", "p2")
    assert cl.rule_firings["eq-rep-o"] >= 1


def test_eq_sym_trans_through_asserted_sameas():
    onto = _tbox("A")
    onto.assert_same_as("a", "b")
    onto.assert_same_as("b", "c")
    onto.assert_type("c", "A")
    cl = _closure(onto)
    assert cl.same_individuals("a") == frozenset({"a", "b", "c"})
    assert cl.are_same("c", "a")
    assert cl.types_of("a") == {"A"}
    assert cl.rule_firings["eq-sym"] >= 2
    assert cl.rule_firings["eq-trans"] >= 1
    assert cl.same_individuals("unknown") == frozenset({"unknown"})


def test_eq_diff1_sameas_vs_differentfrom():
    onto = _tbox(props={"id": ObjectProperty("id", is_inverse_functional=True)})
    onto.assert_property("x", "id", "k")
    onto.assert_property("y", "id", "k")
    onto.assert_different_from("x", "y")
    cl = _closure(onto)
    assert not cl.is_consistent
    assert [inc.rule for inc in cl.inconsistencies] == ["eq-diff1"]
    assert cl.inconsistencies[0].entities == ("x", "y")


def test_prp_pdw_disjoint_properties():
    onto = _tbox(props={"likes": ObjectProperty("likes"), "hates": ObjectProperty("hates")})
    onto.add_property_disjointness("likes", "hates")
    onto.assert_property("a", "likes", "b")
    onto.assert_property("a", "hates", "b")
    onto.assert_property("a", "likes", "c")
    cl = _closure(onto)
    assert [inc.rule for inc in cl.inconsistencies] == ["prp-pdw"]
    assert cl.inconsistencies[0].entities == ("a", "hates", "likes", "b")
    assert cl.rule_firings["prp-pdw"] == 1


def test_prp_asyp_and_prp_irp():
    onto = _tbox(
        props={"beats": ObjectProperty("beats", is_asymmetric=True), "ne": ObjectProperty("ne", is_irreflexive=True)}
    )
    onto.assert_property("a", "beats", "b")
    onto.assert_property("b", "beats", "a")
    onto.assert_property("z", "ne", "z")
    cl = _closure(onto)
    rules = sorted(inc.rule for inc in cl.inconsistencies)
    assert rules == ["prp-asyp", "prp-irp"]


def test_rule_firings_are_counted_for_every_instance_rule():
    """One ontology exercising all instance rules; each must register at least one firing."""
    props = {
        "p": ObjectProperty("p", domains={"D"}, ranges={"R"}, inverses={"pinv"}, is_symmetric=False),
        "pinv": ObjectProperty("pinv"),
        "sub": ObjectProperty("sub", parents={"p"}),
        "sym": ObjectProperty("sym", is_symmetric=True),
        "trp": ObjectProperty("trp", is_transitive=True),
        "fp": ObjectProperty("fp", is_functional=True),
        "ifp": ObjectProperty("ifp", is_inverse_functional=True),
        "chain": ObjectProperty("chain"),
    }
    onto = _tbox("D", "R", "E", "G", "I1", "I2", props=props)
    onto.add_subsumption("D", "E")
    onto.add_subsumption("D", "G")  # reached only through cax-sco
    onto.add_equivalence("R", "E")
    onto.add_intersection("I1", ["D", "R"])
    onto.add_property_chain("chain", ["p", "sub"])
    onto.assert_property("a", "sub", "b")
    onto.assert_property("b", "sub", "c")
    onto.assert_property("a", "sym", "b")
    onto.assert_property("a", "trp", "b")
    onto.assert_property("b", "trp", "c")
    onto.assert_property("a", "fp", "f1")
    onto.assert_property("a", "fp", "f2")
    onto.assert_property("g1", "ifp", "k")
    onto.assert_property("g2", "ifp", "k")
    onto.assert_property("c", "pinv", "d")  # d p c only through prp-inv2
    onto.assert_type("a", "R")
    onto.assert_type("z", "E")  # E -> R is cax-eqc1 (lexicographic direction), R -> E is cax-eqc2
    onto.assert_type("f1", "G")  # replicated onto f2 by eq-rep-s once prp-fp merges them
    cl = _closure(onto)
    for rule in [
        "cax-sco",
        "cax-eqc1",
        "cax-eqc2",
        "cls-int1",
        "prp-dom",
        "prp-rng",
        "prp-spo1",
        "prp-spo2",
        "prp-inv1",
        "prp-inv2",
        "prp-symp",
        "prp-trp",
        "prp-fp",
        "prp-ifp",
        "eq-sym",
        "eq-rep-s",
    ]:
        assert cl.rule_firings[rule] >= 1, rule
    assert cl.is_consistent
    assert cl.has_assertion("a", "chain", "c")
    assert cl.has_assertion("b", "pinv", "a")  # prp-spo1 then prp-inv1
    assert "I1" in cl.types_of("a")  # D via prp-dom, R asserted -> intersection
    assert "G" in cl.types_of("a")
    assert cl.has_assertion("d", "p", "c")
    assert "G" in cl.types_of("f2")


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_abox_json_round_trip():
    onto = _tbox(
        "Person",
        props={"knows": ObjectProperty("knows", is_symmetric=True), "chainTarget": ObjectProperty("chainTarget")},
    )
    onto.add_property_chain("chainTarget", ["knows", "knows"])
    onto.add_property_disjointness("knows", "chainTarget")
    onto.add_individual(Individual("alice", "Alice", types={"Person"}, metadata={"source": "test"}))
    onto.assert_property("alice", "knows", "bob")
    onto.assert_same_as("bob", "robert")
    onto.assert_different_from("alice", "bob")

    data = json.loads(onto.to_json())
    assert "individuals" in data
    prop_row = next(p for p in data["properties"] if p["curie"] == "chainTarget")
    assert prop_row["property_chains"] == [["knows", "knows"]]
    assert "property_chains" not in next(p for p in data["properties"] if p["curie"] == "knows")

    back = Ontology.from_json(json.dumps(data))
    assert set(back.individuals) == {"alice", "bob", "robert"}
    assert back.individuals["alice"].types == {"Person"}
    assert back.individuals["alice"].properties == {"knows": {"bob"}}
    assert back.individuals["alice"].metadata == {"source": "test"}
    assert back.individuals["alice"].different_from == {"bob"}
    assert back.individuals["bob"].same_as == {"robert"}
    assert back.properties["chainTarget"].property_chains == [["knows", "knows"]]
    assert back.properties["knows"].disjoint_with == {"chainTarget"}
    assert back.to_dict() == onto.to_dict()
    assert onto.property_assertions() == {("alice", "knows", "bob")}
    assert _closure(back).same_individuals("robert") == frozenset({"bob", "robert"})


def test_tbox_only_serialisation_unchanged():
    onto = _tbox("A")
    assert "individuals" not in onto.to_dict()
    assert Ontology.from_json(onto.to_json()).individuals == {}


def test_assertions_validate_schema_references():
    onto = _tbox("A")
    with pytest.raises(KeyError):
        onto.assert_type("x", "Missing")
    with pytest.raises(KeyError):
        onto.assert_property("x", "missingProp", "y")
    with pytest.raises(KeyError):
        onto.add_property_disjointness("p", "q")


# ---------------------------------------------------------------------------
# Regression: schema closure of the bundled ontologies is byte-for-byte unchanged
# ---------------------------------------------------------------------------

# Regenerate with:
#   .venv/bin/python -c "import tests.test_abox as t; t.print_schema_closure_hashes()"
# These values were computed with the pre-ABox reasoner (commit 71c1ec8) and must not change
# unless the schema rules themselves change deliberately.
EXPECTED_SCHEMA_CLOSURE_HASHES = {
    "bfo_ontology.json": "c263d7354974eee509b37c98cf4b907782431b21b988fe339df819205828fbe0",
    "doid_ontology.json": "43136395700a07ab3a154dde4f172fb70cfa8ea35aacbd56cdf18533403b00a9",
    "gs1_ontology.json": "ad34eefce8895c2a28d40d2cba213177a7fe1c93b53e327745f8d5ee8632e798",
    "incident_ontology.json": "99d5ebbc2046cecaa819e3fc2a67dc940295112cf32faa9ea3e0fb41a3de75a7",
    "leio_ontology.json": "cf8ea3d6318c72d437859c8f41124eb6c00d6414a67a11825155c42a10542755",
    "schemaorg_ontology.json": "9450b60fe2403c17eef4eb3d94614ec52d50e7506389ab2d353ff3ee7f819d8b",
    "skos_ontology.json": "58e85fcc9302dc96d40cac49c8eec8e2e0a98a424739844595cb75d5b8e8a117",
}


def schema_closure_hash(closure) -> str:
    payload = {
        "subclasses": sorted(closure.subclasses),
        "equivalences": sorted(closure.equivalences),
        "disjointnesses": sorted(closure.disjointnesses),
        "unsatisfiable_classes": sorted(closure.unsatisfiable_classes),
        "property_subsumptions": sorted(closure.property_subsumptions),
        "property_domains": {k: sorted(v) for k, v in sorted(closure.property_domains.items()) if v},
        "property_ranges": {k: sorted(v) for k, v in sorted(closure.property_ranges.items()) if v},
        "property_inverses": {k: sorted(v) for k, v in sorted(closure.property_inverses.items()) if v},
        "is_consistent": closure.is_consistent,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def print_schema_closure_hashes() -> None:
    for name in sorted(p.name for p in DATA_DIR.glob("*.json")):
        onto = Ontology.from_json((DATA_DIR / name).read_text())
        print(f'    "{name}": "{schema_closure_hash(OWL2RLReasoner(onto).compute_closure())}",')


@pytest.mark.parametrize("name", sorted(EXPECTED_SCHEMA_CLOSURE_HASHES))
def test_schema_closure_regression(name: str):
    onto = Ontology.from_json((DATA_DIR / name).read_text())
    assert onto.individuals == {}
    closure = OWL2RLReasoner(onto).compute_closure()
    assert schema_closure_hash(closure) == EXPECTED_SCHEMA_CLOSURE_HASHES[name]
    assert closure.inconsistencies == []
    assert closure.property_assertions == set()


def test_bundled_data_files_are_all_covered():
    assert sorted(p.name for p in DATA_DIR.glob("*.json")) == sorted(EXPECTED_SCHEMA_CLOSURE_HASHES)
