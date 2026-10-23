"""Error paths, caches and serialisation details of olaya.ontology."""

from __future__ import annotations

import pytest

from olaya.ontology import Concept, Individual, ObjectProperty, Ontology


def _onto() -> Ontology:
    onto = Ontology("m")
    for curie in ("R", "A", "B", "C", "D"):
        onto.add_concept(Concept(curie, curie))
    onto.add_subsumption("A", "R")
    onto.add_subsumption("B", "R")
    return onto


def test_axiom_builders_reject_unknown_concepts():
    onto = _onto()
    with pytest.raises(KeyError, match="Child"):
        onto.add_subsumption("GHOST", "R")
    with pytest.raises(KeyError, match="Parent concept"):
        onto.add_subsumption("A", "GHOST")
    with pytest.raises(KeyError):
        onto.add_disjointness("A", "GHOST")
    with pytest.raises(KeyError):
        onto.add_equivalence("A", "GHOST")
    with pytest.raises(KeyError, match="Parent GHOST"):
        onto.add_disjoint_union("GHOST", ["A", "B"])
    with pytest.raises(KeyError):
        onto.add_intersection("GHOST", ["A"])
    with pytest.raises(KeyError):
        onto.add_union("GHOST", ["A"])


def test_properties_subproperty_and_chain_errors():
    onto = _onto()
    onto.add_property(ObjectProperty("p", domains={"A"}, ranges={"B"}))
    onto.add_property(ObjectProperty("q"))
    onto.add_subproperty("p", "q")
    assert "q" in onto.properties["p"].parents and "p" in onto.properties["q"].children
    with pytest.raises(KeyError):
        onto.add_subproperty("ghost", "q")
    with pytest.raises(KeyError):
        onto.add_subproperty("p", "ghost")
    with pytest.raises(KeyError, match="Property ghost"):
        onto.add_property_chain("ghost", ["p", "q"])


def test_equivalence_closure_caches_and_unknowns():
    onto = _onto()
    onto.add_equivalence("C", "D")
    onto.concepts["D"].equivalent_to.add("E-missing")  # ignored: not a concept
    assert onto.get_equivalent_concepts("C") == {"C", "D"}
    assert onto.get_equivalent_concepts("C") is onto.get_equivalent_concepts("C")  # cached
    assert onto.get_equivalent_concepts("GHOST") == set()
    assert onto.get_ancestors("GHOST") == set() and onto.get_descendants("GHOST") == set()
    assert onto.get_descendants("R") is onto.get_descendants("R")
    assert onto.get_lineage("GHOST") == ["GHOST"]


def test_concept_and_ontology_round_trip_with_rich_axioms():
    onto = _onto()
    onto.add_equivalence("C", "D")
    onto.add_intersection("C", ["A", "B"])
    onto.add_union("R", ["A", "B"])
    onto.add_disjoint_union("R", ["A", "B"])
    onto.add_property(ObjectProperty("p", domains={"A"}, ranges={"B"}))
    onto.add_property(ObjectProperty("unrelated", domains={"D"}))
    onto.individuals["i1"] = Individual(curie="i1", types={"A"})

    d = onto.concepts["C"].to_dict()
    assert d["equivalent_to"] == ["D"] and d["intersections"] == [["A", "B"]]
    assert onto.concepts["R"].to_dict()["unions"] == [["A", "B"]]

    data = onto.to_dict()
    assert data["disjoint_unions"] == {"R": [["A", "B"]]} and data["properties"] and data["individuals"]
    again = Ontology.from_json(onto.to_json())
    assert again.concepts["C"].equivalent_to == {"D"} and "i1" in again.individuals
    assert again.disjoint_unions == onto.disjoint_unions

    sub = onto.extract_subgraph(["A", "B"])
    assert "p" in sub.properties and "unrelated" not in sub.properties
    assert sub.disjoint_unions == {"R": [["A", "B"]]}
    assert onto.extract_subgraph(["A"]).disjoint_unions == {}  # partitions need >= 2 surviving members
