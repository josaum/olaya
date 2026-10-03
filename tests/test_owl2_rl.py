"""Comprehensive unit tests for W3C OWL 2 RL Reasoner, Deductive Closure, and Axiomatic Projection."""

from olaya.axioms import enforce_owl2_rl_axioms, verify_owl2_rl_axioms
from olaya.ontology import Concept, ObjectProperty, Ontology
from olaya.owl2_rl import OWL2RLReasoner


def test_scm_sco_transitivity():
    """W3C Rule scm-sco: A <= B and B <= C implies A <= C."""
    onto = Ontology("TransitivityTest")
    onto.add_concept(Concept("A", "Class A"))
    onto.add_concept(Concept("B", "Class B"))
    onto.add_concept(Concept("C", "Class C"))
    onto.add_subsumption("A", "B")
    onto.add_subsumption("B", "C")

    closure = OWL2RLReasoner(onto).compute_closure()
    assert closure.is_subclass_of("A", "C"), "A must be inferred as subclass of C"
    assert closure.rule_firings["scm-sco"] >= 1


def test_scm_eqc_equivalence():
    """W3C Rules scm-eqc1 and scm-eqc2: A == B iff A <= B and B <= A."""
    onto = Ontology("EquivalenceTest")
    onto.add_concept(Concept("A", "Class A"))
    onto.add_concept(Concept("B", "Class B"))
    onto.add_equivalence("A", "B")

    closure = OWL2RLReasoner(onto).compute_closure()
    assert closure.are_equivalent("A", "B")
    assert closure.is_subclass_of("A", "B")
    assert closure.is_subclass_of("B", "A")

    # Projection test: P(A) must equal P(B)
    raw_probs = {"A": 0.8, "B": 0.2}
    post = enforce_owl2_rl_axioms(raw_probs, onto, closure=closure)
    assert abs(post["A"] - post["B"]) < 1e-6
    assert abs(post["A"] - 0.5) < 1e-6


def test_scm_dco_disjointness_inheritance():
    """W3C Rule scm-dco: A <= B and disjoint(B, C) implies disjoint(A, C)."""
    onto = Ontology("DisjointInheritanceTest")
    onto.add_concept(Concept("A", "Class A"))
    onto.add_concept(Concept("B", "Class B"))
    onto.add_concept(Concept("C", "Class C"))
    onto.add_subsumption("A", "B")
    onto.add_disjointness("B", "C")

    closure = OWL2RLReasoner(onto).compute_closure()
    assert closure.are_disjoint("A", "C"), "A must inherit disjointness with C from B"
    assert closure.rule_firings["scm-dco"] >= 1


def test_scm_int_intersections():
    """W3C Rule scm-int: C == (C1 and C2) implies C <= C1 and C <= C2."""
    onto = Ontology("IntersectionTest")
    onto.add_concept(Concept("C", "Class C"))
    onto.add_concept(Concept("C1", "Class C1"))
    onto.add_concept(Concept("C2", "Class C2"))
    onto.add_intersection("C", ["C1", "C2"])

    closure = OWL2RLReasoner(onto).compute_closure()
    assert closure.is_subclass_of("C", "C1")
    assert closure.is_subclass_of("C", "C2")

    # Projection test: P(C) <= min(P(C1), P(C2))
    raw_probs = {"C": 0.9, "C1": 0.5, "C2": 0.7}
    post = enforce_owl2_rl_axioms(raw_probs, onto, closure=closure)
    assert post["C"] <= post["C1"] + 1e-6
    assert post["C"] <= post["C2"] + 1e-6


def test_scm_uni_unions():
    """W3C Rule scm-uni: C == (C1 or C2) implies C1 <= C and C2 <= C."""
    onto = Ontology("UnionTest")
    onto.add_concept(Concept("C", "Class C"))
    onto.add_concept(Concept("C1", "Class C1"))
    onto.add_concept(Concept("C2", "Class C2"))
    onto.add_union("C", ["C1", "C2"])

    closure = OWL2RLReasoner(onto).compute_closure()
    assert closure.is_subclass_of("C1", "C")
    assert closure.is_subclass_of("C2", "C")

    # Projection test: P(C) >= P(C1) and P(C) >= P(C2)
    raw_probs = {"C": 0.2, "C1": 0.6, "C2": 0.4}
    post = enforce_owl2_rl_axioms(raw_probs, onto, closure=closure)
    assert post["C"] >= post["C1"] - 1e-6
    assert post["C"] >= post["C2"] - 1e-6


def test_disjoint_union_of():
    """W3C owl:disjointUnionOf: members are pairwise disjoint and subclasses of parent."""
    onto = Ontology("DisjointUnionTest")
    onto.add_concept(Concept("Vehicle", "Vehicle"))
    onto.add_concept(Concept("Car", "Car"))
    onto.add_concept(Concept("Boat", "Boat"))
    onto.add_concept(Concept("Plane", "Plane"))
    onto.add_disjoint_union("Vehicle", ["Car", "Boat", "Plane"])

    closure = OWL2RLReasoner(onto).compute_closure()
    assert closure.is_subclass_of("Car", "Vehicle")
    assert closure.is_subclass_of("Boat", "Vehicle")
    assert closure.is_subclass_of("Plane", "Vehicle")
    assert closure.are_disjoint("Car", "Boat")
    assert closure.are_disjoint("Boat", "Plane")
    assert closure.are_disjoint("Car", "Plane")


def test_property_hierarchy_and_characteristics():
    """W3C Property rules: scm-spo, scm-dom, scm-rng, prp-inv1."""
    onto = Ontology("PropertyTest")
    onto.add_concept(Concept("Person", "Person"))
    onto.add_concept(Concept("Paper", "Paper"))

    onto.add_property(
        ObjectProperty(
            curie="writes",
            domains={"Person"},
            ranges={"Paper"},
            inverses={"writtenBy"},
        )
    )
    onto.add_property(
        ObjectProperty(
            curie="firstAuthorOf",
            parents={"writes"},
        )
    )

    closure = OWL2RLReasoner(onto).compute_closure()
    assert ("writtenBy", "writes") in closure.property_inverses["writtenBy"] or "writes" in closure.property_inverses[
        "writtenBy"
    ]
    # Domain inheritance (firstAuthorOf inherits Person domain)
    assert "Person" in closure.property_domains["firstAuthorOf"]
    # Range inheritance (firstAuthorOf inherits Paper range)
    assert "Paper" in closure.property_ranges["firstAuthorOf"]


def test_unsatisfiability_detection():
    """W3C Rule cls-nothing2: Subclass of two disjoint classes is unsatisfiable (C <= bot)."""
    onto = Ontology("UnsatTest")
    onto.add_concept(Concept("Male", "Male"))
    onto.add_concept(Concept("Female", "Female"))
    onto.add_concept(Concept("ImpossibleIndividual", "Impossible"))
    onto.add_disjointness("Male", "Female")
    onto.add_subsumption("ImpossibleIndividual", "Male")
    onto.add_subsumption("ImpossibleIndividual", "Female")

    closure = OWL2RLReasoner(onto).compute_closure()
    assert "ImpossibleIndividual" in closure.unsatisfiable_classes
    assert not closure.is_consistent

    # Projection must zero unsatisfiable class
    raw_probs = {"Male": 0.3, "Female": 0.4, "ImpossibleIndividual": 0.8}
    post = enforce_owl2_rl_axioms(raw_probs, onto, closure=closure)
    assert post["ImpossibleIndividual"] == 0.0


def test_full_owl2_rl_verification():
    """Test verify_owl2_rl_axioms reporting violations and post-projection clean state."""
    onto = Ontology("VerificationTest")
    onto.add_concept(Concept("Parent", "Parent"))
    onto.add_concept(Concept("Child", "Child"))
    onto.add_concept(Concept("DisjointSibling", "DisjointSibling"))
    onto.add_subsumption("Child", "Parent")
    onto.add_disjointness("Parent", "DisjointSibling")

    # Inconsistent distribution: Child (0.8) > Parent (0.1), and Child active with DisjointSibling (0.7)
    raw_probs = {"Child": 0.8, "Parent": 0.1, "DisjointSibling": 0.7}
    valid, violations = verify_owl2_rl_axioms(raw_probs, onto, threshold=0.5)
    assert not valid
    assert len(violations) >= 2  # Monotonicity + Disjointness collision

    # Enforce projection
    post = enforce_owl2_rl_axioms(raw_probs, onto, threshold=0.5)
    valid_post, violations_post = verify_owl2_rl_axioms(post, onto, threshold=0.5)
    assert valid_post, f"Violations remained after projection: {violations_post}"
    assert len(violations_post) == 0
