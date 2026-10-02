"""Unit tests for HardNegativeSampler."""

import random
import pytest

from olaya.hard_negative_sampler import HardNegativeSampler
from olaya.ontology import Concept, Ontology, create_incident_ontology


def test_sampler_sibling_cousin_detection():
    # Build a small custom DAG:
    #         Root
    #        /    \
    #     ParentA  ParentB
    #     /    \      |
    #   Child1 Child2 Child3
    onto = Ontology()
    onto.add_concept(Concept(curie="onto:Root", label="Root", description="Root class"))
    onto.add_concept(Concept(curie="onto:ParentA", label="Parent A", description="Parent A", parents={"onto:Root"}))
    onto.add_concept(Concept(curie="onto:ParentB", label="Parent B", description="Parent B", parents={"onto:Root"}))
    onto.add_concept(Concept(curie="onto:Child1", label="Child 1", description="Child 1", parents={"onto:ParentA"}))
    onto.add_concept(Concept(curie="onto:Child2", label="Child 2", description="Child 2", parents={"onto:ParentA"}))
    onto.add_concept(Concept(curie="onto:Child3", label="Child 3", description="Child 3", parents={"onto:ParentB"}))

    sampler = HardNegativeSampler(onto, sibling_ratio=0.5, cousin_ratio=0.5)

    # Siblings of Child1 should be Child2
    siblings = sampler.get_siblings("onto:Child1")
    assert siblings == ["onto:Child2"]

    # Cousins of Child1 should be Child3 (shares grandparent Root)
    cousins = sampler.get_cousins("onto:Child1")
    assert cousins == ["onto:Child3"]

    # Sample distractors for Child1
    rng = random.Random(42)
    distractors = sampler.sample_distractors("onto:Child1", k=2, rng=rng)
    assert len(distractors) == 2
    # Neither Child1 nor ParentA nor Root can be in distractors
    assert "onto:Child1" not in distractors
    assert "onto:ParentA" not in distractors
    assert "onto:Root" not in distractors
    # Must be drawn from Child2, Child3, ParentB
    assert set(distractors).issubset({"onto:Child2", "onto:Child3", "onto:ParentB"})


def test_sampler_on_incident_ontology():
    onto = create_incident_ontology()
    sampler = HardNegativeSampler(onto, sibling_ratio=0.5, cousin_ratio=0.3)

    # Pick a leaf concept
    target = "INCIDENT:DB_OUTAGE"
    assert target in onto.concepts

    ancestors = onto.get_ancestors(target)
    descendants = onto.get_descendants(target)

    rng = random.Random(123)
    k = 4
    distractors = sampler.sample_distractors(target, k=k, rng=rng)

    assert len(distractors) == k
    assert target not in distractors
    for d in distractors:
        assert d not in ancestors, f"Distractor {d} is an ancestor of {target}"
        assert d not in descendants, f"Distractor {d} is a descendant of {target}"


def test_sampler_edge_cases():
    onto = Ontology()
    onto.add_concept(Concept(curie="onto:Single", label="Single", description="Single class"))
    sampler = HardNegativeSampler(onto)

    # Single concept ontology has no distractors
    rng = random.Random(42)
    distractors = sampler.sample_distractors("onto:Single", k=5, rng=rng)
    assert distractors == []
