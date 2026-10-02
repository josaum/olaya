"""Unit and regression tests for OAEI formal validation benchmark."""

from pathlib import Path
from olaya.benchmarks.oaei import (
    OWLParser,
    OAEIAlignmentParser,
    OAEIBenchmark,
)


def test_owl_parser():
    cmt_file = Path("data/benchmarks/oaei/ontologies/cmt.owl")
    assert cmt_file.exists(), "cmt.owl not found"
    onto = OWLParser.parse(cmt_file)
    assert onto.name == "cmt"
    assert "Paper" in onto.concepts
    assert "Reviewer" in onto.concepts
    assert "Document" in onto.concepts
    # Test subsumption
    assert onto.is_subsumed_by("Paper", "Document")
    assert onto.is_subsumed_by("Meta-Reviewer", "Reviewer")
    # Test disjointness
    assert onto.are_disjoint("Paper", "Review")


def test_oaei_alignment_parser():
    rdf_file = Path("data/benchmarks/oaei/reference_alignments/cmt-ekaw.rdf")
    assert rdf_file.exists(), "cmt-ekaw.rdf not found"
    alignments = OAEIAlignmentParser.parse(rdf_file)
    assert len(alignments) == 11
    entities = [(a.entity1, a.entity2) for a in alignments]
    assert ("Document", "Document") in entities
    assert ("PaperFullVersion", "Regular_Paper") in entities


def test_oaei_benchmark_evaluation():
    bench = OAEIBenchmark(
        "data/benchmarks/oaei/ontologies",
        "data/benchmarks/oaei/reference_alignments",
    )
    res = bench.evaluate_pair(
        "cmt",
        "ekaw",
        "data/benchmarks/oaei/reference_alignments/cmt-ekaw.rdf",
    )
    assert res.evaluated_classes > 0
    assert res.hits_at_1 >= 4
    assert res.hits_at_5 >= 6
    assert res.mrr > 0.5
    # Axiom guarantees
    assert res.post_monotonicity_violations == 0
    assert res.post_disjoint_collisions == 0
