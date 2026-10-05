"""Tests for the OAEI Conference (ra1) alignment benchmark: parsers, scoring, one real pair."""

from __future__ import annotations

from pathlib import Path

import pytest

from olaya.benchmarks.oaei import (
    LexicalMatcher,
    OAEIAlignment,
    OAEIAlignmentParser,
    OAEIConferenceBenchmark,
    OWLParser,
    PairEvaluation,
    StringEquivMatcher,
    aggregate,
    evaluate_alignment,
    levenshtein_similarity,
    select_mutual_best,
    tokenize_identifier,
)
from olaya.ontology import Concept, ObjectProperty, Ontology

FIXTURES = Path(__file__).parent / "fixtures" / "oaei"


@pytest.fixture(scope="module")
def bench() -> OAEIConferenceBenchmark:
    return OAEIConferenceBenchmark(FIXTURES / "ontologies", FIXTURES / "reference_alignments")


# -- parsers -----------------------------------------------------------------


def test_owl_parser_classes_and_axioms():
    onto = OWLParser.parse(FIXTURES / "ontologies" / "cmt.owl")
    assert onto.name == "cmt"
    assert {"Paper", "Reviewer", "Document", "Meta-Reviewer"} <= set(onto.concepts)
    assert onto.is_subsumed_by("Paper", "Document")
    assert onto.is_subsumed_by("Meta-Reviewer", "Reviewer")
    assert onto.are_disjoint("Paper", "Review")
    assert "Thing" not in onto.concepts
    assert len(onto.concepts) == 29


def test_owl_parser_properties_declared_under_characteristic_tags():
    onto = OWLParser.parse(FIXTURES / "ontologies" / "cmt.owl")
    # hasAuthor is declared as owl:FunctionalProperty with rdf:type ObjectProperty;
    # writePaper only as owl:InverseFunctionalProperty nested inside owl:inverseOf.
    assert "hasAuthor" in onto.properties
    assert "writePaper" in onto.properties
    assert onto.properties["hasAuthor"].is_functional
    assert onto.properties["hasAuthor"].metadata["kind"] == "object"
    assert "writePaper" in onto.properties["hasAuthor"].inverses
    assert onto.properties["hasAuthor"].domains == {"Paper"}
    assert onto.properties["hasAuthor"].ranges == {"Author"}
    assert onto.properties["title"].metadata["kind"] == "datatype"
    assert len(onto.properties) == 59


def test_oaei_alignment_parser():
    alignments = OAEIAlignmentParser.parse(FIXTURES / "reference_alignments" / "cmt-ekaw.rdf")
    assert len(alignments) == 11
    entities = {(a.entity1, a.entity2) for a in alignments}
    assert ("Document", "Document") in entities
    assert ("PaperFullVersion", "Regular_Paper") in entities
    assert all(a.relation == "=" and a.measure == 1.0 for a in alignments)


def test_all_reference_entities_resolve(bench: OAEIConferenceBenchmark):
    """Every ra1 correspondence must be classifiable as class/class or property/property."""
    from olaya.benchmarks.oaei import CONFERENCE_PAIRS, correspondence_kind

    counts = {"class": 0, "property": 0, "other": 0}
    for s, t in CONFERENCE_PAIRS:
        src, tgt = bench.ontology(s), bench.ontology(t)
        for c in bench.reference(s, t):
            counts[correspondence_kind(c, src, tgt)] += 1
    assert counts == {"class": 259, "property": 46, "other": 0}


# -- text helpers ---------------------------------------------------------------


def test_tokenize_identifier():
    assert tokenize_identifier("PaperFullVersion") == ["paper", "full", "version"]
    assert tokenize_identifier("Regular_Papers") == ["regular", "paper"]
    assert tokenize_identifier("has_an_email", drop_stop=True) == ["email"]
    assert tokenize_identifier("has_an_email", drop_stop=False) == ["has", "an", "email"]
    assert tokenize_identifier("Categories", use_stem=False) == ["categories"]
    assert tokenize_identifier("Categories") == ["category"]


def test_levenshtein_similarity():
    assert levenshtein_similarity("organizer", "organizer") == 1.0
    assert levenshtein_similarity("organizer", "organiser") == pytest.approx(1 - 1 / 9)
    assert levenshtein_similarity("abc", "") == 0.0


# -- precision / recall on a hand-built alignment ---------------------------------


def _tiny_pair() -> tuple[Ontology, Ontology]:
    src = Ontology("src")
    tgt = Ontology("tgt")
    for c in ("Paper", "Person", "Review"):
        src.add_concept(Concept(curie=c, label=c))
        tgt.add_concept(Concept(curie=c, label=c))
    src.add_property(ObjectProperty(curie="hasAuthor"))
    src.add_property(ObjectProperty(curie="title"))
    tgt.add_property(ObjectProperty(curie="writtenBy"))
    tgt.add_property(ObjectProperty(curie="hasTitle"))
    return src, tgt


def test_evaluate_alignment_counts_per_modality():
    src, tgt = _tiny_pair()
    reference = [
        OAEIAlignment("Paper", "Paper"),
        OAEIAlignment("Person", "Person"),
        OAEIAlignment("hasAuthor", "writtenBy"),
        OAEIAlignment("title", "hasTitle"),
    ]
    system = [
        OAEIAlignment("Paper", "Paper", measure=0.9),  # TP (class)
        OAEIAlignment("Review", "Person", measure=0.4),  # FP (class)
        OAEIAlignment("title", "hasTitle", measure=0.7),  # TP (property)
    ]
    res = evaluate_alignment(system, reference, src, tgt)
    assert (res["M1"].tp, res["M1"].fp, res["M1"].fn) == (1, 1, 1)
    assert (res["M2"].tp, res["M2"].fp, res["M2"].fn) == (1, 0, 1)
    assert (res["M3"].tp, res["M3"].fp, res["M3"].fn) == (2, 1, 2)
    assert res["M1"].precision == pytest.approx(0.5)
    assert res["M1"].recall == pytest.approx(0.5)
    assert res["M1"].f1 == pytest.approx(0.5)
    assert res["M2"].precision == pytest.approx(1.0)
    assert res["M2"].recall == pytest.approx(0.5)
    assert res["M2"].f1 == pytest.approx(2 / 3)
    assert res["M3"].precision == pytest.approx(2 / 3)
    assert res["M3"].recall == pytest.approx(0.5)
    assert res["M3"].f1 == pytest.approx(4 / 7)
    assert res["M3"].f_beta(2.0) == pytest.approx(5 * (2 / 3) * 0.5 / (4 * (2 / 3) + 0.5))


def test_aggregate_micro_and_macro():
    src, tgt = _tiny_pair()
    ref = [OAEIAlignment("Paper", "Paper"), OAEIAlignment("Person", "Person")]
    perfect = evaluate_alignment(ref, ref, src, tgt)
    half = evaluate_alignment(ref[:1], ref, src, tgt)
    evals = [
        PairEvaluation("a-b", "a", "b", 2, 2, perfect),
        PairEvaluation("c-d", "c", "d", 2, 1, half),
    ]
    agg = aggregate(evals)
    assert agg["M1"]["micro"]["tp"] == 3 and agg["M1"]["micro"]["fn"] == 1
    assert agg["M1"]["micro"]["precision"] == 1.0
    assert agg["M1"]["micro"]["recall"] == 0.75
    assert agg["M1"]["macro"]["pairs"] == 2
    assert agg["M1"]["macro"]["recall"] == 0.75
    assert agg["M1"]["macro"]["f1"] == pytest.approx((1.0 + 2 / 3) / 2, abs=1e-4)
    # No property references in either pair: M2 macro skips them, micro is all zero.
    assert agg["M2"]["macro"]["pairs"] == 0
    assert agg["M2"]["micro"]["f1"] == 0.0


def test_select_mutual_best_is_one_to_one_and_deterministic():
    scored = [
        OAEIAlignment("A", "x", measure=0.9),
        OAEIAlignment("A", "y", measure=0.8),
        OAEIAlignment("B", "x", measure=0.85),  # x prefers A and B prefers x: B-x is dropped
        OAEIAlignment("B", "y", measure=0.6),  # y prefers A: not mutual either
        OAEIAlignment("C", "z", measure=0.6),
        OAEIAlignment("D", "z", measure=0.6),  # tie with C-z: broken by name -> C wins
    ]
    kept = select_mutual_best(scored)
    assert [(c.entity1, c.entity2) for c in kept] == [("A", "x"), ("C", "z")]
    assert select_mutual_best(reversed(scored)) == kept
    assert len({c.entity1 for c in kept}) == len(kept) == len({c.entity2 for c in kept})


# -- one real pair ------------------------------------------------------------------


def test_string_equiv_baseline_cmt_ekaw(bench: OAEIConferenceBenchmark):
    res = bench.evaluate_pair(StringEquivMatcher(), "cmt", "ekaw", 1.0)
    assert res.reference_size == 11
    # Exact-name matches on this pair: Document, Person, Paper, Review, ... all correct.
    assert res.results["M1"].precision == 1.0
    assert res.results["M1"].tp >= 5


def test_lexical_matcher_cmt_ekaw_floor(bench: OAEIConferenceBenchmark):
    res = bench.evaluate_pair(LexicalMatcher(), "cmt", "ekaw", 0.5)
    m1, m3 = res.results["M1"], res.results["M3"]
    # Measured on 2026-10-03 (benchmarks/results/oaei-conference-ra1.json):
    # M1 P 0.857 R 0.750 F1 0.800 ; M3 P 0.750 R 0.545 F1 0.632.
    # Floors are deliberately conservative so legitimate matcher changes do not flap the test.
    assert m1.f1 >= 0.70
    assert m3.f1 >= 0.55
    assert res.system_size <= 25  # 1:1 selection on 29 cmt classes + 59 properties cannot explode


def test_threshold_is_monotone(bench: OAEIConferenceBenchmark):
    sizes = [len(bench.align(LexicalMatcher(), "cmt", "ekaw", th)) for th in (0.0, 0.3, 0.5, 0.7, 1.0)]
    assert sizes == sorted(sizes, reverse=True)
    assert all(c.measure >= 0.7 for c in bench.align(LexicalMatcher(), "cmt", "ekaw", 0.7))


# -- embedding / hybrid matchers (stub encoder, no download) ---------------------------


def test_build_entity_texts_describes_classes_and_properties():
    from olaya.benchmarks.oaei import build_entity_texts

    onto = Ontology("o")
    onto.add_concept(Concept(curie="Document", label="Document", metadata={"has_label": True}))
    onto.add_concept(
        Concept(curie="PaperFullVersion", label="", description="The full version of a paper.", metadata={})
    )
    onto.add_subsumption("PaperFullVersion", "Document")
    onto.add_property(ObjectProperty(curie="hasAuthor", domains={"PaperFullVersion"}, ranges={"Document"}))
    texts = {t.name: t for t in build_entity_texts(onto)}
    assert texts["PaperFullVersion"].label == "Paper Full Version"
    assert texts["PaperFullVersion"].full == "Paper Full Version. The full version of a paper. Subclass of Document."
    assert texts["hasAuthor"].kind == "property"
    assert texts["hasAuthor"].full == "has Author. Domain: Paper Full Version. Range: Document."


def test_csls_calibrate_maps_background_to_zero_and_perfect_to_one():
    import numpy as np

    from olaya.benchmarks.oaei import csls_calibrate

    sim = np.array([[1.0, 0.4], [0.4, 0.4]])
    out = csls_calibrate(sim, k=2)  # background of row 0 / col 0 is (1.0 + 0.4) / 2 = 0.7
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 1] <= 0.0  # no better than its neighbourhood
    assert out[0, 1] < out[0, 0]


def test_propagate_structure_only_raises_scores_of_pairs_with_matching_parents():
    from olaya.benchmarks.oaei import propagate_structure

    src, tgt = _tiny_pair()
    src.add_concept(Concept(curie="FullPaper", label="FullPaper"))
    tgt.add_concept(Concept(curie="RegularPaper", label="RegularPaper"))
    src.add_subsumption("FullPaper", "Paper")
    tgt.add_subsumption("RegularPaper", "Paper")
    scores = {("Paper", "Paper"): 1.0, ("FullPaper", "RegularPaper"): 0.4, ("FullPaper", "Review"): 0.4}
    out = propagate_structure(scores, src, tgt, alpha=0.5)
    assert out[("FullPaper", "RegularPaper")] == pytest.approx(0.4 + 0.5 * 0.6)
    assert out[("FullPaper", "Review")] == 0.4  # Review has no parent: untouched
    assert out[("Paper", "Paper")] == 1.0


def test_hybrid_cascade_keeps_lexical_matches_and_adds_embedding_ones():
    from olaya.benchmarks.oaei import EmbeddingMatcherConfig, HybridMatcher, HybridMatcherConfig
    from olaya.embeddings import HashEncoder

    src, tgt = _tiny_pair()
    src.add_concept(Concept(curie="Banquet", label="Banquet", description="conference dinner", metadata={}))
    tgt.add_concept(Concept(curie="Dinner", label="Dinner", description="conference dinner", metadata={}))
    cfg = HybridMatcherConfig(combine="cascade", embedding=EmbeddingMatcherConfig(text="full", calibration="none"))
    scored = HybridMatcher(HashEncoder(), cfg).score_all(src, tgt)
    best = {(c.entity1, c.entity2): c.measure for c in select_mutual_best(scored)}
    assert best[("Paper", "Paper")] >= 0.9  # lexical stage, kept with its lexical score
    assert ("Banquet", "Dinner") in best  # no lexical evidence: supplied by the embedding stage
    # Entities taken by the lexical stage are not offered to the embedding stage.
    assert not any(e1 == "Paper" and e2 != "Paper" for e1, e2 in best)


def test_hybrid_matchers_on_real_pair_with_stub_encoder(bench: OAEIConferenceBenchmark):
    from olaya.benchmarks.oaei import EmbeddingMatcher, HybridMatcher, HybridMatcherConfig
    from olaya.embeddings import HashEncoder

    enc = HashEncoder()
    emb = bench.evaluate_pair(EmbeddingMatcher(enc), "cmt", "ekaw", 0.5)
    assert emb.results["M1"].tp >= 3  # token-hash embeddings still find the exact-name pairs
    for combine in ("sum", "cascade"):
        res = bench.evaluate_pair(HybridMatcher(enc, HybridMatcherConfig(combine=combine)), "cmt", "ekaw", 0.5)
        assert res.results["M1"].f1 >= 0.6
        assert res.system_size <= 25
    # Zero embedding weight reproduces the lexical matcher exactly.
    lex = bench.evaluate_pair(LexicalMatcher(), "cmt", "ekaw", 0.5)
    zero = bench.evaluate_pair(HybridMatcher(enc, HybridMatcherConfig(embedding_weight=0.0)), "cmt", "ekaw", 0.5)
    assert zero.results["M3"].to_dict() == lex.results["M3"].to_dict()


def test_cross_validate_never_selects_on_the_held_out_pair():
    from olaya.benchmarks.oaei import PRF, cross_validate, leave_one_ontology_out_folds, leave_one_pair_out_folds

    # Config "a" is best on pair p1 only, "b" is best everywhere else.
    grid = {
        "a": {"p1": {m: PRF(5, 0, 0) for m in ("M1", "M2", "M3")}, "p2": {m: PRF(1, 4, 4) for m in ("M1", "M2", "M3")}},
        "b": {"p1": {m: PRF(2, 3, 3) for m in ("M1", "M2", "M3")}, "p2": {m: PRF(4, 1, 1) for m in ("M1", "M2", "M3")}},
    }
    cv = cross_validate(grid, leave_one_pair_out_folds(["p1", "p2"]))
    assert cv["chosen"] == {"p1": "b", "p2": "a"}  # each pair scored under the other's choice
    assert (cv["M3"]["tp"], cv["M3"]["fp"], cv["M3"]["fn"]) == (2 + 1, 3 + 4, 3 + 4)
    folds = leave_one_ontology_out_folds()
    assert len(folds) == 7
    assert len(folds["cmt"][1]) == 6 and len(folds["cmt"][0]) == 15
    assert not any("cmt" in p.split("-") for p in folds["cmt"][0])


# -- richer entity texts, ensembles, second-stage fusion, decoders ------------------------


def test_build_entity_texts_rich_fields():
    from olaya.benchmarks.oaei import build_entity_texts, text_fields

    onto = Ontology("o")
    for c in ("Document", "Paper", "Review", "RegularPaper"):
        onto.add_concept(Concept(curie=c, label=c))
    onto.add_subsumption("Paper", "Document")
    onto.add_subsumption("Review", "Document")
    onto.add_subsumption("RegularPaper", "Paper")
    onto.add_property(ObjectProperty(curie="hasAuthor", domains={"Paper"}, ranges={"Document"}))
    texts = {t.name: t for t in build_entity_texts(onto)}
    assert texts["RegularPaper"].path == "Document > Paper > Regular Paper"
    assert texts["RegularPaper"].head == "paper"
    assert texts["Paper"].rich == (
        "Paper. Subclass of Document. Path: Document > Paper. Superclass of Regular Paper. Siblings: Review. "
        "Has properties has Author."
    )
    assert texts["Document"].rich == "Document. Superclass of Paper, Review. Is the value of has Author."
    assert texts["hasAuthor"].path == "Paper > has Author > Document"
    assert texts["hasAuthor"].head == "author"  # "has" is a stop word
    assert texts["hasAuthor"].rich.endswith("An object property.")
    # The original field is unchanged.
    assert texts["RegularPaper"].full == "Regular Paper. Subclass of Paper."
    assert text_fields("both") == ["label", "full"]
    assert text_fields("label+path+head") == ["label", "path", "head"]
    with pytest.raises(ValueError):
        text_fields("label+bogus")


def test_embedding_matcher_text_fields_and_ensemble():
    from olaya.benchmarks.oaei import EmbeddingMatcher, EmbeddingMatcherConfig
    from olaya.embeddings import HashEncoder

    src, tgt = _tiny_pair()
    one = EmbeddingMatcher(HashEncoder(), EmbeddingMatcherConfig(text="label+path", calibration="none"))
    st, tt, sim = one.similarity_matrix(src, tgt)
    assert sim.shape == (len(st), len(tt))
    assert "label+path" in one.name
    # An ensemble of the same encoder twice is the encoder itself (mean of identical matrices).
    two = EmbeddingMatcher(
        [HashEncoder(), HashEncoder()], EmbeddingMatcherConfig(text="label+path", calibration="none")
    )
    _, _, sim2 = two.similarity_matrix(src, tgt)
    assert sim2 == pytest.approx(sim)
    assert two.name.startswith("olaya-embedding[hash-64+hash-64,")
    # A different second encoder changes the matrix.
    three = EmbeddingMatcher(
        [HashEncoder(), HashEncoder(seed=1)], EmbeddingMatcherConfig(text="label", calibration="none")
    )
    _, _, sim3 = three.similarity_matrix(src, tgt)
    assert sim3.shape == sim.shape and not (sim3 == sim).all()


def test_combine_scores_and_cascade_stage2():
    from olaya.benchmarks.oaei import (
        EmbeddingMatcherConfig,
        HybridMatcher,
        HybridMatcherConfig,
        combine_scores,
    )
    from olaya.embeddings import HashEncoder

    assert combine_scores(0.4, 0.2, "max") == 0.4
    assert combine_scores(0.4, 0.2, "mean") == pytest.approx(0.3)
    assert combine_scores(0.4, 0.2, "noisy_or") == pytest.approx(1 - 0.6 * 0.8)
    assert combine_scores(0.4, 0.2, "embedding") == 0.2
    with pytest.raises(ValueError):
        combine_scores(0.4, 0.2, "bogus")
    src, tgt = _tiny_pair()
    src.add_concept(Concept(curie="RegularAuthor", label="Regular Author", metadata={}))
    tgt.add_concept(Concept(curie="Author", label="Author", metadata={}))
    emb = EmbeddingMatcherConfig(text="label", calibration="none")
    only = HybridMatcher(HashEncoder(), HybridMatcherConfig(combine="cascade", embedding=emb))
    fused = HybridMatcher(HashEncoder(), HybridMatcherConfig(combine="cascade", stage2="max", embedding=emb))
    assert "stage2=max" in fused.name and "stage2" not in only.name
    s_only = only.base_scores(src, tgt)[("RegularAuthor", "Author")]
    s_fused = fused.base_scores(src, tgt)[("RegularAuthor", "Author")]
    assert s_fused >= s_only  # max(lex, emb) never lowers a leftover pair


def test_hungarian_and_select_hungarian():
    from olaya.benchmarks.oaei import hungarian, select_hungarian

    # Classic 3x3: optimal assignment cost 5 (0->1, 1->0, 2->2 ... any minimum).
    cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    a = hungarian(cost)
    assert sorted(a) == [0, 1, 2]
    assert sum(cost[i][j] for i, j in enumerate(a)) == 5
    # Rectangular (fewer rows than columns).
    a = hungarian([[1, 9, 9], [9, 9, 1]])
    assert a == [0, 2]
    # Mutual best keeps A-x only (0.9); the assignment prefers A-y + B-x (1.65 > 0.9 + 0.6).
    scored = [
        OAEIAlignment("A", "x", measure=0.9),
        OAEIAlignment("A", "y", measure=0.8),
        OAEIAlignment("B", "x", measure=0.85),
        OAEIAlignment("B", "y", measure=0.6),
    ]
    kept = {(c.entity1, c.entity2) for c in select_hungarian(scored)}
    assert kept == {("A", "y"), ("B", "x")}
    assert {(c.entity1, c.entity2) for c in select_mutual_best(scored)} == {("A", "x")}
    assert select_hungarian([]) == []


def test_margin_and_adaptive_threshold():
    from olaya.benchmarks.oaei import apply_adaptive_threshold, select_mutual_best_margin

    scored = [
        OAEIAlignment("A", "x", measure=0.9),
        OAEIAlignment("A", "y", measure=0.85),  # runner-up of A: margin 0.05
        OAEIAlignment("B", "z", measure=0.4),  # no competition at all
        OAEIAlignment("C", "w", measure=0.3),
        OAEIAlignment("C", "z", measure=0.18),  # runner-up of column z and of row C
    ]
    assert [(c.entity1, c.entity2) for c in select_mutual_best_margin(scored, 0.0)] == [
        ("A", "x"),
        ("B", "z"),
        ("C", "w"),
    ]
    # A-x beats its runner-up by 0.05 only and fails a 0.1 margin; C-w (0.12) and B-z pass.
    assert [(c.entity1, c.entity2) for c in select_mutual_best_margin(scored, 0.1)] == [("B", "z"), ("C", "w")]
    mutual = select_mutual_best(scored)
    # Absolute clause alone (t_rel >= 1): plain threshold.
    assert [(c.entity1, c.entity2) for c in apply_adaptive_threshold(mutual, scored, 0.5, 1.0)] == [("A", "x")]
    # Runner-up clause: B-z's runner-up (0.18, column z) <= 0.5 * 0.4 passes; C-w's 0.18 > 0.5 * 0.3 fails.
    kept = apply_adaptive_threshold(mutual, scored, 0.5, 0.5, mode="runner_up")
    assert [(c.entity1, c.entity2) for c in kept] == [("A", "x"), ("B", "z")]
    # Pair-max clause: 0.4 >= 0.4 * 0.9 passes, 0.3 does not.
    kept = apply_adaptive_threshold(mutual, scored, 0.5, 0.4, mode="pair_max")
    assert [(c.entity1, c.entity2) for c in kept] == [("A", "x"), ("B", "z")]


def test_benchmark_align_selection_rules(bench: OAEIConferenceBenchmark):
    m = LexicalMatcher()
    mutual = bench.align(m, "cmt", "ekaw", 0.5)
    hung = bench.align(m, "cmt", "ekaw", 0.5, selection="hungarian")
    margin = bench.align(m, "cmt", "ekaw", 0.5, selection="margin=0.2")
    adaptive = bench.align(m, "cmt", "ekaw", 0.5, relative=0.5)
    for alignment in (mutual, hung, margin, adaptive):
        assert len({c.entity1 for c in alignment}) == len(alignment) == len({c.entity2 for c in alignment})
    assert set(margin) <= set(mutual)
    assert set(mutual) <= set(adaptive)
    assert bench.evaluate_pair(m, "cmt", "ekaw", 0.5, selection="hungarian").results["M1"].f1 >= 0.6
    with pytest.raises(ValueError):
        bench.align(m, "cmt", "ekaw", 0.5, selection="bogus")


def test_lexical_head_weight_is_off_by_default_and_named():
    from olaya.benchmarks.oaei import LexicalMatcherConfig

    src, tgt = _tiny_pair()
    src.add_concept(Concept(curie="RegularAuthor", label="Regular Author", metadata={}))
    tgt.add_concept(Concept(curie="Author", label="Author", metadata={}))
    plain = {(a.entity1, a.entity2): a.measure for a in LexicalMatcher().score_all(src, tgt)}
    head = LexicalMatcher(LexicalMatcherConfig(head_weight=0.5))
    boosted = {(a.entity1, a.entity2): a.measure for a in head.score_all(src, tgt)}
    assert boosted[("RegularAuthor", "Author")] > plain[("RegularAuthor", "Author")]
    assert head.name == "olaya-lexical[head=0.50]" and LexicalMatcher().name == "olaya-lexical"


def test_error_analysis_records_and_categories(bench: OAEIConferenceBenchmark):
    from olaya.benchmarks.analysis import (
        EntityView,
        ErrorRecord,
        analyse_pair,
        categorise,
        category_counts,
        stage_counts,
    )

    recs = analyse_pair(bench, LexicalMatcher(), "cmt", "ekaw", 0.5)
    by_status = {s: [r for r in recs if r.status == s] for s in ("tp", "fn", "fp")}
    res = bench.evaluate_pair(LexicalMatcher(), "cmt", "ekaw", 0.5).results["M3"]
    assert (len(by_status["tp"]), len(by_status["fn"]), len(by_status["fp"])) == (res.tp, res.fn, res.fp)
    assert all(r.stage in ("cascade", "scoring", "mutual-best", "threshold") for r in by_status["fn"])
    assert all(r.category for r in by_status["fn"] + by_status["fp"])
    assert sum(stage_counts(recs).values()) == res.fn
    assert sum(v["all"] for v in category_counts(recs).values()) == res.fn
    d = by_status["fn"][0].to_dict()
    assert {"pair", "source_label", "target_label", "rank_row", "rank_col", "stage", "category"} <= set(d)

    def rec(a: str, b: str, kind: str = "class") -> ErrorRecord:
        return ErrorRecord("p", "fn", EntityView(a, kind, a, "", []), EntityView(b, kind, b, "", []), 0.0)

    assert categorise(rec("Paper", "Paper")) == "identical label"
    assert categorise(rec("Call for paper", "Call For Papers")) == "inflection / spelling"
    assert categorise(rec("Regular author", "Author")) == "head noun shared, modifier differs"
    assert categorise(rec("Conference_document", "Document")) == "head noun shared, modifier differs"
    assert categorise(rec("Accepted contribution", "Accepted Paper")) == "compound / partial overlap"
    assert categorise(rec("Chairman", "Chair")) == "synonym / paraphrase"
    assert categorise(rec("Paper", "Contribution")) == "synonym / paraphrase"
    assert categorise(rec("PC", "Program Committee")) == "abbreviation"
    assert categorise(rec("hasAuthor", "writtenBy", "property")) == "property naming"
    assert categorise(rec("hasName", "Name of conference", "property")) == "compound / partial overlap"
