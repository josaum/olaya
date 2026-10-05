"""Tests for the cross-encoder reranker: pair building, fusion, CV folds (stubs, no model).

The model-backed tests (loading / scoring / a two-step fine-tune and the vendored
anatomy fixture) only run with ``OLAYA_RUN_RERANKER_TESTS=1`` because they need the
Hugging Face checkpoint on disk; nothing here downloads in CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from olaya.benchmarks.oaei import PRF, OAEIAlignment, cross_validate, leave_one_pair_out_folds, select_config
from olaya.benchmarks.reranker import (
    Pair,
    auc,
    build_pairs,
    candidate_pairs,
    config_label,
    fuse,
    fuse_scores,
    hits_at_1,
    pair_text,
    pairs_digest,
    select,
    split_by_source,
    split_sources,
    topk_from_matrix,
    validation_report,
)

FIXTURES = Path(__file__).parent / "fixtures" / "oaei"
RUN_MODEL = os.environ.get("OLAYA_RUN_RERANKER_TESTS") == "1"


# -- candidates and pairs --------------------------------------------------------


def _scores() -> dict[tuple[str, str], float]:
    return {
        ("a", "x"): 0.9,
        ("a", "y"): 0.4,
        ("a", "z"): 0.3,
        ("b", "x"): 0.5,
        ("b", "y"): 0.8,
        ("c", "z"): 0.2,
        ("c", "y"): 0.2,  # tie with c-z: broken by name
    }


def test_candidate_pairs_topk_per_source_and_target():
    cands = candidate_pairs(_scores(), 1)
    # per source: a-x, b-y, c-y (tie broken by target name); per target: x<-a, y<-b, z<-a
    assert cands == {("a", "x"), ("b", "y"), ("c", "y"), ("a", "z")}
    assert candidate_pairs(_scores(), 0) == set()
    assert candidate_pairs(_scores(), 10) == set(_scores())


def test_topk_from_matrix_matches_dict_candidates():
    rows, cols = ["a", "b", "c"], ["x", "y", "z"]
    sim = np.zeros((3, 3))
    for (r, c), v in _scores().items():
        sim[rows.index(r), cols.index(c)] = v
    sparse = topk_from_matrix(sim, rows, cols, 1)
    dense = {p: v for p, v in _scores().items()}
    dense.update({(r, c): 0.0 for r in rows for c in cols if (r, c) not in dense})
    assert set(sparse) >= candidate_pairs(dense, 1)
    assert all(abs(sparse[p] - dense[p]) < 1e-9 for p in sparse)


def test_build_pairs_positives_and_hard_negatives():
    ref = [OAEIAlignment("a", "x"), OAEIAlignment("b", "x"), OAEIAlignment("c", "w")]  # c-w not a candidate
    pairs = build_pairs(ref, _scores(), 1)
    by_label: dict[int, set[tuple[str, str]]] = {0: set(), 1: set()}
    for p in pairs:
        by_label[p.label].add((p.source, p.target))
    assert by_label[1] == {("a", "x"), ("b", "x"), ("c", "w")}
    assert by_label[0] == {("b", "y"), ("c", "y"), ("a", "z")}  # candidates that are not in the reference
    assert next(p for p in pairs if p.source == "c" and p.target == "w").base == 0.0
    assert next(p for p in pairs if p.source == "a" and p.target == "x").base == 0.9
    # restricted to a split of the source entities
    only_a = build_pairs(ref, _scores(), 1, sources={"a"})
    assert {(p.source, p.target, p.label) for p in only_a} == {("a", "x", 1), ("a", "z", 0)}
    assert build_pairs(ref, _scores(), 1, include_missing_positives=False, sources={"c"}) == [Pair("c", "y", 0, 0.2)]


def test_split_sources_is_deterministic_and_disjoint():
    names = [f"e{i}" for i in range(50)]
    tr, va = split_sources(names, fraction=0.8, seed=0)
    assert tr.isdisjoint(va) and tr | va == set(names) and len(tr) == 40
    assert split_sources(names, fraction=0.8, seed=0) == (tr, va)
    assert split_sources(names, fraction=0.8, seed=1) != (tr, va)
    ref = [OAEIAlignment(n, "t") for n in names]
    assert split_by_source(ref, fraction=0.8, seed=0) == (tr, va)


def test_pairs_digest_is_order_independent():
    a = [Pair("a", "x", 1), Pair("b", "y", 0)]
    assert pairs_digest(a) == pairs_digest(list(reversed(a)))
    assert pairs_digest(a) != pairs_digest([Pair("a", "x", 0), Pair("b", "y", 0)])


def test_pair_text_normalises_underscores():
    assert pair_text("Spermatic_Cord. Subclass of  Organ.") == "Spermatic Cord. Subclass of Organ."


# -- fusion and selection -------------------------------------------------------


def test_fuse_math():
    assert fuse(0.2, 0.8, 0.0) == pytest.approx(0.2)
    assert fuse(0.2, 0.8, 1.0) == pytest.approx(0.8)
    assert fuse(0.2, 0.8, 0.5) == pytest.approx(0.5)
    fused = fuse_scores({("a", "x"): 0.2, ("a", "y"): 0.9}, {("a", "x"): 0.8}, 0.5)
    assert fused == {("a", "x"): pytest.approx(0.5)}  # non-candidates are dropped
    assert fuse_scores({}, {("a", "x"): 0.8}, 0.5) == {("a", "x"): pytest.approx(0.4)}


def test_select_is_mutual_best_then_threshold():
    fused = {("a", "x"): 0.9, ("a", "y"): 0.7, ("b", "y"): 0.6, ("b", "x"): 0.95, ("c", "z"): 0.3}
    chosen = {(c.entity1, c.entity2) for c in select(fused, 0.5)}
    # b-x is mutual best; a's best (x) is taken, and mutual-best is strict (not greedy), so a-y is not kept;
    # c-z is below the threshold.
    assert chosen == {("b", "x")}
    assert select(fused, 0.99) == []


def test_config_label():
    assert config_label("cascade", 5, 0.5, 0.5) == "base=cascade,k=5,w=0.50,th=0.50"


# -- ranking metrics ---------------------------------------------------------------


def test_auc_and_hits():
    assert auc([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]) == 1.0
    assert auc([1, 0], [0.5, 0.5]) == 0.5
    assert auc([1, 1], [0.5, 0.1]) == 0.5
    assert auc([0, 1], [0.9, 0.1]) == 0.0
    pairs = [Pair("a", "x", 1), Pair("a", "y", 0), Pair("b", "x", 0), Pair("b", "y", 1), Pair("c", "z", 0)]
    assert hits_at_1(pairs, [0.9, 0.1, 0.9, 0.1, 0.5]) == 0.5  # a right, b wrong, c has no positive
    rep = validation_report(pairs, [0.9, 0.1, 0.1, 0.9, 0.5], weights=(0.5,))
    assert rep["positives"] == 2 and rep["negatives"] == 3
    assert rep["cross_encoder"] == {"auc": 1.0, "hits@1": 1.0}
    assert "w=0.50" in rep["fused"]


# -- cross-validation folds ---------------------------------------------------------


def _grid_entry(tp: int, fp: int, fn: int) -> dict[str, PRF]:
    return {m: PRF(tp, fp, fn) for m in ("M1", "M2", "M3")}


def test_leave_one_pair_out_never_selects_on_the_held_out_pair():
    pairs = ["p1", "p2", "p3"]
    # config A is solid everywhere; config B is poor on p1, p2 and overwhelming on p3 only
    grid = {
        "A": {"p1": _grid_entry(8, 2, 2), "p2": _grid_entry(8, 2, 2), "p3": _grid_entry(8, 2, 2)},
        "B": {"p1": _grid_entry(2, 8, 8), "p2": _grid_entry(2, 8, 8), "p3": _grid_entry(100, 0, 0)},
    }
    folds = leave_one_pair_out_folds(pairs)
    assert folds["p3"] == (["p1", "p2"], ["p3"])
    assert select_config(grid, ["p1", "p2"]) == "A"
    assert select_config(grid, ["p1", "p2", "p3"]) == "B"  # optimistic selection on everything picks B
    cv = cross_validate(grid, folds)
    assert cv["chosen"]["p3"] == "A"  # p3 cannot pick B for itself: B is chosen only when p3 is in the training pairs
    assert cv["chosen"]["p1"] == "B" and cv["chosen"]["p2"] == "B"
    # totals: B on p1 and p2 (2, 8, 8 each) + A on p3 (8, 2, 2)
    assert (cv["M3"]["tp"], cv["M3"]["fp"], cv["M3"]["fn"]) == (12, 18, 18)


# -- model-backed tests (opt-in) --------------------------------------------------------


@pytest.mark.skipif(not RUN_MODEL, reason="set OLAYA_RUN_RERANKER_TESTS=1 (needs the HF checkpoint on disk)")
def test_anatomy_fixture_loads():
    from olaya.benchmarks.reranker import load_anatomy

    data = load_anatomy(FIXTURES / "anatomy")
    assert len(data.reference) == 1516
    assert len(data.mouse.concepts) > 2700 and len(data.human.concepts) > 3300
    assert all(a.entity1.startswith("MA_") and a.entity2.startswith("NCI_") for a in data.reference)


@pytest.mark.skipif(not RUN_MODEL, reason="set OLAYA_RUN_RERANKER_TESTS=1 (needs the HF checkpoint on disk)")
def test_cross_encoder_scores_and_trains_two_steps():
    from olaya.benchmarks.reranker import CrossEncoderReranker, TrainingConfig

    rr = CrossEncoderReranker(threads=1, seed=0)
    probs = rr.score([("heart. Subclass of organ.", "Heart. Subclass of Organ."), ("heart.", "Toe nail.")])
    assert probs.shape == (2,) and 0.0 <= probs.min() and probs.max() <= 1.0
    pairs = [Pair("a", "x", 1), Pair("a", "y", 0), Pair("b", "y", 1), Pair("b", "x", 0)]
    texts = [("heart.", "Heart."), ("heart.", "Toe."), ("toe.", "Toe."), ("toe.", "Heart.")]
    cfg = TrainingConfig(epochs=1, batch_size=2, time_budget_s=600, threads=1, log_every=1)
    stats = rr.train(pairs, texts, cfg, validation=(pairs, texts))
    assert stats.steps == 2 and len(stats.loss_curve) == 2 and stats.validation
    assert not stats.stopped_by_budget
