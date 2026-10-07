"""Tests for the LLM-judge ensemble: cache alignment, combiners, abstentions, nested CV, script end-to-end.

No network and no model: three synthetic judge caches over six candidate pairs, and reference-derived
caches for the script run (hash encoder).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from olaya.benchmarks.llm_ensemble import (
    ABSTAIN_RATIONALE,
    COMBINERS,
    abstain,
    align_verdicts,
    combine_anchor,
    combine_majority,
    combine_unanimous,
    combine_weighted,
    coverage,
    cross_validate_nested,
    ensemble_verdicts,
    judge_weights,
    load_judge_cache,
    load_judge_caches,
    single_judge_verdicts,
)
from olaya.benchmarks.llm_judge import (
    PROMPT_VERSION,
    Candidate,
    CandidateConfig,
    CandidateGenerator,
    CandidateSet,
    Judgment,
    decode,
    judge_quality,
)
from olaya.benchmarks.oaei import PRF, OAEIAlignment, OAEIConferenceBenchmark, leave_one_pair_out_folds
from olaya.embeddings import HashEncoder

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "oaei"

PAIRS = [
    ("Paper", "Contribution"),
    ("Event", "Activity"),
    ("Person", "Person"),
    ("Topic", "Vehicle"),
    ("A", "B"),
    ("C", "D"),
]


def _cand(s: str, t: str, *, accepted: bool = False, hybrid: float = 0.4) -> Candidate:
    return Candidate(s, t, "class", hybrid, 0.0, 0.0, accepted, ("hybrid",))


def _cands() -> CandidateSet:
    return CandidateSet(
        "cmt", "ekaw", [_cand(s, t, accepted=(s == t), hybrid=0.9 if s == t else 0.4) for s, t in PAIRS]
    )


def _rec(model: str, s: str, t: str, equivalent: bool, conf: float, *, version: str = PROMPT_VERSION) -> dict:
    return {
        "key": f"{model}:{s}:{t}",  # digests differ between models: alignment must not use them
        "model": model,
        "prompt_version": version,
        "source_ontology": "cmt",
        "target_ontology": "ekaw",
        "source": s,
        "target": t,
        "kind": "class",
        "verdict": {
            "equivalent": equivalent,
            "confidence": conf,
            "relation": "equivalent" if equivalent else "unrelated",
            "rationale": "",
        },
    }


def _write(path: Path, recs: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


@pytest.fixture
def caches(tmp_path):
    """Three fake judges; judge 2 (``b``) has no verdict for the last two pairs."""
    yes = {("Paper", "Contribution"), ("Event", "Activity"), ("Person", "Person")}
    a = [_rec("model-a", s, t, (s, t) in yes, 0.9) for s, t in PAIRS]
    b = [_rec("model-b", s, t, (s, t) in yes or (s, t) == ("Topic", "Vehicle"), 0.7) for s, t in PAIRS[:4]]
    c = [_rec("model-c", s, t, (s, t) == ("Person", "Person") or (s, t) == ("A", "B"), 0.6) for s, t in PAIRS]
    d = tmp_path / "j"
    d.mkdir()
    _write(d / "p__a.jsonl", a)
    _write(d / "p__b.jsonl", b)
    _write(d / "p__c.jsonl", c)
    return d


# -- loading and alignment -------------------------------------------------------------


def test_load_and_align_by_pair_identity(caches):
    loaded = load_judge_caches(caches, ["p__a", "p__b", "p__c"])
    assert [c.model for c in loaded] == ["model-a", "model-b", "model-c"]
    assert [c.short for c in loaded] == ["a", "b", "c"] and all(c.prompt_version == PROMPT_VERSION for c in loaded)
    assert [len(c.verdicts) for c in loaded] == [6, 4, 6]
    aligned = align_verdicts(loaded, _cands(), "cmt", "ekaw")
    assert list(aligned) == PAIRS and all(len(row) == 3 for row in aligned.values())
    assert aligned[("Paper", "Contribution")][0] == Judgment(True, 0.9, "equivalent")
    assert aligned[("A", "B")][1] is None and aligned[("C", "D")][1] is None  # judge b abstains
    assert aligned[("A", "B")][2].equivalent
    assert coverage(aligned, 3) == [
        {"judged": 6, "candidates": 6, "fraction": 1.0},
        {"judged": 4, "candidates": 6, "fraction": 0.6667},
        {"judged": 6, "candidates": 6, "fraction": 1.0},
    ]
    # the wrong ontology names align nothing
    assert all(row == [None, None, None] for row in align_verdicts(loaded, _cands(), "cmt", "edas").values())
    with pytest.raises(FileNotFoundError):
        load_judge_caches(caches, ["p__a", "p__missing"])


def test_cache_validation(tmp_path):
    bad = tmp_path / "mixed.jsonl"
    _write(bad, [_rec("m1", "A", "B", True, 1.0), _rec("m2", "C", "D", True, 1.0)])
    with pytest.raises(ValueError, match="several models"):
        load_judge_cache(bad)
    other = tmp_path / "v2.jsonl"
    _write(other, [_rec("m1", "A", "B", True, 1.0, version="oaei-judge-v2")])
    with pytest.raises(ValueError, match="prompt version"):
        load_judge_cache(other)
    assert load_judge_cache(other, prompt_version=None).prompt_version == "oaei-judge-v2"
    dup = tmp_path / "dup.jsonl"
    _write(dup, [_rec("m1", "A", "B", True, 1.0), _rec("m1", "A", "B", False, 0.8)])
    cache = load_judge_cache(dup)
    assert cache.records == 2 and len(cache.verdicts) == 1 and not cache.get(("cmt", "ekaw", "A", "B")).equivalent


# -- combiners ------------------------------------------------------------------------


Y9 = Judgment(True, 0.9, "equivalent")
Y7 = Judgment(True, 0.7, "equivalent")
N8 = Judgment(False, 0.8, "unrelated")
N6 = Judgment(False, 0.6, "subsumes")


def test_majority_semantics():
    v = combine_majority([Y9, Y7, N8])
    assert v.equivalent and v.confidence == pytest.approx(0.8) and v.relation == "equivalent"
    v = combine_majority([Y9, N8, N6])
    assert not v.equivalent and v.confidence == pytest.approx(0.7) and v.relation in ("unrelated", "subsumes")
    assert not combine_majority([Y9, N8]).equivalent  # a 1-1 tie is not a majority
    assert combine_majority([Y9, None, Y7]).equivalent  # abstentions do not vote
    assert not combine_majority([Y9, None, N8]).equivalent
    assert combine_majority([None, None]) == abstain() and abstain().rationale == ABSTAIN_RATIONALE
    assert abstain().score() == 0.0


def test_unanimous_semantics():
    v = combine_unanimous([Y9, Y7, Y9])
    assert v.equivalent and v.confidence == pytest.approx(0.7)  # min
    v = combine_unanimous([Y9, Y7, N6])
    assert not v.equivalent and v.confidence == pytest.approx(0.6)
    assert combine_unanimous([Y9, None, Y7]).equivalent and combine_unanimous([None, Y7]).confidence == 0.7
    assert combine_unanimous([None, None, None]) == abstain()


def test_weighted_semantics():
    # P(eq): Y9 -> 0.9, N8 -> 0.2.  Equal weights: (0.9 + 0.2) / 2 = 0.55 -> equivalent, confidence 0.55.
    v = combine_weighted([Y9, N8])
    assert v.equivalent and v.confidence == pytest.approx(0.55)
    # Weight the negative judge 3x: (0.9 + 3 * 0.2) / 4 = 0.375 -> not equivalent, confidence 0.625.
    v = combine_weighted([Y9, N8], weights=[1.0, 3.0])
    assert not v.equivalent and v.confidence == pytest.approx(0.625)
    assert combine_weighted([Y9, None], weights=[1.0, 5.0]).confidence == pytest.approx(0.9)  # abstention has no weight
    assert combine_weighted([Y9, N8], weights=[0.0, 1.0]).equivalent is False  # zero weight = ignored
    assert combine_weighted([None, N8], weights=[1.0, 0.0]) == abstain()


def test_anchor_semantics():
    v = combine_anchor([Y9, N8, Y7])
    assert v.equivalent and v.confidence == pytest.approx(0.8)  # anchor + the agreeing judge
    v = combine_anchor([Y9, N8, N6])  # anchor says yes, nobody agrees
    assert not v.equivalent and v.confidence == pytest.approx(0.7)
    assert combine_anchor([N8, Y9, Y7]) == Judgment(False, 0.8, "unrelated", "anchor: not equivalent")
    assert combine_anchor([None, Y9, Y7]) == abstain()  # anchor abstains
    assert combine_anchor([Y9, None, None]) == abstain()  # nobody can agree
    assert combine_anchor([N8, Y9, Y7], anchor=1).equivalent  # another judge as the anchor


# -- missing verdicts and the Judgment-compatible view ----------------------------------


def test_ensemble_view_runs_decode_and_judge_quality_unchanged(caches):
    loaded = load_judge_caches(caches, ["p__a", "p__b", "p__c"])
    cands = _cands()
    aligned = align_verdicts(loaded, cands, "cmt", "ekaw")
    ref = [OAEIAlignment(s, t, "=", 1.0) for s, t in PAIRS[:3]]
    for name in COMBINERS:
        verdicts = ensemble_verdicts(aligned, name, weights=[1.0, 1.0, 1.0], anchor=0)
        assert set(verdicts) == set(aligned) and all(isinstance(v, Judgment) for v in verdicts.values())
        counts = judge_quality(cands, verdicts, ref)
        assert set(counts) == {"tp", "fp", "fn"}
        for mode in ("judge", "repair", "fused"):
            decode(cands, verdicts, mode=mode, threshold=0.5)
    maj = ensemble_verdicts(aligned, "majority")
    assert maj[("Paper", "Contribution")].equivalent and not maj[("Topic", "Vehicle")].equivalent
    assert maj[("A", "B")] == Judgment(False, 0.9, "unrelated", "majority 1/2 equivalent")  # b abstains, a vs c tie
    una = ensemble_verdicts(aligned, "unanimous")
    assert una[("Person", "Person")] == Judgment(True, 0.6, "equivalent", "unanimous equivalent")
    anc = ensemble_verdicts(aligned, "anchor")
    assert anc[("Paper", "Contribution")].equivalent and not anc[("A", "B")].equivalent
    # a pair nobody judged: abstain; the repair decoding then keeps the hybrid decision
    none = {p: [None, None, None] for p in aligned}
    for name in COMBINERS:
        v = ensemble_verdicts(none, name)
        assert all(j == abstain() for j in v.values())
        kept = decode(cands, v, mode="repair", threshold=0.5)
        assert {(a.entity1, a.entity2) for a in kept} == {("Person", "Person")}
        assert decode(cands, v, mode="judge", threshold=0.3) == []
    single = single_judge_verdicts(aligned, 1)
    assert single[("C", "D")] == abstain() and single[("Topic", "Vehicle")].equivalent


# -- weights and nested cross-validation ---------------------------------------------


def test_judge_weights_use_only_training_pairs(caches):
    loaded = load_judge_caches(caches, ["p__a", "p__b", "p__c"])
    cands = _cands()
    aligned = align_verdicts(loaded, cands, "cmt", "ekaw")
    ref = [OAEIAlignment(s, t, "=", 1.0) for s, t in PAIRS[:3]]
    per_pair = {"p1": (cands, aligned, ref)}
    w = judge_weights(per_pair, 3, ["p1"])
    assert w[0] == 1.0  # judge a is perfect
    assert w[1] == pytest.approx(PRF(3, 1, 0).f1, abs=1e-5)  # b: one FP, abstentions skipped
    assert w[2] == pytest.approx(PRF(1, 1, 2).f1, abs=1e-5)
    assert judge_weights(per_pair, 3, []) == [0.0, 0.0, 0.0]  # no training pairs: no evidence


def _entry(tp: int, fp: int, fn: int) -> dict[str, PRF]:
    return {m: PRF(tp, fp, fn) for m in ("M1", "M2", "M3")}


def test_nested_cv_never_selects_or_fits_on_the_held_out_pair():
    pairs = ["p1", "p2", "p3"]
    # config A is solid everywhere; config B is poor on p1, p2 and overwhelming on p3 only
    grid = {
        "A": {"p1": _entry(8, 2, 2), "p2": _entry(8, 2, 2), "p3": _entry(8, 2, 2)},
        "B": {"p1": _entry(2, 8, 8), "p2": _entry(2, 8, 8), "p3": _entry(100, 0, 0)},
    }
    folds = leave_one_pair_out_folds(pairs)
    seen: dict[str, list[str]] = {}

    def grid_for(train):
        seen[tuple(sorted(train))] = list(train)
        return grid

    cv = cross_validate_nested(grid_for, folds)
    for held, (train, test) in folds.items():
        assert test == [held] and held not in train
    assert sorted(seen) == [("p1", "p2"), ("p1", "p3"), ("p2", "p3")]  # the grid was built without the held-out pair
    assert cv["chosen"]["p3"] == "A" and cv["chosen"]["p1"] == "B" and cv["chosen"]["p2"] == "B"
    assert (cv["M3"]["tp"], cv["M3"]["fp"], cv["M3"]["fn"]) == (12, 18, 18)
    assert cv["folds"] == 3 and cv["distinct_configs_chosen"] == ["A", "B"]


# -- script end to end ----------------------------------------------------------------


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "benchmark_oaei_llm_ensemble", REPO / "scripts" / "benchmark_oaei_llm_ensemble.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reference_caches(tmp_path: Path, pairs: list[tuple[str, str]], *, drop_last_pair_for: str | None = None) -> Path:
    """Three judges derived from ra1 over the hash-encoder candidates: two perfect, one that rejects everything."""
    bench = OAEIConferenceBenchmark(FIXTURES / "ontologies", FIXTURES / "reference_alignments")
    gen = CandidateGenerator(HashEncoder(), CandidateConfig())
    recs: dict[str, list[dict]] = {"fake__good": [], "fake__also-good": [], "fake__contrarian": []}
    for s, t in pairs:
        src, tgt = bench.ontology(s), bench.ontology(t)
        gold = {(r.entity1, r.entity2) for r in bench.reference(s, t)}
        for c in gen.build(src, tgt).candidates:
            for name, equivalent in (
                ("fake__good", c.pair in gold),
                ("fake__also-good", c.pair in gold),
                ("fake__contrarian", False),
            ):
                if drop_last_pair_for == name and (s, t) == pairs[-1]:
                    continue
                rec = _rec(name.split("__")[1], c.source, c.target, equivalent, 1.0)
                rec["source_ontology"], rec["target_ontology"], rec["kind"] = src.name, tgt.name, c.kind
                recs[name].append(rec)
    d = tmp_path / "judgments"
    d.mkdir(exist_ok=True)
    for name, rs in recs.items():
        _write(d / f"{name}.jsonl", rs)
    return d


def test_ensemble_script_end_to_end_two_pairs(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pairs = [("cmt", "conference"), ("cmt", "confOf")]
    jdir = _reference_caches(tmp_path, pairs)
    mod = _load_script()
    base = ["--encoder", "hash", "--pairs", "2", "--results-dir", str(tmp_path), "--judgments-dir", str(jdir)]
    assert mod.main([*base, "--judgments", "fake__good"]) == 2  # one cache is not an ensemble
    assert mod.main([*base, "--judgments", "fake__good,fake__nope"]) == 2  # missing cache
    assert mod.main([*base, "--judgments", "fake__good,fake__contrarian", "--combiner", "mean"]) == 2
    assert mod.main([*base, "--judgments", "fake__good,fake__contrarian", "--anchor", "fake__x"]) == 2
    argv = [*base, "--judgments", "fake__good,fake__also-good,fake__contrarian"]
    assert mod.main(argv) == 0
    res = json.loads((tmp_path / "oaei-llm-ensemble.json").read_text())
    assert res["pairs"] == ["cmt-conference", "cmt-confOf"] and res["pairs_dropped"] == []
    assert res["judges"] == ["fake__good", "fake__also-good", "fake__contrarian"] and res["anchor"] == "fake__good"
    assert res["judge_models"] == ["good", "also-good", "contrarian"] and res["prompt_version"] == PROMPT_VERSION
    assert all(c["fraction"] == 1.0 for c in res["coverage"]) and res["candidates"]["total"] > 0
    assert set(res["per_combiner"]) == set(COMBINERS) and set(res["per_judge"]) == set(res["judges"])
    assert res["apriori"]["label"] == "combiner=majority,mode=repair,th=0.50"
    # two perfect judges outvote the contrarian: precision 1 and recall at the candidate ceiling (up to 1:1 selection)
    m3 = res["apriori"]["aggregates"]["M3"]["micro"]
    assert m3["precision"] == 1.0 and m3["recall"] >= res["candidates"]["recall_ceiling_micro"]["M3"] - 0.05
    # unanimity with a contrarian accepts nothing: the repair decoding keeps only the hybrid alignment
    una = res["per_combiner"]["unanimous"]["judge_quality_on_candidates"]["M3"]
    assert una["tp"] == 0 and una["fp"] == 0
    assert res["per_combiner"]["anchor"]["apriori"]["aggregates"]["M3"]["micro"]["precision"] == 1.0
    assert res["weights_all_pairs"]["fake__good"] == 1.0 and res["weights_all_pairs"]["fake__contrarian"] == 0.0
    assert res["lopo_cv"]["M3"]["f1"] >= 0.9 and res["looo_cv"]["folds"] >= 1 and res["grid_points"] == 4 * 255
    assert res["per_judge"]["fake__contrarian"]["lopo_cv"]["M3"]["f1"] < res["lopo_cv"]["M3"]["f1"]
    assert res["standings"][0]["system"] == "olaya (this run)" and res["rank_M3"] == 1
    assert "LLM-based ensemble" in res["category"]
    md = (tmp_path / "oaei-llm-ensemble.md").read_text()
    assert "All verdicts are cached" in md and "GraphMatcher" in md and "Where we land" in md
    assert "All 2 ontology pairs are covered by every judge." in md
    # a subset of combiners still reports the fixed a-priori row (majority) and only the chosen combiners' grids
    assert mod.main([*argv, "--combiner", "unanimous,anchor", "--result-stem", "sub"]) == 0
    sub = json.loads((tmp_path / "sub.json").read_text())
    assert list(sub["per_combiner"]) == ["unanimous", "anchor"] and sub["grid_points"] == 2 * 255
    assert sub["apriori"]["label"] == "combiner=majority,mode=repair,th=0.50"


def test_ensemble_script_drops_pairs_below_coverage(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pairs = [("cmt", "conference"), ("cmt", "confOf"), ("cmt", "edas")]
    jdir = _reference_caches(tmp_path, pairs, drop_last_pair_for="fake__also-good")
    mod = _load_script()
    base = ["--encoder", "hash", "--pairs", "3", "--results-dir", str(tmp_path), "--judgments-dir", str(jdir)]
    argv = [*base, "--judgments", "fake__good,fake__also-good", "--combiner", "majority"]
    assert mod.main(argv) == 0
    res = json.loads((tmp_path / "oaei-llm-ensemble.json").read_text())
    assert res["pairs_dropped"] == ["cmt-edas"] and res["pairs"] == ["cmt-conference", "cmt-confOf"]
    assert res["coverage_per_pair"]["cmt-edas"] == [1.0, 0.0] and res["coverage"][1]["fraction"] < 1.0
    assert "ontology pairs left out" in (tmp_path / "oaei-llm-ensemble.md").read_text()
    # with the coverage requirement lifted the pair stays in and the missing judge abstains there
    assert mod.main([*argv, "--min-coverage", "0", "--result-stem", "all"]) == 0
    res = json.loads((tmp_path / "all.json").read_text())
    assert res["pairs_dropped"] == [] and len(res["pairs"]) == 3
    assert res["apriori"]["aggregates"]["M3"]["micro"]["precision"] == 1.0  # a lone perfect judge still decides
