"""Tests for the LLM judge stage: candidates, prompt rendering, cache, decoding, stub end-to-end.

No network: the candidates use the hash encoder, the judge is the reference-backed stub.
The real-judge test runs only with ``OLAYA_RUN_LLM_TESTS=1`` and an ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from olaya.benchmarks.llm_judge import (
    OUTPUT_SCHEMA,
    PROMPT_VERSION,
    RELATIONS,
    AnthropicJudge,
    Candidate,
    CandidateConfig,
    CandidateGenerator,
    CandidateSet,
    JudgeRequest,
    Judgment,
    JudgmentCache,
    NullJudge,
    StubJudge,
    Usage,
    cost_estimate,
    decode,
    entity_cards,
    judge_candidates,
    judge_quality,
    render_batch,
    topk_pairs,
)
from olaya.benchmarks.oaei import OAEIAlignment, OAEIConferenceBenchmark
from olaya.embeddings import HashEncoder

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "oaei"
RUN_REAL = os.environ.get("OLAYA_RUN_LLM_TESTS") == "1" and bool(os.environ.get("ANTHROPIC_API_KEY"))


@pytest.fixture(scope="module")
def bench() -> OAEIConferenceBenchmark:
    return OAEIConferenceBenchmark(FIXTURES / "ontologies", FIXTURES / "reference_alignments")


@pytest.fixture(scope="module")
def cmt_ekaw(bench):
    gen = CandidateGenerator(HashEncoder(), CandidateConfig())
    src, tgt = bench.ontology("cmt"), bench.ontology("ekaw")
    return gen.build(src, tgt), entity_cards(src), entity_cards(tgt), bench.reference("cmt", "ekaw")


# -- candidates -----------------------------------------------------------------------


def test_topk_pairs_union_of_both_sides_with_min_score():
    scores = {("a", "x"): 0.9, ("a", "y"): 0.4, ("b", "x"): 0.5, ("b", "y"): 0.1, ("c", "y"): 0.1}
    assert topk_pairs(scores, 1, 0.15) == {("a", "x"), ("b", "x"), ("a", "y")}  # y's best above 0.15 is a
    assert topk_pairs(scores, 1, 0.0) == {("a", "x"), ("b", "x"), ("a", "y"), ("c", "y")}  # c keeps its top-1
    assert topk_pairs(scores, 2, 0.0) == set(scores)
    assert topk_pairs(scores, 0, 0.0) == set()


def test_candidate_generation_recall_on_cmt_ekaw(cmt_ekaw):
    cands, src_cards, tgt_cards, ref = cmt_ekaw
    assert cands.candidates == sorted(cands.candidates, key=lambda c: c.pair)  # deterministic order
    assert all(c.kind in ("class", "property") for c in cands.candidates)
    assert all(c.source in src_cards and c.target in tgt_cards for c in cands.candidates)
    accepted = [c for c in cands.candidates if c.accepted]
    assert accepted and all("accepted" in c.origin for c in accepted)
    assert {o for c in cands.candidates for o in c.origin} >= {"accepted", "hybrid", "path", "rich"}
    rec = cands.recall(ref)
    # Token-hash embeddings are not semantic; the real encoder reaches 1.0 on this pair (see the report).
    assert rec["M3"] >= 0.75 and rec["M1"] >= 0.8
    assert len(cands.candidates) < 1500  # 29 x 74 classes + 59 x 33 properties cross product is ~4100


def test_candidate_recall_counts_only_pairs_in_the_set():
    cs = CandidateSet("a", "b", [Candidate("Paper", "Paper", "class", 1.0, 1.0, 1.0, True, ("accepted",))])
    ref = [OAEIAlignment("Paper", "Paper"), OAEIAlignment("Review", "Review"), OAEIAlignment("hasX", "hasY")]
    assert cs.recall(ref) == {"M1": 0.5, "M2": 0.0, "M3": pytest.approx(1 / 3)}


# -- prompt -------------------------------------------------------------------------


def test_entity_cards_and_prompt_rendering_are_deterministic(bench, cmt_ekaw):
    cands, src_cards, tgt_cards, _ = cmt_ekaw
    again = entity_cards(bench.ontology("cmt"))
    assert again == src_cards
    paper = src_cards["Paper"]
    assert paper.kind == "class" and paper.ontology == "cmt" and paper.label == "Paper"
    assert "Document" in paper.parents and paper.path.endswith("Paper")
    assert any("(" in p and "->" in p for p in paper.properties)  # properties with domain -> range
    prop = src_cards["hasAuthor"]
    assert prop.kind == "property" and prop.domain and prop.range
    text = prop.render()
    assert "domain:" in text and "range:" in text and "object property" in text
    batch = [JudgeRequest(src_cards[c.source], tgt_cards[c.target]) for c in cands.candidates[:12]]
    rendered = render_batch(batch)
    assert rendered == render_batch(list(batch))
    assert rendered.count("### Pair ") == 12 and "Ontology A: cmt" in rendered and "Ontology B: ekaw" in rendered
    d1 = batch[0].digest("claude-opus-5-5")
    assert d1 == batch[0].digest("claude-opus-5-5") and d1 != batch[0].digest("claude-sonnet-5-5")
    assert d1 != batch[0].digest("claude-opus-5-5", prompt_version=PROMPT_VERSION + "x")
    assert d1 != batch[1].digest("claude-opus-5-5")


def test_output_schema_is_strict():
    item = OUTPUT_SCHEMA["properties"]["verdicts"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"id", "equivalent", "confidence", "relation", "rationale"}
    assert tuple(item["properties"]["relation"]["enum"]) == RELATIONS


def test_anthropic_judge_builds_request_without_network(cmt_ekaw):
    cands, src_cards, tgt_cards, _ = cmt_ekaw

    class FakeMessages:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            n = kwargs["messages"][0]["content"].count("### Pair ")
            verdicts = [
                {"id": i, "equivalent": i % 2 == 1, "confidence": 0.9, "relation": "equivalent", "rationale": "x"}
                for i in range(1, n + 1)
            ]
            text = json.dumps({"verdicts": verdicts})
            return type(
                "R",
                (),
                {
                    "stop_reason": "end_turn",
                    "content": [type("T", (), {"type": "text", "text": text})()],
                    "usage": type("U", (), {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0})(),
                    "_request_id": "req_test",
                },
            )()

    class FakeClient:
        def __init__(self):
            self.messages = FakeMessages()

    client = FakeClient()
    judge = AnthropicJudge("claude-opus-5-5", effort="low", client=client)
    batch = [JudgeRequest(src_cards[c.source], tgt_cards[c.target]) for c in cands.candidates[:4]]
    verdicts, usage = judge.judge(batch)
    kw = client.messages.kwargs
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"] == {"type": "json_schema", "schema": OUTPUT_SCHEMA}
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "thinking" not in kw  # adaptive by default on these models
    assert [v.equivalent for v in verdicts] == [True, False, True, False]
    assert usage.input_tokens == 100 and usage.output_tokens == 50 and usage.request_id == "req_test"
    assert usage.cost_usd() == pytest.approx((100 * 4 + 50 * 20) / 1e6)
    haiku = AnthropicJudge("claude-haiku-4-5", client=client)
    assert "effort" not in haiku._request_kwargs(batch)["output_config"]


# -- cache ---------------------------------------------------------------------------


def test_cache_round_trip_and_idempotent_judging(tmp_path, cmt_ekaw):
    cands, src_cards, tgt_cards, ref = cmt_ekaw
    reference = {("cmt", "ekaw"): {(r.entity1, r.entity2) for r in ref}}
    judge = StubJudge(reference)
    path = tmp_path / "stub.jsonl"
    cache = JudgmentCache(path)
    ledger: list[Usage] = []
    v1 = judge_candidates(cands, src_cards, tgt_cards, judge, cache, batch_size=10, ledger=ledger)
    assert len(v1) == len(cands.candidates) and sum(u.n_pairs for u in ledger) == len(cands.candidates)
    assert len(ledger) == -(-len(cands.candidates) // 10)
    rec = json.loads(path.read_text().splitlines()[0])
    assert set(rec) >= {"key", "model", "prompt_version", "source", "target", "verdict"}
    # Second run: everything comes from the cache, nothing is sent, same verdicts.
    reloaded = JudgmentCache(path)
    assert len(reloaded) == len(cands.candidates)
    ledger2: list[Usage] = []
    judge_b = StubJudge(reference)
    v2 = judge_candidates(cands, src_cards, tgt_cards, judge_b, reloaded, ledger=ledger2)
    assert ledger2 == [] and v2 == v1
    # A different model does not share the cache entries.
    judge_b.model = "other"
    ledger3: list[Usage] = []
    judge_candidates(cands, src_cards, tgt_cards, judge_b, reloaded, ledger=ledger3)
    assert sum(u.n_pairs for u in ledger3) == len(cands.candidates)
    reloaded.rewrite()
    lines = path.read_text().splitlines()
    assert len(lines) == 2 * len(cands.candidates) and lines == sorted(lines)


def test_stub_noise_is_deterministic(cmt_ekaw):
    cands, src_cards, tgt_cards, ref = cmt_ekaw
    reference = {("cmt", "ekaw"): {(r.entity1, r.entity2) for r in ref}}
    batch = [JudgeRequest(src_cards[c.source], tgt_cards[c.target]) for c in cands.candidates]
    a, _ = StubJudge(reference, noise=0.3, seed=1).judge(batch)
    b, _ = StubJudge(reference, noise=0.3, seed=1).judge(batch)
    c, _ = StubJudge(reference).judge(batch)
    assert a == b
    flips = sum(x.equivalent != y.equivalent for x, y in zip(a, c, strict=True))
    assert 0.15 * len(batch) < flips < 0.45 * len(batch)
    assert all(v.relation == "unrelated" and not v.equivalent for v in NullJudge().judge(batch)[0])


# -- decoding ------------------------------------------------------------------------


def _toy() -> tuple[CandidateSet, dict]:
    cs = CandidateSet(
        "a",
        "b",
        [
            Candidate("Paper", "Paper", "class", 1.0, 1.0, 0.9, True, ("accepted",)),  # right, hybrid-accepted
            Candidate("Document", "Document", "class", 1.0, 1.0, 0.9, True, ("accepted",)),  # wrong, hybrid-accepted
            Candidate("Event", "Activity", "class", 0.2, 0.3, 0.8, False, ("rich",)),  # right, uncertain
            Candidate("Event", "Event", "class", 0.4, 0.4, 0.8, False, ("hybrid",)),  # wrong, uncertain
            Candidate("hasX", "hasY", "property", 0.3, 0.0, 0.7, False, ("hybrid",)),  # right, uncertain
        ],
    )
    verdicts = {
        ("Paper", "Paper"): Judgment(True, 0.95, "equivalent"),
        ("Document", "Document"): Judgment(False, 0.8, "related"),
        ("Event", "Activity"): Judgment(True, 0.7, "equivalent"),
        ("Event", "Event"): Judgment(False, 0.6, "subsumes"),
        ("hasX", "hasY"): Judgment(True, 0.4, "equivalent"),
    }
    return cs, verdicts


def test_decode_judge_repair_and_fused():
    cs, v = _toy()
    pairs = lambda al: {(c.entity1, c.entity2) for c in al}  # noqa: E731
    assert pairs(decode(cs, v, mode="judge", threshold=0.5)) == {("Paper", "Paper"), ("Event", "Activity")}
    assert pairs(decode(cs, v, mode="judge", threshold=0.3)) == {
        ("Paper", "Paper"),
        ("Event", "Activity"),
        ("hasX", "hasY"),
    }
    # repair: Document/Document rejected (conf 0.8 >= 0.5), Event/Activity added, hasX/hasY not (0.4 < 0.5)
    assert pairs(decode(cs, v, mode="repair", threshold=0.5)) == {("Paper", "Paper"), ("Event", "Activity")}
    # at a stricter threshold the rejection no longer fires and the accepted pair survives
    assert ("Document", "Document") in pairs(decode(cs, v, mode="repair", threshold=0.9))
    fused = {(c.entity1, c.entity2): c.measure for c in decode(cs, v, mode="fused", threshold=0.0, weight=0.5)}
    assert fused[("Paper", "Paper")] == pytest.approx(0.5 * 1.0 + 0.5 * 0.95)
    assert fused[("Event", "Activity")] == pytest.approx(0.5 * 0.2 + 0.5 * 0.7)  # beats Event/Event (0.2)
    assert ("Event", "Event") not in fused and fused[("Document", "Document")] == pytest.approx(0.5)
    assert pairs(decode(cs, v, mode="fused", threshold=0.5, weight=0.5)) == {
        ("Paper", "Paper"),
        ("Document", "Document"),
    }
    with pytest.raises(ValueError):
        decode(cs, v, mode="nope", threshold=0.5)
    ref = [OAEIAlignment("Paper", "Paper"), OAEIAlignment("Event", "Activity"), OAEIAlignment("hasX", "hasY")]
    assert judge_quality(cs, v, ref) == {"tp": 2, "fp": 0, "fn": 1}
    assert Judgment(True, 1.4, "equivalent").score() == 1.0 and Judgment(False, 1.0, "related").score() == 0.0
    assert Judgment.from_dict({"relation": "bogus"}).relation == "unrelated"


def test_perfect_stub_reproduces_reference_up_to_candidate_ceiling(cmt_ekaw):
    cands, src_cards, tgt_cards, ref = cmt_ekaw
    verdicts = judge_candidates(
        cands, src_cards, tgt_cards, StubJudge({("cmt", "ekaw"): {(r.entity1, r.entity2) for r in ref}})
    )
    gold = {(r.entity1, r.entity2) for r in ref} & cands.pairs
    out = {(c.entity1, c.entity2) for c in decode(cands, verdicts, mode="judge", threshold=0.5)}
    assert out <= gold and len(out) >= len(gold) - 1  # 1:1 selection may drop a non-1:1 reference pair
    repaired = {(c.entity1, c.entity2) for c in decode(cands, verdicts, mode="repair", threshold=0.5)}
    assert repaired == out


def test_cost_estimate_per_model(cmt_ekaw):
    cands, src_cards, tgt_cards, _ = cmt_ekaw
    batch = [JudgeRequest(src_cards[c.source], tgt_cards[c.target]) for c in cands.candidates[:15]]
    est = cost_estimate([batch, batch])
    assert set(est) == {"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"}
    assert est["claude-opus-5-5"]["pairs"] == 30 and est["claude-opus-5-5"]["requests"] == 2
    assert est["claude-opus-5-5"]["usd"] > est["claude-sonnet-5-5"]["usd"] > est["claude-haiku-4-5"]["usd"] > 0
    counted = cost_estimate([batch], counted_input=[1234])
    assert (
        counted["claude-haiku-4-5"]["input_tokens"] == 1234 and counted["claude-haiku-4-5"]["method"] == "count_tokens"
    )


# -- end to end ----------------------------------------------------------------------


def test_stub_benchmark_end_to_end_two_pairs(tmp_path):
    t0 = time.perf_counter()
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "benchmark_oaei_llm.py"),
        "--judge",
        "stub",
        "--encoder",
        "hash",
        "--pairs",
        "2",
        "--results-dir",
        str(tmp_path),
        "--judgments-dir",
        str(tmp_path / "judgments"),
        "--cache-stub",
    ]
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert time.perf_counter() - t0 < 20
    res = json.loads((tmp_path / "oaei-llm.json").read_text())
    assert res["judge"] == "stub" and res["pairs"] == ["cmt-conference", "cmt-confOf"]
    assert (tmp_path / "judgments" / "stub.jsonl").exists() and (tmp_path / "oaei-llm.md").exists()
    assert res["judgments_sent"] == res["candidates"]["total"] > 0
    # A perfect stub: precision 1 and recall close to the candidate ceiling in the a-priori decoding.
    m3 = res["apriori"]["aggregates"]["M3"]["micro"]
    assert m3["precision"] == 1.0 and m3["recall"] >= res["candidates"]["recall_ceiling_micro"]["M3"] - 0.05
    assert res["lopo_cv"]["M3"]["f1"] >= 0.9
    assert "LLM-based" in res["category"] and "No language model was used" in (tmp_path / "oaei-llm.md").read_text()


@pytest.mark.skipif(not RUN_REAL, reason="set OLAYA_RUN_LLM_TESTS=1 and ANTHROPIC_API_KEY to call the real judge")
def test_real_judge_one_batch(cmt_ekaw):
    cands, src_cards, tgt_cards, ref = cmt_ekaw
    gold = {(r.entity1, r.entity2) for r in ref}
    pick = [c for c in cands.candidates if c.pair in gold][:3] + [c for c in cands.candidates if c.pair not in gold][:3]
    batch = [JudgeRequest(src_cards[c.source], tgt_cards[c.target]) for c in pick]
    verdicts, usage = AnthropicJudge("claude-haiku-4-5").judge(batch)
    assert len(verdicts) == 6 and usage.input_tokens > 0 and all(v.relation in RELATIONS for v in verdicts)
