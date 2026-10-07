"""Tests for the LLM judge stage: candidates, prompt rendering, cache, decoding, stub end-to-end.

No network: the candidates use the hash encoder, the judge is the reference-backed stub.
The real-judge test runs only with ``OLAYA_RUN_LLM_TESTS=1`` and an ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from olaya.benchmarks.llm_judge import (
    OUTPUT_SCHEMA,
    PROMPT_VERSION,
    RELATIONS,
    SYSTEM_PROMPT,
    AnthropicJudge,
    Candidate,
    CandidateConfig,
    CandidateGenerator,
    CandidateSet,
    HTTPStatusError,
    JudgeRequest,
    Judgment,
    JudgmentCache,
    NullJudge,
    OpenAICompatJudge,
    StubJudge,
    TransportError,
    Usage,
    cost_estimate,
    decode,
    entity_cards,
    extract_json,
    judge_candidates,
    judge_quality,
    render_batch,
    resolve_api_key,
    topk_pairs,
    validate_verdicts,
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


# -- OpenAI-compatible judge (fake transport, no network) -----------------------------

QWEN_URL = "https://maas.qwencloudapi.com/compatible-mode/v1"


def _verdicts(n: int) -> list[dict]:
    return [
        {"id": i, "equivalent": i % 2 == 1, "confidence": 0.8, "relation": "equivalent" if i % 2 else "unrelated"}
        for i in range(1, n + 1)
    ]


def _chat_response(content: str, *, reasoning: str | None = None, finish: str = "stop") -> bytes:
    msg: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "qwen3.8-flash",
            "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
            "usage": {"prompt_tokens": 321, "completion_tokens": 45, "total_tokens": 366},
        }
    ).encode()


class FakeTransport:
    """Canned replies in order (the last one repeats): ``(status, body)``, an exception, or a callable."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, url, headers, body, timeout):
        with self.lock:
            self.calls.append({"url": url, "headers": dict(headers), "body": body, "timeout": timeout})
            reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(url, headers, body)
        return reply


def _echo_verdicts(url, headers, body):
    """A reply with one verdict per pair in the request, wrapped in prose and a code fence."""
    n = json.loads(body)["messages"][1]["content"].count("### Pair ")
    return 200, _chat_response("Here are the verdicts:\n```json\n" + json.dumps({"verdicts": _verdicts(n)}) + "\n```")


def _judge(transport, **kw) -> OpenAICompatJudge:
    kw.setdefault("provider", "qwen")
    kw.setdefault("base_url", QWEN_URL)
    kw.setdefault("api_key", "sk-test")
    kw.setdefault("sleep", lambda s: None)
    return OpenAICompatJudge("qwen3.8-flash", transport=transport, **kw)


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure! Here it is:\n```json\n{"verdicts": []}\n```\nHope this helps.') == {"verdicts": []}
    assert extract_json('prose with braces } first, then [1, 2] and {"x": 2}') == [1, 2]
    assert extract_json('```\n[{"id": 1}]\n```') == [{"id": 1}]
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_validate_verdicts_schema_semantics():
    good = validate_verdicts({"verdicts": _verdicts(2)}, 2)
    assert [v.equivalent for v in good] == [True, False] and good[0].rationale == ""
    assert validate_verdicts(_verdicts(1), 1)[0].confidence == 0.8  # bare array accepted
    out_of_order = validate_verdicts({"verdicts": list(reversed(_verdicts(3)))}, 3)
    assert [v.equivalent for v in out_of_order] == [True, False, True]
    with pytest.raises(ValueError, match="missing pair"):
        validate_verdicts({"verdicts": _verdicts(1)}, 2)
    for bad in (
        {"verdicts": [{**_verdicts(1)[0], "relation": "maybe"}]},
        {"verdicts": [{**_verdicts(1)[0], "confidence": 1.5}]},
        {"verdicts": [{**_verdicts(1)[0], "equivalent": "yes"}]},
        {"verdicts": [{"id": 1, "equivalent": True}]},
        {"answer": 1},
        {"verdicts": _verdicts(1) + _verdicts(1)},
    ):
        with pytest.raises(ValueError):
            validate_verdicts(bad, 1)


def test_api_key_and_base_url_resolution(monkeypatch):
    for name in ("LLM_JUDGE_API_KEY", "LLM_JUDGE_BASE_URL", "ZAI_API_KEY", "Z_AI_API_KEY", "ZAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="LLM_JUDGE_API_KEY"):
        resolve_api_key()
    with pytest.raises(RuntimeError, match="no base URL"):
        OpenAICompatJudge("m", transport=FakeTransport([(200, b"{}")]))
    with pytest.raises(RuntimeError, match="no API key"):
        OpenAICompatJudge("m", base_url=QWEN_URL, transport=FakeTransport([(200, b"{}")]))
    with pytest.raises(RuntimeError, match="ZAI_API_KEY or Z_AI_API_KEY"):
        OpenAICompatJudge.zai(transport=FakeTransport([(200, b"{}")]))
    assert resolve_api_key("MY_KEY", {"MY_KEY": " k1 "}) == "k1"
    assert resolve_api_key(("A", "B"), {"B": "b"}) == "b"
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "generic")
    monkeypatch.setenv("LLM_JUDGE_BASE_URL", QWEN_URL + "/")
    j = OpenAICompatJudge(transport=FakeTransport([(200, b"{}")]))
    assert j.api_key == "generic" and j.base_url == QWEN_URL and j.model == "qwen3.8-flash" and j.provider == "openai"
    assert j.thinking is None and j.json_mode
    monkeypatch.setenv("Z_AI_API_KEY", "from-alias")
    z = OpenAICompatJudge.zai(transport=FakeTransport([(200, b"{}")]))
    assert z.api_key == "from-alias" and z.base_url == "https://api.z.ai/api/paas/v4" and z.provider == "zai"
    assert z.model == "glm-5.3-flash" and z.thinking == "off" and z.name.startswith("zai[glm-5.3-flash,")
    monkeypatch.setenv("ZAI_BASE_URL", "https://example.test/v4/")
    assert OpenAICompatJudge.zai(transport=FakeTransport([])).base_url == "https://example.test/v4"


def test_openai_compat_judge_request_and_parsing(cmt_ekaw):
    cands, src_cards, tgt_cards, _ = cmt_ekaw
    batch = [JudgeRequest(src_cards[c.source], tgt_cards[c.target]) for c in cands.candidates[:4]]
    fence = "Sure, here is my analysis.\n```json\n" + json.dumps({"verdicts": _verdicts(4)}) + "\n```\nDone."
    transport = FakeTransport(
        [
            (200, _chat_response(fence)),
            (200, _chat_response(json.dumps({"verdicts": _verdicts(4)}), reasoning="<think>long...</think>")),
        ]
    )
    judge = _judge(transport)
    verdicts, usage = judge.judge(batch)
    assert [v.equivalent for v in verdicts] == [True, False, True, False]
    assert usage.input_tokens == 321 and usage.output_tokens == 45 and usage.request_id == "chatcmpl-test"
    assert usage.model == "qwen3.8-flash" and usage.provider == "qwen" and usage.cost_usd() == 0.0  # not pinned
    assert Usage("glm-5.3-flash", 1, 321, 45, provider="zai").cost_usd() == pytest.approx((321 * 0.15 + 45 * 0.5) / 1e6)
    call = transport.calls[0]
    assert call["url"] == QWEN_URL + "/chat/completions" and call["timeout"] == 180.0
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    body = json.loads(call["body"])
    assert body["model"] == "qwen3.8-flash" and body["temperature"] == 0 and body["stream"] is False
    assert body["response_format"] == {"type": "json_object"} and "thinking" not in body
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert body["messages"][1]["content"] == render_batch(batch)
    verdicts2, _ = judge.judge(batch)  # reasoning_content is ignored
    assert verdicts2 == verdicts
    assert _judge(transport, thinking="on", thinking_style="zai")._request_body(batch)["thinking"] == {
        "type": "enabled"
    }
    assert _judge(transport, thinking="off", thinking_style="zai")._request_body(batch)["thinking"] == {
        "type": "disabled"
    }
    assert _judge(transport, thinking="off")._request_body(batch)["enable_thinking"] is False
    assert "response_format" not in _judge(transport, json_mode=False)._request_body(batch)
    with pytest.raises(ValueError):
        _judge(transport, thinking="maybe")


def test_openai_compat_judge_falls_back_when_response_format_is_rejected(cmt_ekaw):
    cands, src_cards, tgt_cards, _ = cmt_ekaw
    batch = [JudgeRequest(src_cards[c.source], tgt_cards[c.target]) for c in cands.candidates[:2]]
    transport = FakeTransport(
        [
            (400, b'{"error": {"message": "unsupported parameter: response_format"}}'),
            (400, b'{"error": {"message": "unknown field thinking"}}'),
            _echo_verdicts,
        ]
    )
    judge = _judge(transport, thinking="off", thinking_style="zai")
    verdicts, _ = judge.judge(batch)
    assert len(verdicts) == 2 and len(transport.calls) == 3
    assert "response_format" in json.loads(transport.calls[0]["body"])
    assert "response_format" not in json.loads(transport.calls[1]["body"])
    assert "thinking" not in json.loads(transport.calls[2]["body"]) and judge.json_mode is False
    judge.judge(batch)  # the fallback sticks: no further 400 round trip
    assert len(transport.calls) == 4
    other = _judge(FakeTransport([(400, b'{"error": "bad model"}')]))
    with pytest.raises(HTTPStatusError, match="400"):
        other.judge(batch)


def test_openai_compat_judge_retries_then_succeeds_and_gives_up(cmt_ekaw):
    cands, src_cards, tgt_cards, _ = cmt_ekaw
    batch = [JudgeRequest(src_cards[c.source], tgt_cards[c.target]) for c in cands.candidates[:2]]
    slept: list[float] = []
    transport = FakeTransport(
        [
            (429, b'{"error": {"message": "rate limited"}}'),
            TransportError("timed out"),
            (503, b"upstream"),
            _echo_verdicts,
        ]
    )
    verdicts, _ = _judge(transport, sleep=slept.append, backoff_s=0.5).judge(batch)
    assert len(verdicts) == 2 and len(transport.calls) == 4
    assert len(slept) == 3 and 0.5 <= slept[0] < 0.7 and 1.0 <= slept[1] < 1.2 and 2.0 <= slept[2] < 2.2
    with pytest.raises(HTTPStatusError, match="401"):  # 4xx other than 429 is not retried
        _judge(FakeTransport([(401, b"unauthorized")])).judge(batch)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        _judge(FakeTransport([(500, b"boom")]), max_retries=2).judge(batch)
    with pytest.raises(RuntimeError, match="truncated"):
        _judge(FakeTransport([(200, _chat_response("{", finish="length"))])).judge(batch)
    with pytest.raises(ValueError, match="missing pair"):
        _judge(FakeTransport([(200, _chat_response(json.dumps(_verdicts(1))))])).judge(batch)


def test_list_models():
    listing = json.dumps({"object": "list", "data": [{"id": "qwen3.8-max"}, {"id": "glm-5.3"}]}).encode()
    transport = FakeTransport([(200, listing)])
    assert _judge(transport).list_models() == ["glm-5.3", "qwen3.8-max"]
    assert transport.calls[0]["url"] == QWEN_URL + "/models" and transport.calls[0]["body"] is None


def test_concurrent_judging_is_ordered_and_cache_is_thread_safe(tmp_path, cmt_ekaw):
    cands, src_cards, tgt_cards, _ = cmt_ekaw
    transport = FakeTransport([_echo_verdicts])
    judge = _judge(transport)
    cache = JudgmentCache(tmp_path / "qwen__qwen3.8-flash.jsonl")
    ledger: list[Usage] = []
    v = judge_candidates(cands, src_cards, tgt_cards, judge, cache, batch_size=10, ledger=ledger, concurrency=4)
    seq = judge_candidates(cands, src_cards, tgt_cards, judge, None, batch_size=10, concurrency=1)
    assert v == seq and len(ledger) == -(-len(cands.candidates) // 10)
    lines = cache.path.read_text().splitlines()
    assert len(lines) == len(cands.candidates) == len(cache)
    assert all(json.loads(line)["model"] == "qwen3.8-flash" for line in lines)
    # idempotent: re-putting the same verdict does not append a duplicate line
    req = JudgeRequest(src_cards[cands.candidates[0].source], tgt_cards[cands.candidates[0].target])
    cache.put(req.digest(judge.model), req, judge.model, v[req.pair])
    assert len(cache.path.read_text().splitlines()) == len(lines)


def _load_script():
    spec = importlib.util.spec_from_file_location("benchmark_oaei_llm", REPO / "scripts" / "benchmark_oaei_llm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_openai_benchmark_end_to_end_two_pairs(tmp_path, monkeypatch):
    import olaya.benchmarks.llm_judge as lj

    transport = FakeTransport([_echo_verdicts])
    monkeypatch.setattr(lj, "urllib_transport", transport)
    monkeypatch.delenv("LLM_JUDGE_API_KEY", raising=False)
    monkeypatch.setenv("QWEN_KEY", "sk-fake")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    mod = _load_script()
    base = ["--encoder", "hash", "--pairs", "2", "--results-dir", str(tmp_path), "--judgments-dir", str(tmp_path / "j")]
    argv = [
        *base,
        "--judge",
        "openai",
        "--base-url",
        QWEN_URL,
        "--api-key-env",
        "QWEN_KEY",
        "--provider",
        "qwen",
        "--model",
        "qwen3.8-flash",
        "--concurrency",
        "3",
    ]
    assert mod.main([*base, "--judge", "openai", "--provider", "qwen"]) == 2  # no base URL
    assert mod.main([*base, "--judge", "openai", "--base-url", QWEN_URL]) == 2  # no LLM_JUDGE_API_KEY
    assert mod.main(argv) == 0
    res = json.loads((tmp_path / "oaei-llm.json").read_text())
    assert res["judge"] == "openai" and res["model"] == "qwen3.8-flash" and res["provider"] == "qwen"
    assert res["base_url"] == QWEN_URL and res["json_mode"] == "on" and res["thinking"] is None
    assert res["judgments_sent"] == res["candidates"]["total"] > 0 and res["usage_total"]["input_tokens"] > 0
    assert res["usage_total"]["usd"] == 0 and res["cost_estimate"] == {
        "qwen3.8-flash": res["cost_estimate"]["qwen3.8-flash"]
    }
    assert (tmp_path / "j" / "qwen__qwen3.8-flash.jsonl").exists()
    assert all(c["url"] == QWEN_URL + "/chat/completions" for c in transport.calls)
    assert all(c["headers"]["Authorization"] == "Bearer sk-fake" for c in transport.calls)
    assert all("thinking" not in json.loads(c["body"]) for c in transport.calls)
    md = (tmp_path / "oaei-llm.md").read_text()
    assert "No language model was used" not in md and "OpenAI-compatible backend" in md and "qwen[qwen3.8-flash" in md
    n_calls = len(transport.calls)
    assert mod.main(argv) == 0  # second run: everything from the cache, no requests
    assert len(transport.calls) == n_calls
    assert json.loads((tmp_path / "oaei-llm.json").read_text())["judgments_sent"] == 0


def test_zai_alias_and_list_models_cli(tmp_path, monkeypatch, capsys):
    import olaya.benchmarks.llm_judge as lj

    listing = json.dumps({"data": [{"id": "glm-5.3-flash"}, {"id": "glm-4.7-flash"}]}).encode()
    transport = FakeTransport([(200, listing), _echo_verdicts])
    monkeypatch.setattr(lj, "urllib_transport", transport)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setenv("Z_AI_API_KEY", "sk-zai")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    mod = _load_script()
    assert mod.main(["--judge", "zai", "--list-models"]) == 0
    assert capsys.readouterr().out.splitlines() == ["glm-4.7-flash", "glm-5.3-flash"]
    assert transport.calls[0]["url"] == "https://api.z.ai/api/paas/v4/models"
    base = ["--encoder", "hash", "--pairs", "1", "--results-dir", str(tmp_path), "--judgments-dir", str(tmp_path / "j")]
    assert mod.main([*base, "--judge", "zai"]) == 0
    res = json.loads((tmp_path / "oaei-llm.json").read_text())
    assert res["provider"] == "zai" and res["model"] == "glm-5.3-flash" and res["thinking"] == "off"
    assert res["usage_total"]["usd"] > 0 and (tmp_path / "j" / "zai__glm-5.3-flash.jsonl").exists()
    assert all(json.loads(c["body"])["thinking"] == {"type": "disabled"} for c in transport.calls[1:])


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


def test_qwen_thinking_style_and_relation_aliases():
    from olaya.benchmarks.llm_judge import OpenAICompatJudge, _canonical_relation

    judge = OpenAICompatJudge(
        "qwen3.8-flash",
        provider="qwen",
        base_url="https://example.invalid/v1",
        api_key="k",
        thinking="off",
        transport=lambda *a, **k: (200, b"{}"),
    )
    assert judge.thinking_style == "qwen"
    body = judge._request_body([])
    assert body["enable_thinking"] is False and "thinking" not in body
    zai = OpenAICompatJudge(
        "glm-5.3-flash",
        provider="zai",
        base_url="https://example.invalid/v1",
        api_key="k",
        thinking="off",
        transport=lambda *a, **k: (200, b"{}"),
    )
    assert zai._request_body([])["thinking"] == {"type": "disabled"}
    assert _canonical_relation("disjoint") == "unrelated"
    assert _canonical_relation("Subsumed-By") == "subsumed_by"
    assert _canonical_relation("equivalent") == "equivalent"
    assert _canonical_relation("weird") == "weird"
