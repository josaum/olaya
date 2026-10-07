"""LLM-distilled corpus: leakage filters, disagreement filter, hashing and split determinism (no network)."""

from __future__ import annotations

import json

import pytest

from olaya.corpus import (
    CorpusRow,
    OpenAICompatClient,
    build_corpus,
    corpus_manifest,
    corpus_sha256,
    label_leaks,
    parse_generations,
    parse_label,
    read_jsonl,
    sample_classes,
    split_corpus,
    split_report,
    write_jsonl,
)
from olaya.ontology import Concept, create_incident_ontology


class StubLLM:
    """Generation answers come from a table; the labeller returns a configurable curie per text."""

    def __init__(self, generations: dict[str, list[str]], label_for: dict[str, str] | None = None):
        self.generations = generations
        self.label_for = label_for or {}
        self.calls: list[str] = []

    def __call__(self, messages: list[dict[str, str]], model: str) -> str:
        self.calls.append(model)
        user = messages[-1]["content"]
        if "Write " in user and '"observations"' in user:
            curie = user.split("Class: ", 1)[1].split("\n", 1)[0].rsplit("(", 1)[1].rstrip(")")
            seed = int(user.split("variation seed ")[1].split(".")[0])
            texts = [f"{t} [{seed}]" for t in self.generations.get(curie, [])]
            return json.dumps({"observations": texts})
        text = user.split('"""')[1].strip()
        wanted = self.label_for.get(text.split(" [")[0])
        if wanted is None:  # agree with whatever gold the stub generated for: find the line of its curie
            # The gold curie is not in the prompt; the stub remembers it through the generation table.
            for curie, texts in self.generations.items():
                if any(text.startswith(t) for t in texts):
                    wanted = curie
        return json.dumps({"curie": wanted or "none", "index": -1, "confidence": 0.9, "reason": "stub"})


@pytest.fixture
def ontology():
    return create_incident_ontology()


def test_label_leak_filter_catches_label_curie_and_camelcase():
    c = Concept(curie="INCIDENT:DB_OUTAGE", label="Database Outage", synonyms=["DB down"])
    assert label_leaks("Postgres primary unreachable, database outage declared", c)
    assert label_leaks("tag=INCIDENT:DB_OUTAGE", c)
    assert label_leaks("code db_outage fired", c)
    assert label_leaks("the db DOWN alarm", c)
    assert not label_leaks("Postgres primary unreachable from 3 app hosts, connection refused on 5432", c)
    c2 = Concept(curie="schema:LocalBusiness", label="LocalBusiness")
    assert label_leaks("A local business with opening hours", c2)
    assert not label_leaks("Open Mon-Fri 9-5 at 12 Main St, call 555-0100", c2)


def test_parsers_tolerate_fences_and_indices():
    assert parse_generations('```json\n{"observations": ["a long enough observation text", "x"]}\n```') == [
        "a long enough observation text"
    ]
    assert parse_label('{"index": 1, "confidence": 2, "reason": "r"}', ["A", "B"]) == ("B", 1.0, "r")
    assert parse_label('{"curie": "none", "confidence": 0.3}', ["A", "B"])[0] is None


def test_build_filters_leaks_and_disagreements(ontology):
    gens = {
        "INCIDENT:DB_OUTAGE": [
            "pg_isready failed on primary, 5432 connection refused for 4 minutes",
            "Database outage on the primary cluster",  # leaks the label
            "replica lag 900s then writes time out across the billing pool",
        ],
        "INCIDENT:DDOS": [
            "east-west packet loss 60% between az-a and az-b, BGP session flapping",
            "consul members show two leaders after the switch firmware upgrade",
        ],
    }
    # The labeller disagrees on one network text.
    stub = StubLLM(gens, {"consul members show two leaders after the switch firmware upgrade": "INCIDENT:DB_OUTAGE"})
    rows, stats = build_corpus(
        ontology,
        "incident",
        stub,
        generator_model="gen",
        labeller_model="lab",
        classes=list(gens),
        batches_per_class=1,
        per_batch=3,
        concurrency=2,
    )
    assert stats.generated == 5 and stats.leaked == 1 and stats.disagreed == 1 and stats.kept == 3
    assert {r.target_curie for r in rows} == set(gens)
    assert all(r.labeller_curie == r.target_curie for r in rows)
    assert all(r.target_curie in r.candidates and len(r.candidates) >= 5 for r in rows)
    assert stub.calls.count("gen") == 2 and stub.calls.count("lab") == 4
    assert all(r.generator_model == "gen" and r.labeller_model == "lab" and r.prompt_version for r in rows)


def _rows(ontology, n_batches=3):
    classes = sorted(c for c in ontology.concepts if c != "INCIDENT:ROOT")
    gens = {c: [f"observation {i} on host {k} for class number {k}" for i in range(2)] for k, c in enumerate(classes)}
    rows, _ = build_corpus(
        ontology,
        "incident",
        StubLLM(gens),
        generator_model="g",
        labeller_model="l",
        classes=classes,
        batches_per_class=n_batches,
        per_batch=2,
        concurrency=3,
    )
    return rows


def test_split_is_deterministic_batch_disjoint_and_hashed(ontology, tmp_path):
    rows = _rows(ontology)
    assert len(rows) == 12 * 3 * 2
    a = split_corpus(rows, seed=1)
    b = split_corpus(list(reversed(rows)), seed=1)
    assert {k: [r.row_id for r in v] for k, v in a.items()} == {k: [r.row_id for r in v] for k, v in b.items()}
    assert sum(len(v) for v in a.values()) == len(rows)
    batches = {k: {r.batch_id for r in v} for k, v in a.items()}
    names = list(batches)
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            assert not batches[x] & batches[y]
    assert split_corpus(rows, seed=2) != a

    report = split_report(a)
    assert report["train"]["classes"] >= 1 and "classes_unseen_in_train" in report["test"]

    path = tmp_path / "c.jsonl"
    write_jsonl(rows, path)
    back = read_jsonl(path)
    assert back == rows
    assert corpus_sha256(back) == corpus_sha256(list(reversed(rows)))
    changed = [CorpusRow.from_dict({**rows[0].to_dict(), "text": rows[0].text + "!"})] + rows[1:]
    assert corpus_sha256(changed) != corpus_sha256(rows)

    manifest = corpus_manifest(ontology, None, rows, a, None)
    assert manifest["corpus_sha256"] == corpus_sha256(rows)
    assert set(manifest["splits"]) == {"train", "validation", "calibration", "test"}
    assert "source_ids" not in manifest["splits"]["test"]
    assert "Not human" in manifest["label_provenance"]


def test_sample_classes_is_stratified_and_deterministic(ontology):
    pool = sample_classes(ontology, None)
    assert "INCIDENT:ROOT" in pool and len(pool) == 13
    chosen = sample_classes(ontology, 4, seed=3)
    assert chosen == sample_classes(ontology, 4, seed=3) and len(chosen) == 4
    assert "INCIDENT:ROOT" in chosen  # depth 0 is represented


def test_client_backs_off_on_429_and_counts_usage():
    answers = iter(
        [
            (429, b"slow down"),
            (
                200,
                json.dumps(
                    {"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
                ).encode(),
            ),
        ]
    )
    slept: list[float] = []
    client = OpenAICompatClient("https://example.invalid/v1", "k", sleep=slept.append, backoff_s=0.5)
    client.transport = lambda url, headers, body, timeout: next(answers)
    assert client.chat([{"role": "user", "content": "x"}], "m") == "{}"
    assert client.rate_limit_hits == 1 and len(slept) == 1 and client.prompt_tokens == 3
    assert "k" not in repr(client)
