"""Hermetic coverage for the model-backed code paths, with no downloads.

A tiny randomly initialised BERT plus the bundled ModernBERT tokenizer are saved to a temp dir and
loaded through the same ``from_pretrained`` calls the real checkpoints use. Covers
``olaya.embeddings`` (TransformerEncoder, EmbeddingCache, device resolution),
``olaya.embedding_router`` (finetune_bi_encoder on a plain transformer, load_finetuned_encoder),
``OLaya.from_ontology`` with a model id / checkpoint, and ``CrossEncoderReranker``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from olaya.embeddings import (  # noqa: E402
    EmbeddingCache,
    EncoderUnavailable,
    HashEncoder,
    TransformerEncoder,
    _resolve_commit,
    cache_dir,
    load_encoder,
    resolve_device,
    text_key,
)
from olaya.ontology import Concept, Ontology  # noqa: E402
from olaya.tokenizer import load_local_tokenizer  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from transformers import BertConfig, BertModel

    d = tmp_path_factory.mktemp("tiny-bert")
    tok = load_local_tokenizer()
    tok.save_pretrained(d)
    torch.manual_seed(0)
    cfg = BertConfig(
        vocab_size=len(tok),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    BertModel(cfg).save_pretrained(d)
    return d


def _onto() -> Ontology:
    onto = Ontology("pets")
    onto.add_concept(Concept("T:ROOT", "Thing"))
    onto.add_concept(Concept("T:ANIMAL", "Animal", "A living creature."))
    onto.add_concept(Concept("T:VEHICLE", "Vehicle", "A machine for transport."))
    onto.add_concept(Concept("T:CAT", "Cat", "A small feline that purrs.", synonyms=["kitty"]))
    onto.add_concept(Concept("T:DOG", "Dog", "A canine that barks."))
    onto.add_concept(Concept("T:TRUCK", "Truck", "A heavy goods vehicle."))
    for child, parent in [
        ("T:ANIMAL", "T:ROOT"),
        ("T:VEHICLE", "T:ROOT"),
        ("T:CAT", "T:ANIMAL"),
        ("T:DOG", "T:ANIMAL"),
        ("T:TRUCK", "T:VEHICLE"),
    ]:
        onto.add_subsumption(child, parent)
    onto.add_disjointness("T:ANIMAL", "T:VEHICLE")
    return onto


# --- TransformerEncoder + cache ---------------------------------------------------------------


def test_transformer_encoder_encodes_and_caches(tiny_model_dir, tmp_path):
    enc = TransformerEncoder(str(tiny_model_dir), cache_root=tmp_path, batch_size=2, threads=1, prefix="q: ")
    vecs = enc.encode(["a purring cat", "", "a barking dog", "a purring cat"])
    assert vecs.shape == (4, 32) and vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(vecs[0], vecs[3])  # duplicate texts run once
    assert enc.calls == 3
    enc.encode(["a purring cat"])
    assert enc.calls == 3  # served from the cache
    info = enc.info.to_dict()
    assert info["dim"] == 32 and info["prefix"] == "q: " and info["cache"]

    # A fresh encoder over the same cache root reads the flushed vectors back.
    again = TransformerEncoder(str(tiny_model_dir), cache_root=tmp_path, prefix="q: ")
    np.testing.assert_allclose(again.encode(["a barking dog"])[0], vecs[2], atol=1e-6)
    assert again.calls == 0

    uncached = load_encoder(str(tiny_model_dir), use_cache=False)
    assert uncached.cache is None and uncached.info.to_dict()["cache"] is None
    assert uncached.encode([]).shape == (0, 32)


def test_transformer_encoder_unavailable(tmp_path):
    with pytest.raises(EncoderUnavailable):
        TransformerEncoder(str(tmp_path / "does-not-exist"))


def test_embedding_cache_edge_cases(tmp_path, monkeypatch):
    cache = EmbeddingCache("org/model", "rev", tmp_path)
    assert len(cache) == 0 and cache.get("missing") is None
    cache.put_many([], np.zeros((0, 2)))
    cache.flush()  # nothing dirty: no files
    assert not (cache.dir / "index.json").exists()
    cache.put_many(["a", "b"], np.eye(2))
    cache.put_many(["a"], np.ones((1, 2)))  # already present: ignored
    cache.flush()
    assert len(EmbeddingCache("org/model", "rev", tmp_path)) == 2

    # Corrupt / mismatched caches are ignored rather than raised.
    (cache.dir / "index.json").write_text(json.dumps({"a": 0, "b": 1, "c": 2}))
    assert len(EmbeddingCache("org/model", "rev", tmp_path)) == 0
    (cache.dir / "index.json").write_text("{not json")
    assert EmbeddingCache("org/model", "rev", tmp_path).get("a") is None

    # A read-only cache dir must not break encoding.
    ro = EmbeddingCache("org/model", "ro", tmp_path)
    ro.put_many(["x"], np.ones((1, 2)))
    monkeypatch.setattr(type(ro.dir), "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    ro.flush()
    assert ro._dirty is True

    monkeypatch.setenv("OLAYA_CACHE_DIR", str(tmp_path / "custom"))
    assert cache_dir() == tmp_path / "custom" / "embeddings"
    assert text_key("x") == hashlib.sha256(b"x").hexdigest()


class _FakeBackend:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _fake_torch(cuda: bool, mps: bool):
    return types.SimpleNamespace(cuda=_FakeBackend(cuda), backends=types.SimpleNamespace(mps=_FakeBackend(mps)))


def test_resolve_device(monkeypatch):
    monkeypatch.delenv("OLAYA_ENCODER_DEVICE", raising=False)
    assert resolve_device(None, _fake_torch(False, False)) == "cpu"
    assert resolve_device("auto", _fake_torch(True, True)) == "cuda"
    assert resolve_device("auto", _fake_torch(False, True)) == "mps"
    assert resolve_device("auto", _fake_torch(False, False)) == "cpu"
    assert resolve_device("MPS", _fake_torch(False, True)) == "mps"
    with pytest.raises(EncoderUnavailable):
        resolve_device("cuda", _fake_torch(False, False))
    with pytest.raises(EncoderUnavailable):
        resolve_device("mps", _fake_torch(False, False))
    monkeypatch.setenv("OLAYA_ENCODER_DEVICE", "auto")
    assert resolve_device(None, _fake_torch(True, False)) == "cuda"


def test_resolve_commit_unknown_model_is_empty():
    assert _resolve_commit("olaya/definitely-not-cached-model", "main") == ""


def test_sentence_transformer_encoder_unavailable(monkeypatch):
    from olaya.embeddings import SentenceTransformerEncoder

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(EncoderUnavailable, match="sentence-transformers"):
        SentenceTransformerEncoder("google/embeddinggemma-2")


# --- fine-tuning on a plain transformer ------------------------------------------------------


def test_finetune_bi_encoder_transformer_path(tiny_model_dir, tmp_path):
    from olaya.embedding_router import EmbeddingRouter, finetune_bi_encoder, load_finetuned_encoder

    onto = _onto()
    enc = TransformerEncoder(str(tiny_model_dir), cache_root=tmp_path / "cache")
    pairs = [
        ("the kitty purrs", "T:CAT"),
        ("a dog barks loudly", "T:DOG"),
        ("an eighteen wheeler on the road", "T:TRUCK"),
        ("unknown class is skipped", "T:NOPE"),
    ]
    messages: list[str] = []

    def validate(encoder) -> float:
        router = EmbeddingRouter(onto, encoder)
        return float(router.rank("the kitty purrs")[0][0] == "T:CAT")

    ckpt = tmp_path / "ft.pt"
    result = finetune_bi_encoder(
        enc,
        onto,
        pairs,
        checkpoint=ckpt,
        validate=validate,
        max_epochs=2,
        batch_size=2,
        lr=1e-3,
        mine_hard_negatives=2,
        freeze_embeddings=True,
        patience=1,
        baseline_validation=-1.0,
        progress=messages.append,
    )
    assert ckpt.is_file() and result.sha256 == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert 1 <= result.epochs_run <= 2 and result.stop_reason in {"epochs", "patience"}
    assert messages and enc.cache is None  # the base-model cache is dropped once weights change

    reloaded = load_finetuned_encoder(ckpt)
    assert reloaded.name.startswith(f"{tiny_model_dir}+ft:") and reloaded.cache is None
    np.testing.assert_allclose(reloaded.encode(["the kitty purrs"]), enc._run(["the kitty purrs"]), atol=1e-5)

    # Budget stop on the first step.
    stopped = finetune_bi_encoder(
        TransformerEncoder(str(tiny_model_dir), use_cache=False),
        onto,
        pairs,
        checkpoint=tmp_path / "b.pt",
        budget_s=0.0,
    )
    assert stopped.stop_reason == "budget" and stopped.epochs_run == 1


def test_finetune_bi_encoder_argument_errors(tiny_model_dir, tmp_path):
    from olaya.embedding_router import finetune_bi_encoder

    enc = TransformerEncoder(str(tiny_model_dir), use_cache=False)
    with pytest.raises(ValueError, match="class_texts"):
        finetune_bi_encoder(enc, None, [("x", "T:CAT")], checkpoint=tmp_path / "x.pt")
    with pytest.raises(ValueError, match="hard_negatives"):
        finetune_bi_encoder(enc, None, [("x", "T:CAT")], checkpoint=tmp_path / "x.pt", class_texts={"T:CAT": "Cat"})
    with pytest.raises(ValueError, match="no training pairs"):
        finetune_bi_encoder(enc, _onto(), [("x", "T:NOPE")], checkpoint=tmp_path / "x.pt")


def test_load_finetuned_encoder_rejects_foreign_files(tmp_path):
    from olaya.embedding_router import load_finetuned_encoder

    bad = tmp_path / "bad.pt"
    torch.save({"format": "something-else"}, bad)
    with pytest.raises(ValueError, match="not an olaya bi-encoder"):
        load_finetuned_encoder(bad)


# --- engine construction from a model id / checkpoint ----------------------------------------


def test_from_ontology_with_model_id_and_checkpoint(tiny_model_dir, tmp_path, tokenizer):
    from olaya.embedding_router import finetune_bi_encoder
    from olaya.engine import OLaya

    onto = _onto()
    engine = OLaya.from_ontology(
        onto,
        encoder=str(tiny_model_dir),
        encoder_kwargs={"use_cache": False},
        router="embedding",
        tokenizer=tokenizer,
        device="cpu",
    )
    answer = engine.system_one("a purring kitty", {"q": {"type": "concept", "root": "T:ROOT"}}).answers["q"]
    assert answer.axioms_verified is True and answer.lineage[0] == "T:ROOT"

    ckpt = tmp_path / "pack.pt"
    finetune_bi_encoder(
        TransformerEncoder(str(tiny_model_dir), use_cache=False),
        onto,
        [("kitty", "T:CAT"), ("dog", "T:DOG")],
        checkpoint=ckpt,
        max_epochs=1,
    )
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    tuned = OLaya.from_ontology(onto, checkpoint=ckpt, checkpoint_sha256=sha, tokenizer=tokenizer, device="cpu")
    assert "+ft:" in tuned.model_id
    with pytest.raises(FileNotFoundError):
        OLaya.from_ontology(onto, checkpoint=tmp_path / "missing.pt", tokenizer=tokenizer)

    # No tokenizer given: the bundled local tokenizer is used when the repository ships it.
    local = OLaya.from_ontology(onto, encoder=HashEncoder(), device="cpu")
    assert local.tokenizer is not None


# --- cross-encoder reranker ------------------------------------------------------------------


def test_cross_encoder_reranker_scores_trains_and_saves(tiny_model_dir, tmp_path):
    from olaya.benchmarks.oaei import EntityText
    from olaya.benchmarks.reranker import CrossEncoderReranker, Pair, TrainingConfig

    rr = CrossEncoderReranker(tiny_model_dir, threads=1, seed=0, batch_size=2)
    texts = [("heart.", "Heart."), ("heart.", "Toe."), ("toe.", "Toe."), ("toe.", "Heart.")]
    probs = rr.score(texts)
    assert probs.shape == (4,) and 0.0 <= probs.min() and probs.max() <= 1.0

    src = {"a": EntityText("a", "class", "heart", "heart"), "b": EntityText("b", "class", "toe", "toe")}
    tgt = {"x": EntityText("x", "class", "Heart", "Heart"), "y": EntityText("y", "class", "Toe", "Toe")}
    scored = rr.score_pairs({("a", "x"), ("b", "y")}, src, tgt)
    assert set(scored) == {("a", "x"), ("b", "y")}

    pairs = [Pair("a", "x", 1), Pair("a", "y", 0), Pair("b", "y", 1), Pair("b", "x", 0)]
    cfg = TrainingConfig(base_model=str(tiny_model_dir), epochs=2, batch_size=2, warmup_fraction=0.5, log_every=1)
    stats = rr.train(pairs, texts, cfg, validation=(pairs, texts))
    assert stats.steps == 4 and len(stats.validation) == 2 and not stats.stopped_by_budget
    assert stats.to_dict()["epoch_losses"]

    budget = rr.train(pairs, texts, TrainingConfig(epochs=1, batch_size=2, time_budget_s=-1.0))
    assert budget.stopped_by_budget and budget.steps == 0

    out = rr.save(tmp_path / "rr", {"note": "tiny"})
    reloaded = CrossEncoderReranker(out)
    assert reloaded.manifest == {"note": "tiny"}
    np.testing.assert_allclose(reloaded.score(texts), rr.score(texts), atol=1e-5)
