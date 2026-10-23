"""Hermetic coverage of the EmbeddingGemma 2 code path (sentence-transformers, prompted encoders).

EmbeddingGemma 2 is the default encoder (``olaya.embeddings.DEFAULT_ENCODER``) and ships with
``leio-decision[ml]``. A tiny randomly initialised BERT wrapped as a prompted SentenceTransformer
(query / document prompts, mean pooling, normalisation) exercises the same code without downloads:
``SentenceTransformerEncoder``, fine-tuning on the prompted path (full and LoRA), checkpoint reload,
and ``OLaya.from_ontology`` with the default and an explicit ``embeddinggemma`` model id.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

from olaya.embeddings import DEFAULT_ENCODER, HashEncoder, SentenceTransformerEncoder  # noqa: E402
from olaya.ontology import Concept, Ontology  # noqa: E402
from olaya.tokenizer import load_local_tokenizer  # noqa: E402

QUERY_PROMPT = "task: search result | query: "
DOCUMENT_PROMPT = "title: none | text: "


def _st_modules():
    try:
        from sentence_transformers.sentence_transformer import modules
    except ImportError:  # older sentence-transformers
        from sentence_transformers import models as modules
    return modules


@pytest.fixture(scope="module")
def tiny_gemma_dir(tmp_path_factory):
    """A prompted SentenceTransformer saved under a name containing ``embeddinggemma``."""
    from sentence_transformers import SentenceTransformer
    from transformers import BertConfig, BertModel

    root = tmp_path_factory.mktemp("st")
    base = root / "base"
    tok = load_local_tokenizer()
    tok.save_pretrained(base)
    torch.manual_seed(0)
    BertModel(
        BertConfig(
            vocab_size=len(tok),
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=128,
        )
    ).save_pretrained(base)
    modules = _st_modules()
    word = modules.Transformer(str(base), max_seq_length=64)
    dim = (
        word.get_embedding_dimension()
        if hasattr(word, "get_embedding_dimension")
        else word.get_word_embedding_dimension()
    )
    st = SentenceTransformer(
        modules=[word, modules.Pooling(dim, pooling_mode="mean"), modules.Normalize()],
        prompts={"query": QUERY_PROMPT, "document": DOCUMENT_PROMPT},
    )
    out = root / "tiny-embeddinggemma"
    st.save(str(out))
    return out


def _onto() -> Ontology:
    onto = Ontology("pets")
    onto.add_concept(Concept("T:ROOT", "Thing"))
    onto.add_concept(Concept("T:ANIMAL", "Animal"))
    onto.add_concept(Concept("T:VEHICLE", "Vehicle"))
    onto.add_concept(Concept("T:CAT", "Cat", "A small feline that purrs."))
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


def test_default_encoder_is_embeddinggemma_2():
    assert DEFAULT_ENCODER == "google/embeddinggemma-2"


def test_prompted_encoder_query_and_document_sides(tiny_gemma_dir):
    enc = SentenceTransformerEncoder(str(tiny_gemma_dir), device="cpu", threads=1, batch_size=2)
    assert enc.query_prefix == QUERY_PROMPT and enc.document_prefix == DOCUMENT_PROMPT and enc.prefix == QUERY_PROMPT
    q = enc.encode(["a purring cat", ""])
    d = enc.encode_documents(["a purring cat"])
    assert q.shape == (2, 32) and d.shape == (1, 32)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(q[0], d[0])  # different prompts, different embeddings
    assert enc.encode([]).shape == (0, 32) and enc.calls == 3
    info = enc.info.to_dict()
    assert info["backend"] == "sentence-transformers" and info["document_prefix"] == DOCUMENT_PROMPT
    assert enc.model is enc.st

    emb = enc.embed_batch(["a cat"], documents=True)
    assert emb.requires_grad and emb.shape == (1, 32)

    small = SentenceTransformerEncoder(
        str(tiny_gemma_dir), device="cpu", truncate_dim=16, dtype="float32", text_only=False, query_prompt=None
    )
    assert small.dim == 16 and small.query_prefix == "" and small.encode(["x"]).shape == (1, 16)
    assert small.embed_batch(["x"]).shape == (1, 16)


@pytest.mark.parametrize("lora_rank", [0, 4])
def test_finetune_prompted_encoder_and_reload(tiny_gemma_dir, tmp_path, lora_rank):
    from olaya.embedding_router import EmbeddingRouter, finetune_bi_encoder, load_finetuned_encoder

    if lora_rank:
        pytest.importorskip("peft")
    onto = _onto()
    enc = SentenceTransformerEncoder(str(tiny_gemma_dir), device="cpu")
    ckpt = tmp_path / "ft.pt"
    result = finetune_bi_encoder(
        enc,
        onto,
        [("the kitty purrs", "T:CAT"), ("a dog barks", "T:DOG"), ("a lorry on the road", "T:TRUCK")],
        checkpoint=ckpt,
        validate=lambda e: float(EmbeddingRouter(onto, e).rank("the kitty purrs")[0][0] == "T:CAT"),
        max_epochs=1,
        batch_size=3,
        lr=1e-3,
        lora_rank=lora_rank,
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert payload["format"] == "olaya-st-encoder-v1" and payload["document_prefix"] == DOCUMENT_PROMPT
    assert result.sha256 == hashlib.sha256(ckpt.read_bytes()).hexdigest()

    reloaded = load_finetuned_encoder(ckpt, device="cpu")
    assert isinstance(reloaded, SentenceTransformerEncoder)
    assert reloaded.name.startswith(f"{tiny_gemma_dir}+ft:")
    assert reloaded.query_prefix == QUERY_PROMPT and reloaded.document_prefix == DOCUMENT_PROMPT
    np.testing.assert_allclose(reloaded.encode(["the kitty purrs"]), enc.encode(["the kitty purrs"]), atol=1e-5)


def test_from_ontology_uses_embeddinggemma_by_default(monkeypatch, tokenizer):
    import olaya.embeddings as embeddings
    from olaya.engine import OLaya

    seen: list[str] = []

    class RecordingEncoder(HashEncoder):
        def __init__(self, model, **kwargs):
            seen.append(model)
            super().__init__(dim=64)

    monkeypatch.setattr(embeddings, "SentenceTransformerEncoder", RecordingEncoder)
    engine = OLaya.from_ontology(_onto(), tokenizer=tokenizer, device="cpu")
    assert seen == [DEFAULT_ENCODER]
    assert engine.system_one("kitty", {"q": {"type": "concept", "root": "T:ROOT"}}).answers["q"].axioms_verified


def test_from_ontology_with_embeddinggemma_model_id(tiny_gemma_dir, tokenizer):
    from olaya.engine import OLaya

    engine = OLaya.from_ontology(
        _onto(), encoder=str(tiny_gemma_dir), encoder_kwargs={"device": "cpu"}, tokenizer=tokenizer, device="cpu"
    )
    assert "embedding:" in engine.model_id and str(tiny_gemma_dir) in engine.model_id
    answer = engine.system_one("a purring kitty", {"q": {"type": "concept", "root": "T:ROOT"}}).answers["q"]
    assert answer.axioms_verified is True
