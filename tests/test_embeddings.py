"""Tests for ``olaya.embeddings``: the stub encoder, the on-disk cache and (opt-in) the real model.

The real checkpoint is only exercised when ``OLAYA_RUN_EMBEDDING_TESTS=1`` is set, so
CI stays network-free; if the model cannot be loaded the test skips cleanly.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from olaya.embeddings import (
    DEFAULT_MODEL,
    EmbeddingCache,
    Encoder,
    EncoderUnavailable,
    HashEncoder,
    cache_dir,
    text_key,
)


def test_hash_encoder_is_deterministic_and_normalised():
    enc = HashEncoder(dim=32)
    a = enc.encode(["Conference Paper", "paper", ""])
    b = HashEncoder(dim=32).encode(["Conference Paper", "paper", ""])
    assert a.shape == (3, 32) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert np.linalg.norm(a[0]) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(a[1]) == pytest.approx(1.0, abs=1e-5)
    assert np.all(a[2] == 0.0)  # empty text -> zero vector
    assert isinstance(enc, Encoder)


def test_hash_encoder_shares_mass_on_shared_tokens():
    enc = HashEncoder(dim=64)
    v = enc.encode(["conference paper", "paper", "banquet dinner"])
    assert v[0] @ v[1] > 0.5  # one of two tokens shared
    assert abs(v[0] @ v[2]) < 0.4  # disjoint tokens: near-orthogonal


def test_cache_dir_honours_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OLAYA_CACHE_DIR", str(tmp_path))
    assert cache_dir() == tmp_path / "embeddings"
    monkeypatch.delenv("OLAYA_CACHE_DIR")
    assert cache_dir() == Path.home() / ".cache" / "olaya" / "embeddings"


def test_embedding_cache_roundtrip(tmp_path: Path):
    cache = EmbeddingCache("org/model", "abc123", root=tmp_path)
    assert len(cache) == 0 and cache.get(text_key("x")) is None
    keys = [text_key("x"), text_key("y")]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cache.put_many(keys, vecs)
    cache.put_many(keys, vecs)  # duplicates are ignored
    assert len(cache) == 2
    cache.flush()
    assert (tmp_path / "org__model" / "abc123" / "vectors.npy").exists()

    reloaded = EmbeddingCache("org/model", "abc123", root=tmp_path)
    assert len(reloaded) == 2
    assert np.array_equal(reloaded.get(keys[1]), vecs[1])
    assert reloaded.get(text_key("z")) is None


def test_embedding_cache_ignores_corrupt_files(tmp_path: Path):
    d = tmp_path / "m" / "r"
    d.mkdir(parents=True)
    (d / "index.json").write_text("{not json")
    (d / "vectors.npy").write_bytes(b"garbage")
    cache = EmbeddingCache("m", "r", root=tmp_path)
    assert len(cache) == 0


@pytest.mark.skipif(
    os.environ.get("OLAYA_RUN_EMBEDDING_TESTS") != "1",
    reason="set OLAYA_RUN_EMBEDDING_TESTS=1 to exercise the real checkpoint (may download)",
)
def test_transformer_encoder_real_model(tmp_path: Path):
    from olaya.embeddings import load_encoder

    try:
        enc = load_encoder(DEFAULT_MODEL, cache_root=tmp_path)
    except EncoderUnavailable as exc:
        pytest.skip(f"model not loadable: {exc}")
    v = enc.encode(["Paper", "Article", "Dinner banquet"])
    assert v.shape == (3, enc.dim)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2]  # semantic neighbours score higher
    # Second call is served from the on-disk cache: deterministic and no model calls.
    calls = enc.calls
    again = enc.encode(["Paper"])
    assert enc.calls == calls
    assert np.allclose(again[0], v[0], atol=1e-6)
    assert enc.info.to_dict()["model"] == DEFAULT_MODEL
