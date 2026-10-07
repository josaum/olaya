"""Small local text encoder: mean-pooled, L2-normalised sentence embeddings.

Built on ``transformers`` + ``torch`` only (no ``sentence-transformers`` dependency).
The default checkpoint is ``sentence-transformers/all-MiniLM-L6-v2`` (22M parameters,
384-d), which runs comfortably on a CPU.

Design points:

* **Deterministic.**  Inference only, ``torch.no_grad``, fixed thread count, no dropout,
  and the mean pooling + normalisation is done in float32.
* **Offline friendly.**  ``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1`` are honoured by
  ``transformers``; :func:`load_encoder` reports a clean :class:`EncoderUnavailable`
  instead of a stack trace when the checkpoint cannot be loaded.
* **On-disk cache.**  Every embedding is cached under ``$OLAYA_CACHE_DIR/embeddings``
  (default ``~/.cache/olaya/embeddings``) keyed by ``(model id, revision, text sha256)``
  so a repeated benchmark run does not touch the model at all.

The :class:`Encoder` protocol is what consumers (e.g. the OAEI benchmark's
``EmbeddingMatcher``) depend on; tests substitute a tiny deterministic encoder.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: Checkpoints that expect an instruction prefix.  Both symmetric-similarity models
#: (``bge``) and ``e5`` recommend ``"query: "`` for symmetric tasks; MiniLM uses none.
_PREFIXES = {
    "intfloat/e5-small-v2": "query: ",
    "intfloat/e5-base-v2": "query: ",
    "BAAI/bge-small-en-v1.5": "",
}


class EncoderUnavailable(RuntimeError):
    """Raised when the checkpoint cannot be loaded (offline, missing, broken install)."""


@runtime_checkable
class Encoder(Protocol):
    """Anything that maps a list of strings to an ``(n, d)`` L2-normalised float32 array."""

    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def cache_dir() -> Path:
    base = os.environ.get("OLAYA_CACHE_DIR")
    root = Path(base).expanduser() if base else Path.home() / ".cache" / "olaya"
    return root / "embeddings"


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Append-only on-disk cache of embeddings for one ``(model, revision)``.

    Layout: ``<cache_dir>/<model-slug>/<revision>/<first 2 hex of sha>.npz`` would be
    many small files; instead we keep one ``index.json`` (sha -> row) plus one
    ``vectors.npy`` per model, loaded lazily and rewritten on :meth:`flush`.
    """

    def __init__(self, model: str, revision: str, root: Path | None = None):
        slug = model.replace("/", "__")
        self.dir = (root or cache_dir()) / slug / (revision or "main")
        self._index: dict[str, int] | None = None
        self._vectors: np.ndarray | None = None
        self._dirty = False

    def _load(self) -> None:
        if self._index is not None:
            return
        idx, vec = self.dir / "index.json", self.dir / "vectors.npy"
        if idx.exists() and vec.exists():
            try:
                self._index = json.loads(idx.read_text())
                self._vectors = np.load(vec)
                if self._vectors.shape[0] != len(self._index):
                    raise ValueError("cache index / vectors mismatch")
                return
            except Exception:  # noqa: BLE001 - a corrupt cache is simply ignored
                pass
        self._index, self._vectors = {}, None

    def get(self, key: str) -> np.ndarray | None:
        self._load()
        assert self._index is not None
        row = self._index.get(key)
        if row is None or self._vectors is None:
            return None
        return self._vectors[row]

    def put_many(self, keys: Sequence[str], vectors: np.ndarray) -> None:
        if not keys:
            return
        self._load()
        assert self._index is not None
        fresh = [(k, v) for k, v in zip(keys, vectors, strict=True) if k not in self._index]
        if not fresh:
            return
        block = np.stack([v for _, v in fresh]).astype(np.float32)
        start = 0 if self._vectors is None else self._vectors.shape[0]
        for i, (k, _) in enumerate(fresh):
            self._index[k] = start + i
        self._vectors = block if self._vectors is None else np.concatenate([self._vectors, block])
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty or self._index is None or self._vectors is None:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp_vec = self.dir / "vectors.tmp.npy"
            with open(tmp_vec, "wb") as fh:
                np.save(fh, self._vectors)
            tmp_vec.replace(self.dir / "vectors.npy")
            tmp_idx = self.dir / "index.json.tmp"
            tmp_idx.write_text(json.dumps(self._index))
            tmp_idx.replace(self.dir / "index.json")
            self._dirty = False
        except OSError:
            pass  # a read-only cache dir must not break encoding

    def __len__(self) -> int:
        self._load()
        assert self._index is not None
        return len(self._index)


@dataclass
class EncoderInfo:
    """Provenance of a loaded checkpoint (written into benchmark result files)."""

    model: str
    revision: str
    commit_sha: str
    dim: int
    max_length: int
    prefix: str
    pooling: str = "mean"
    normalize: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "revision": self.revision,
            "commit_sha": self.commit_sha,
            "dim": self.dim,
            "max_length": self.max_length,
            "prefix": self.prefix,
            "pooling": self.pooling,
            "normalize": self.normalize,
            **self.extra,
        }


class TransformerEncoder:
    """Mean-pooled, L2-normalised embeddings from a Hugging Face checkpoint (CPU, batched)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        revision: str = "main",
        max_length: int = 128,
        batch_size: int = 64,
        prefix: str | None = None,
        cache_root: Path | None = None,
        use_cache: bool = True,
        threads: int | None = None,
        device: str | None = None,
    ):
        """``device``: ``"cpu"`` (default), ``"mps"``, ``"cuda"`` or ``"auto"`` (the best available).
        Defaults to the ``OLAYA_ENCODER_DEVICE`` environment variable when set. Embeddings are
        returned on the host as float32 whatever the device; the cache is device-independent."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            raise EncoderUnavailable(f"torch/transformers not importable: {exc}") from exc

        self._torch = torch
        torch.manual_seed(0)
        if threads:
            torch.set_num_threads(threads)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
            self.model = AutoModel.from_pretrained(model, revision=revision)
        except Exception as exc:  # noqa: BLE001 - network / missing files / bad checkpoint
            raise EncoderUnavailable(f"cannot load {model}@{revision}: {exc}") from exc
        self.device = resolve_device(device, torch)
        self.model.to(self.device)
        self.model.eval()
        self.name = model
        self.revision = revision
        self.max_length = max_length
        self.batch_size = batch_size
        self.prefix = _PREFIXES.get(model, "") if prefix is None else prefix
        self.commit_sha = _resolve_commit(model, revision)
        self.dim = int(self.model.config.hidden_size)
        self.cache = EmbeddingCache(model, self.commit_sha or revision, cache_root) if use_cache else None
        self.calls = 0  # number of texts actually run through the model (cache misses)

    @property
    def info(self) -> EncoderInfo:
        return EncoderInfo(
            model=self.name,
            revision=self.revision,
            commit_sha=self.commit_sha,
            dim=self.dim,
            max_length=self.max_length,
            prefix=self.prefix,
            extra={"batch_size": self.batch_size, "cache": str(self.cache.dir) if self.cache is not None else None},
        )

    def _run(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = [self.prefix + t for t in texts[i : i + self.batch_size]]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state.float()
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.cpu().numpy().astype(np.float32))
        self.calls += len(texts)
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [t if t else " " for t in texts]
        result = np.zeros((len(texts), self.dim), dtype=np.float32)
        missing: dict[str, list[int]] = {}
        for i, t in enumerate(texts):
            key = text_key(self.prefix + t)
            cached = self.cache.get(key) if self.cache is not None else None
            if cached is not None:
                result[i] = cached
            else:
                missing.setdefault(t, []).append(i)
        if missing:
            uniq = sorted(missing)
            vecs = self._run(uniq)
            for t, v in zip(uniq, vecs, strict=True):
                for i in missing[t]:
                    result[i] = v
            if self.cache is not None:
                self.cache.put_many([text_key(self.prefix + t) for t in uniq], vecs)
                self.cache.flush()
        return result


def resolve_device(device: str | None, torch: Any) -> str:
    """Pick the torch device for an encoder: explicit > ``OLAYA_ENCODER_DEVICE`` > ``cpu``.

    ``"auto"`` prefers CUDA, then Apple MPS, then CPU. An unavailable explicit device is an error
    rather than a silent fallback, so a run that was meant to be fast never quietly becomes slow.
    """
    import os

    choice = (device or os.environ.get("OLAYA_ENCODER_DEVICE") or "cpu").lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise EncoderUnavailable("device 'cuda' requested but CUDA is not available")
    if choice == "mps" and not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        raise EncoderUnavailable("device 'mps' requested but Apple MPS is not available")
    return choice


def _resolve_commit(model: str, revision: str) -> str:
    """Commit sha of the locally cached snapshot (no network), ``""`` if unknown."""
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(model, "config.json", revision=revision)
        if isinstance(path, str):
            # .../snapshots/<sha>/config.json
            return Path(path).parent.name
    except Exception:  # noqa: BLE001
        pass
    return ""


def load_encoder(model: str = DEFAULT_MODEL, **kwargs: Any) -> TransformerEncoder:
    """Load a :class:`TransformerEncoder`, raising :class:`EncoderUnavailable` on failure."""
    return TransformerEncoder(model, **kwargs)


class HashEncoder:
    """Deterministic, dependency-free stand-in encoder for tests and offline runs.

    Each lowercase alphanumeric token is hashed to a pseudo-random unit vector; a text
    is the normalised sum of its token vectors.  Texts sharing tokens have positive
    cosine similarity, disjoint texts are (nearly) orthogonal.  Not a semantic model.
    """

    def __init__(self, dim: int = 64, seed: int = 0):
        self.name = f"hash-{dim}"
        self.dim = dim
        self.seed = seed

    def _token_vec(self, tok: str) -> np.ndarray:
        h = int.from_bytes(hashlib.sha256(f"{self.seed}:{tok}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(h)
        v = rng.standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import re

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = re.findall(r"[a-z0-9]+", t.lower())
            if not toks:
                continue
            v = np.sum([self._token_vec(tok) for tok in toks], axis=0)
            n = np.linalg.norm(v)
            out[i] = v / n if n else v
        return out


__all__ = [
    "DEFAULT_MODEL",
    "EmbeddingCache",
    "Encoder",
    "EncoderInfo",
    "resolve_device",
    "EncoderUnavailable",
    "HashEncoder",
    "TransformerEncoder",
    "cache_dir",
    "load_encoder",
    "text_key",
]
