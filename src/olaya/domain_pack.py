"""Domain packs: everything client-specific, kept outside this package and loaded at runtime.

O-Laya ships no client ontologies, evidence or checkpoints. A *domain pack* is a directory owned by
the consumer (typically in their own private repository) with a ``pack.json`` manifest::

    {
      "format": "olaya-domain-pack-v1",
      "name": "acme-orders",
      "ontology": "ontology.ttl",                    # any format olaya.ontology_io.load_ontology reads
      "ontology_fingerprint": "<sha256>",            # optional: olaya.ontology_io.ontology_fingerprint
      "encoder": {
        "model": "google/embeddinggemma-2",          # base encoder when there is no checkpoint
        "checkpoint": "checkpoints/acme.pt",         # optional fine-tune (finetune_bi_encoder output)
        "checkpoint_sha256": "<sha256>",             # optional, verified before loading
        "temperature": 0.05                          # optional scorer temperature
      },
      "router": "hybrid",                            # "hybrid" (lexical + embedding RRF) or "embedding"
      "candidate_policy": "routed",
      "max_candidates": 12
    }

Relative paths resolve against the pack directory. ``OLaya.from_pack(path)`` builds the engine;
:func:`load_domain_pack` only parses and validates the manifest (no model download), so packs can
be checked in CI without an encoder.

The engine is *conditioned* on the pack at instantiation: class texts are built from the ontology
and embedded once, the router ranks classes against the observation, and the axiom projection runs on
the pack's DAG. A checkpoint only improves ranking; correctness (monotonicity, disjointness,
``axioms_verified``) comes from the ontology and the verified kernel, never from the weights.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PACK_FORMAT = "olaya-domain-pack-v1"
PACK_MANIFEST = "pack.json"
ROUTERS = frozenset({"hybrid", "embedding"})


@dataclass(frozen=True)
class DomainPack:
    """A parsed ``pack.json``; paths are absolute."""

    name: str
    root: Path
    ontology: Path
    ontology_fingerprint: str | None = None
    encoder_model: str | None = None
    checkpoint: Path | None = None
    checkpoint_sha256: str | None = None
    temperature: float | None = None
    router: str = "hybrid"
    candidate_policy: str = "routed"
    max_candidates: int = 12
    extra: dict[str, Any] = field(default_factory=dict)

    def engine_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :meth:`olaya.engine.OLaya.from_ontology`."""
        return {
            "encoder": self.encoder_model,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "expected_fingerprint": self.ontology_fingerprint,
            "router": self.router,
            "temperature": self.temperature,
            "candidate_policy": self.candidate_policy,
            "max_candidates": self.max_candidates,
        }


def load_domain_pack(path: str | Path) -> DomainPack:
    """Parse ``path`` (a pack directory or its ``pack.json``) into a :class:`DomainPack`."""
    path = Path(path)
    manifest = path / PACK_MANIFEST if path.is_dir() else path
    if not manifest.is_file():
        raise FileNotFoundError(f"no {PACK_MANIFEST} at {path}")
    root = manifest.parent.resolve()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("format") != PACK_FORMAT:
        raise ValueError(f"{manifest}: format must be {PACK_FORMAT!r}, got {data.get('format')!r}")
    for key in ("name", "ontology"):
        if not data.get(key):
            raise ValueError(f"{manifest}: missing {key!r}")

    def resolve(p: str | None) -> Path | None:
        if not p:
            return None
        q = Path(p).expanduser()
        return q if q.is_absolute() else (root / q)

    ontology = resolve(data["ontology"])
    assert ontology is not None
    if not ontology.exists():
        raise FileNotFoundError(f"{manifest}: ontology {ontology} not found")
    enc = data.get("encoder") or {}
    router = data.get("router", "hybrid")
    if router not in ROUTERS:
        raise ValueError(f"{manifest}: router must be one of {sorted(ROUTERS)}, got {router!r}")
    known = {
        "format",
        "name",
        "ontology",
        "ontology_fingerprint",
        "encoder",
        "router",
        "candidate_policy",
        "max_candidates",
    }
    return DomainPack(
        name=data["name"],
        root=root,
        ontology=ontology,
        ontology_fingerprint=data.get("ontology_fingerprint"),
        encoder_model=enc.get("model"),
        checkpoint=resolve(enc.get("checkpoint")),
        checkpoint_sha256=enc.get("checkpoint_sha256"),
        temperature=enc.get("temperature"),
        router=router,
        candidate_policy=data.get("candidate_policy", "routed"),
        max_candidates=int(data.get("max_candidates", 12)),
        extra={k: v for k, v in data.items() if k not in known},
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = ["PACK_FORMAT", "PACK_MANIFEST", "ROUTERS", "DomainPack", "file_sha256", "load_domain_pack"]
