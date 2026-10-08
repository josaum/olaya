"""Inference latency of the concept path per router / scorer configuration.

Measures ``OLaya.system_one`` with one ``concept`` question over each committed ontology for:

* ``lexical``: the default engine (lexical router, no scorer injected); the neural head runs with
  ``weights_path=False`` so this is the routing + projection cost, not a trained head;
* ``bge-small``: embedding router + scorer on the fine-tuned joint bge-small checkpoint when
  present (``models/checkpoints/olaya_embed_joint60.pt``), else zero-shot ``BAAI/bge-small-en-v1.5``;
* ``embeddinggemma-2``: the same with ``google/embeddinggemma-2`` (text-only), fine-tuned joint
  checkpoint when present (``olaya_embed_joint-gemma.pt``), else zero-shot; skipped when the
  ``gemma`` extra is not installed.

Class texts are embedded once at load (reported as index time); the per-decision cost is one query
encode plus routing, scoring and the axiom projection. Numbers are for the machine that ran the
script; the Rust runtime's lexical path is measured by ``cargo run --release -p olaya-core --bin
bench_arrow``. Writes ``benchmarks/results/router-latency.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OLAYA_TORCH_THREADS", "4")

import torch
from scripts.train_from_corpus import CONCEPT_ROOT, ONTOLOGIES

from olaya.corpus import read_jsonl
from olaya.embedding_router import EmbeddingRouter, EmbeddingScorer, build_class_texts, load_finetuned_encoder
from olaya.embeddings import EncoderUnavailable, TransformerEncoder
from olaya.engine import OLaya
from olaya.ontology import Ontology
from olaya.tokenizer import load_local_tokenizer

torch.set_num_threads(int(os.environ["OLAYA_TORCH_THREADS"]))

CHECKPOINTS = ROOT / "models" / "checkpoints"
CONFIGS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", CHECKPOINTS / "olaya_embed_joint60.pt"),
    "embeddinggemma-2": ("google/embeddinggemma-2", CHECKPOINTS / "olaya_embed_joint-gemma.pt"),
}


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]


def load_encoder(key: str) -> tuple[Any, str]:
    model, ckpt = CONFIGS[key]
    if ckpt.is_file():
        return load_finetuned_encoder(ckpt, threads=int(os.environ["OLAYA_TORCH_THREADS"]), batch_size=32), "fine-tuned"
    if key == "embeddinggemma-2":
        from olaya.embeddings import SentenceTransformerEncoder

        return SentenceTransformerEncoder(
            model, threads=int(os.environ["OLAYA_TORCH_THREADS"]), batch_size=32
        ), "zero-shot"
    return TransformerEncoder(
        model, threads=int(os.environ["OLAYA_TORCH_THREADS"]), batch_size=32, use_cache=False
    ), "zero-shot"


def measure(engine: OLaya, texts: list[str], root: str | None, runs: int) -> dict[str, Any]:
    question = {"q": {"type": "concept", "root": root}}
    for t in texts[:3]:
        engine.system_one(t, question)
    lat: list[float] = []
    for i in range(runs):
        t = texts[i % len(texts)]
        # Fresh text per call where possible so the router's query cache does not hide the encode cost.
        if hasattr(engine.router, "_query_cache"):
            engine.router._query_cache.pop(t, None)
        t0 = time.perf_counter()
        engine.system_one(t, question)
        lat.append((time.perf_counter() - t0) * 1000.0)
    return {
        "runs": runs,
        "median_ms": statistics.median(lat),
        "mean_ms": statistics.mean(lat),
        "p95_ms": percentile(lat, 0.95),
        "min_ms": min(lat),
        "decisions_per_s": 1000.0 / statistics.median(lat),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--configs", nargs="*", default=["lexical", *CONFIGS], choices=["lexical", *CONFIGS])
    parser.add_argument("--results-dir", type=Path, default=ROOT / "benchmarks" / "results")
    args = parser.parse_args()
    tokenizer = load_local_tokenizer()
    report: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cpu_count": os.cpu_count(),
            "device": os.environ.get("OLAYA_ENCODER_DEVICE", "cpu"),
        },
        "runs": args.runs,
        "domains": {},
    }
    encoders: dict[str, tuple[Any, str]] = {}
    for key in args.configs:
        if key == "lexical":
            continue
        try:
            encoders[key] = load_encoder(key)
        except (EncoderUnavailable, ImportError) as exc:
            print(f"skip {key}: {exc}", file=sys.stderr)
            report.setdefault("skipped", {})[key] = str(exc)
    for domain, path in sorted(ONTOLOGIES.items()):
        ontology = Ontology.from_json(path.read_text(encoding="utf-8"))
        rows = read_jsonl(ROOT / "data" / "corpora" / f"{domain}.jsonl")
        texts = [r.text for r in rows[:200]]
        root = CONCEPT_ROOT.get(domain)
        entry: dict[str, Any] = {"concepts": len(ontology.concepts), "texts": len(texts), "configs": {}}
        if "lexical" in args.configs:
            engine = OLaya(ontology=ontology, tokenizer=tokenizer, device="cpu", weights_path=False)
            entry["configs"]["lexical"] = {
                "weights": "none (lexical router, untrained head)",
                **measure(engine, texts, root, args.runs),
            }
        for key, (encoder, weights) in encoders.items():
            t0 = time.perf_counter()
            router = EmbeddingRouter(ontology, encoder, class_texts=build_class_texts(ontology))
            index_s = time.perf_counter() - t0
            engine = OLaya(
                ontology=ontology,
                tokenizer=tokenizer,
                device="cpu",
                weights_path=False,
                router=router,
                scorer=EmbeddingScorer(router, temperature=0.05),
                candidate_policy="routed",
            )
            entry["configs"][key] = {
                "weights": weights,
                "encoder": getattr(encoder, "name", "?"),
                "index_s": index_s,
                **measure(engine, texts, root, args.runs),
            }
        report["domains"][domain] = entry
        print(domain, {k: f"{v['median_ms']:.1f} ms" for k, v in entry["configs"].items()})
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "router-latency.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.results_dir / "router-latency.md").write_text(render(report))
    print("wrote", args.results_dir / "router-latency.md")


def render(r: dict[str, Any]) -> str:
    env = r["environment"]
    lines = [
        "# Concept-path inference latency per router / scorer",
        "",
        f"`OLaya.system_one`, one `concept` question, {r['runs']} timed decisions per cell after warm-up, fresh state text",
        "each call (the router's query cache is bypassed). Embedding configurations use the routed candidate policy.",
        f"Machine: {env['platform']}, {env['cpu_count']} CPUs, torch {env['torch']} on `{env['device']}` with {env['torch_threads']} threads.",
        "Class texts are embedded once at load (`index`); nothing here is the Rust runtime, see `bench_arrow` for that.",
        "",
        "| domain | concepts | configuration | weights | index (s) | median (ms) | p95 (ms) | decisions / s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d, e in r["domains"].items():
        for k, c in e["configs"].items():
            idx = f"{c['index_s']:.1f}" if "index_s" in c else "-"
            lines.append(
                f"| {d} | {e['concepts']} | {k} | {c['weights']} | {idx} | {c['median_ms']:.1f} | {c['p95_ms']:.1f} | {c['decisions_per_s']:.0f} |"
            )
    if r.get("skipped"):
        lines += ["", "Skipped: " + "; ".join(f"{k}: {v}" for k, v in r["skipped"].items())]
    lines += [
        "",
        "The embedding path adds one query encode per decision on top of routing and projection; the per-decision",
        "cost does not grow with the number of classes beyond the matrix product. Latency is machine-dependent and is",
        "not a published claim; run the script on the target hardware.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
