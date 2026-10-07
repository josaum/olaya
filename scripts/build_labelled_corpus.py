"""Build the LLM-distilled labelled corpus for one domain (see ``olaya.corpus``).

Reads ``QWEN_API_KEY`` and ``QWEN_BASE_URL`` from the environment (never printed). Writes
``data/corpora/<domain>.jsonl`` plus ``<domain>.manifest.json`` (split sizes, hashes, build
statistics). Re-running with ``--resume`` keeps the already-built rows and only adds classes
that have no rows yet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from olaya.corpus import (
    OpenAICompatClient,
    build_corpus,
    corpus_manifest,
    read_jsonl,
    sample_classes,
    split_corpus,
    write_jsonl,
)
from olaya.ontology import Ontology

ONTOLOGIES = {
    "incident": ROOT / "src/olaya/data/incident_ontology.json",
    "gs1": ROOT / "src/olaya/data/gs1_ontology.json",
    "schemaorg": ROOT / "src/olaya/data/schemaorg_ontology.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=sorted(ONTOLOGIES), required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "corpora")
    parser.add_argument("--max-classes", type=int, default=None, help="Stratified class sample (Schema.org: 120)")
    parser.add_argument("--batches-per-class", type=int, default=3)
    parser.add_argument("--per-batch", type=int, default=3)
    parser.add_argument("--generator-model", default="qwen3.8-flash")
    parser.add_argument("--labeller-model", default="qwen3.8-max")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base_url = os.environ.get("QWEN_BASE_URL", "")
    api_key = os.environ.get("QWEN_API_KEY", "")
    if not base_url or not api_key:
        sys.exit("QWEN_BASE_URL and QWEN_API_KEY must be set in the environment")
    client = OpenAICompatClient(base_url, api_key, extra={"enable_thinking": False})

    ontology_path = ONTOLOGIES[args.domain]
    ontology = Ontology.from_json(ontology_path.read_text(encoding="utf-8"))
    classes = sample_classes(ontology, args.max_classes, seed=args.seed)
    out_path = args.out_dir / f"{args.domain}.jsonl"
    existing = read_jsonl(out_path) if args.resume and out_path.is_file() else []
    done = {r.target_curie for r in existing}
    todo = [c for c in classes if c not in done]
    print(f"{args.domain}: {len(classes)} classes, {len(todo)} to build, {len(existing)} existing rows")

    def chat(messages, model):
        return client.chat(messages, model, temperature=0.0 if model == args.labeller_model else 0.8)

    t0 = time.perf_counter()
    rows, stats = build_corpus(
        ontology,
        args.domain,
        chat,
        generator_model=args.generator_model,
        labeller_model=args.labeller_model,
        classes=todo,
        batches_per_class=args.batches_per_class,
        per_batch=args.per_batch,
        seed=args.seed,
        concurrency=args.concurrency,
        progress=print,
    )
    rows = sorted(existing + rows, key=lambda r: r.row_id)
    elapsed = time.perf_counter() - t0
    write_jsonl(rows, out_path)
    splits = split_corpus(rows, seed=args.seed)
    manifest = corpus_manifest(ontology, ontology_path, rows, splits, stats)
    manifest["build_runtime_s"] = round(elapsed, 1)
    manifest["llm_usage"] = {
        "calls": client.calls,
        "prompt_tokens": client.prompt_tokens,
        "completion_tokens": client.completion_tokens,
        "rate_limit_hits": client.rate_limit_hits,
        "cost_usd_estimate": 0.0,
        "cost_note": "prices for these models are not pinned in this repository; cost is reported as 0",
    }
    (args.out_dir / f"{args.domain}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"rows": len(rows), "stats": manifest["build_stats"], "splits": manifest["split_report"]}, indent=1)
    )
    if stats.stopped_early:
        print(f"STOPPED EARLY: {stats.stopped_early}")


if __name__ == "__main__":
    main()
