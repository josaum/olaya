"""Fine-tune the cross-encoder reranker on the OAEI Anatomy track (mouse / human).

Usage::

    .venv/bin/python scripts/train_reranker.py [--out models/reranker/anatomy-minilm-l6] [--time-budget 900]

Pipeline (all deterministic for a fixed ``--seed`` and ``--threads``):

1. parse ``tests/fixtures/oaei/anatomy/{mouse,human}.owl`` and ``reference.rdf``;
2. score mouse x human classes with the same CSLS-calibrated ``intfloat/e5-small-v2``
   cosine the Conference hybrid matcher uses and keep the top-``k`` per entity;
3. split the *source (mouse) entities* 80 / 20 (seed); positives = reference pairs,
   hard negatives = non-reference candidates, each restricted to its split;
4. fine-tune ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (one logit, BCE) on the training
   pairs under a hard CPU time budget, validating AUC / hits@1 on the held-out split;
5. save the checkpoint and ``manifest.json`` (data hashes, pair digests, hyper-parameters,
   seed, loss curve, validation) to ``--out``.

The checkpoint (~90 MB) is not committed; ``models/reranker/.gitignore`` keeps it out.
Re-run this script to reproduce it.  The Conference track is never read here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from olaya.benchmarks.reranker import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_BASE_REVISION,
    CrossEncoderReranker,
    TrainingConfig,
    anatomy_candidate_scores,
    build_pairs,
    candidate_pairs,
    entity_texts,
    load_anatomy,
    pair_texts,
    pairs_digest,
    split_sources,
    validation_report,
)
from olaya.embeddings import load_encoder  # noqa: E402

ANATOMY = REPO / "tests" / "fixtures" / "oaei" / "anatomy"
DEFAULT_OUT = REPO / "models" / "reranker" / "anatomy-minilm-l6"
CANDIDATE_ENCODER = "intfloat/e5-small-v2"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def hub_commit(model: str, revision: str) -> str | None:
    try:
        from huggingface_hub import model_info

        return model_info(model, revision=revision).sha
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anatomy-dir", type=Path, default=ANATOMY)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--revision", default=DEFAULT_BASE_REVISION)
    ap.add_argument("--candidate-encoder", default=CANDIDATE_ENCODER)
    ap.add_argument("--k", type=int, default=5, help="top-k candidates per entity for the hard negatives")
    ap.add_argument("--fraction", type=float, default=0.8, help="train fraction of the source entities")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=96)
    ap.add_argument("--time-budget", type=float, default=900.0, help="hard stop in seconds (default 15 min)")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)

    t_all = time.perf_counter()
    data = load_anatomy(args.anatomy_dir)
    print(
        f"anatomy: mouse {len(data.mouse.concepts)} classes, human {len(data.human.concepts)} classes, "
        f"{len(data.reference)} reference equivalences"
    )

    t0 = time.perf_counter()
    encoder = load_encoder(args.candidate_encoder, threads=args.threads)
    scores = anatomy_candidate_scores(encoder, data, args.k)
    cands = candidate_pairs(scores, args.k)
    ref_set = {(a.entity1, a.entity2) for a in data.reference}
    recall = sum(1 for p in ref_set if p in cands) / len(ref_set)
    print(
        f"candidates: {len(cands)} pairs (top-{args.k} per entity), reference recall {recall:.3f}, "
        f"{time.perf_counter() - t0:.1f}s"
    )

    all_sources = sorted({s for s, _ in scores} | {a.entity1 for a in data.reference})
    train_src, val_src = split_sources(all_sources, fraction=args.fraction, seed=args.seed)
    train_pairs = build_pairs(data.reference, scores, args.k, sources=train_src)
    val_pairs = build_pairs(data.reference, scores, args.k, sources=val_src)
    mt, ht = entity_texts(data.mouse), entity_texts(data.human)
    train_texts = pair_texts([(p.source, p.target) for p in train_pairs], mt, ht)
    val_texts = pair_texts([(p.source, p.target) for p in val_pairs], mt, ht)
    n_pos = sum(p.label for p in train_pairs)
    print(
        f"train: {len(train_pairs)} pairs ({n_pos} positives) from {len(train_src)} source entities; "
        f"validation: {len(val_pairs)} pairs ({sum(p.label for p in val_pairs)} positives) from {len(val_src)}"
    )

    reranker = CrossEncoderReranker(
        args.base_model, revision=args.revision, max_length=args.max_length, threads=args.threads, seed=args.seed
    )
    before = validation_report(val_pairs, reranker.score(val_texts).tolist())
    print(
        f"validation before training: base auc {before['base']['auc']} hits@1 {before['base']['hits@1']}; "
        f"cross-encoder auc {before['cross_encoder']['auc']} hits@1 {before['cross_encoder']['hits@1']}"
    )

    cfg = TrainingConfig(
        base_model=args.base_model,
        revision=args.revision,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        time_budget_s=args.time_budget,
        threads=args.threads,
    )
    stats = reranker.train(train_pairs, train_texts, cfg, validation=(val_pairs, val_texts))
    print(
        f"trained {stats.steps} steps ({stats.epochs_completed:.2f} epochs) in {stats.elapsed_s}s"
        f"{' - stopped by the time budget' if stats.stopped_by_budget else ''}; epoch losses {stats.epoch_losses}"
    )
    for rep in stats.validation:
        print(
            f"  epoch {rep['epoch']}: cross-encoder auc {rep['cross_encoder']['auc']} hits@1 "
            f"{rep['cross_encoder']['hits@1']}; fused w=0.5 auc {rep['fused']['w=0.50']['auc']} "
            f"hits@1 {rep['fused']['w=0.50']['hits@1']}"
        )

    manifest = {
        "date_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": git_commit(),
        "base_model": {
            "name": args.base_model,
            "revision": args.revision,
            "commit_sha": hub_commit(args.base_model, args.revision),
        },
        "candidate_encoder": {"name": encoder.name, "commit_sha": encoder.commit_sha, "csls_k": 10, "k": args.k},
        "data": {
            "track": "OAEI Anatomy (mouse -> human), see tests/fixtures/oaei/anatomy/README.md",
            "files": data.hashes(),
            "reference_equivalences": len(data.reference),
            "candidate_pairs": len(cands),
            "candidate_reference_recall": round(recall, 4),
            "split": {
                "by": "source entity",
                "fraction": args.fraction,
                "seed": args.seed,
                "train_sources": len(train_src),
                "validation_sources": len(val_src),
            },
            "train_pairs": len(train_pairs),
            "train_positives": n_pos,
            "validation_pairs": len(val_pairs),
            "validation_positives": sum(p.label for p in val_pairs),
            "train_pairs_sha256": pairs_digest(train_pairs),
            "validation_pairs_sha256": pairs_digest(val_pairs),
        },
        "hyperparameters": {k: v for k, v in cfg.__dict__.items()},
        "training": stats.to_dict(),
        "validation_before_training": before,
        "total_seconds": round(time.perf_counter() - t_all, 1),
    }
    out = reranker.save(args.out, manifest)
    print(f"saved checkpoint + manifest to {out} ({time.perf_counter() - t_all:.0f}s total)")
    print(json.dumps({"validation": stats.validation[-1] if stats.validation else before}, indent=1)[:800])
    return 0


if __name__ == "__main__":
    sys.exit(main())
