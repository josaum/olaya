"""Fine-tune one bi-encoder jointly on every LLM-distilled corpus, with mined hard negatives.

Protocol (same as ``scripts/evaluate_routers.py``, which this script imports):

* one training set = the union of the three domains' *train* splits (batch-disjoint by default,
  ``--split class`` for the class-disjoint protocol); class texts are the union of the three
  ontologies' path-context texts (CURIE namespaces are checked to be disjoint);
* loss = InfoNCE over in-batch gold classes + one sibling hard negative per query + the encoder's
  own top-``m`` non-gold classes for that query, re-mined at the start of every epoch;
* model selection = the *mean* over domains of the validation-split full-ontology top-1 of the
  cosine router; the best epoch is kept;
* reporting = ``DomainContext.evaluate_encoder`` per domain on the test split, once, written to
  ``benchmarks/results/router-eval-joint[-classdisjoint].{json,md}`` and compared with the lexical
  baseline, the zero-shot base encoder and the per-domain fine-tuned checkpoints of
  ``router-eval-<domain>.json`` when present.

Labels are LLM-distilled, not human (see ``decision-head-summary.md``).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OLAYA_TORCH_THREADS", "4")

import torch
from scripts.evaluate_routers import (
    ENCODERS,
    THREADS,
    DomainContext,
    _git_commit,
    _rel,
    evaluate_encoder_safely,
    ranking_metrics,
    say,
)
from scripts.train_from_corpus import ONTOLOGIES

from olaya.embedding_router import (
    EmbeddingRouter,
    finetune_bi_encoder,
    load_finetuned_encoder,
    sibling_curies,
)
from olaya.embeddings import TransformerEncoder


def _f(x: Any) -> str:
    return "-" if x is None else f"{x:.3f}"


def build_contexts(args: argparse.Namespace) -> dict[str, DomainContext]:
    contexts = {d: DomainContext(args, d) for d in sorted(ONTOLOGIES)}
    seen: dict[str, str] = {}
    for d, ctx in contexts.items():
        for curie in ctx.class_texts:
            if curie in seen and seen[curie] != d:
                raise SystemExit(f"CURIE {curie} exists in both {seen[curie]} and {d}; joint class texts would collide")
            seen[curie] = d
    return contexts


def joint_validate(contexts: dict[str, DomainContext]):
    def validate(encoder) -> float:
        scores = []
        for ctx in contexts.values():
            router = EmbeddingRouter(ctx.ontology, encoder, class_texts=ctx.class_texts)
            rankings = [router.rank(r.text, ctx.root) for r in ctx.splits["validation"]]
            scores.append(ranking_metrics(ctx.ontology, rankings, ctx.golds("validation"))["top1"])
        return sum(scores) / len(scores)

    return validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", choices=list(ENCODERS), default="bge-small")
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "data" / "corpora")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "benchmarks" / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("batch", "class"), default="batch")
    parser.add_argument("--candidate-policy", nargs="+", choices=("leaves", "routed"), default=["leaves", "routed"])
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--mine", type=int, default=2, help="mined hard negatives per query per epoch (0 disables)")
    parser.add_argument("--tag", default="joint", help="checkpoint / report name suffix")
    args = parser.parse_args()
    torch.set_num_threads(THREADS)
    t_start = time.perf_counter()

    contexts = build_contexts(args)
    model_name, prefix = ENCODERS[args.base]
    encoder = TransformerEncoder(
        model_name, prefix=prefix, threads=THREADS, batch_size=32, max_length=128, use_cache=False
    )
    texts: dict[str, str] = {}
    pairs: list[tuple[str, str]] = []
    for ctx in contexts.values():
        texts.update(ctx.class_texts)
        pairs += [(r.text, r.target_curie) for r in ctx.splits["train"]]
    owner = {curie: d for d, ctx in contexts.items() for curie in ctx.class_texts}

    def hard_negatives(gold: str) -> list[str]:
        return sibling_curies(contexts[owner[gold]].ontology, gold)

    validate = joint_validate(contexts)
    zero_shot = validate(encoder)
    say(
        f"joint fine-tune from {model_name}: {len(pairs)} train pairs over {len(texts)} classes; zero-shot mean validation top-1 {zero_shot:.3f}"
    )
    suffix = "" if args.split == "batch" else "-classdisjoint"
    checkpoint = args.output_dir / f"olaya_embed_{args.tag}{suffix}.pt"
    result = finetune_bi_encoder(
        encoder,
        None,
        pairs,
        checkpoint=checkpoint,
        validate=validate,
        budget_s=args.minutes * 60,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        class_texts=texts,
        hard_negatives=hard_negatives,
        mine_hard_negatives=args.mine,
        progress=say,
    )
    tuned = load_finetuned_encoder(checkpoint, threads=THREADS, batch_size=32)

    report: dict[str, Any] = {
        "tag": args.tag,
        "split_unit": args.split,
        "candidate_policies": list(args.candidate_policy),
        "commit": _git_commit(),
        "seed": args.seed,
        "finetune": {
            "base_model": model_name,
            "base_key": args.base,
            "checkpoint": _rel(checkpoint),
            "sha256": result.sha256,
            "size_bytes": checkpoint.stat().st_size,
            "train_pairs": len(pairs),
            "classes": len(texts),
            "epochs_run": result.epochs_run,
            "best_epoch": result.best_epoch,
            "best_mean_validation_top1": result.best_validation,
            "zero_shot_mean_validation_top1": zero_shot,
            "budget_minutes": args.minutes,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "mined_hard_negatives_per_query": args.mine,
            "runtime_s": round(result.runtime_s, 1),
            "history": result.history,
            "loss": "InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query"
            + (f" + {args.mine} mined hard negatives per query re-mined each epoch" if args.mine else ""),
        },
        "domains": {},
    }
    for d, ctx in contexts.items():
        say(f"{d}: evaluating joint encoder")
        entry = evaluate_encoder_safely(ctx, tuned)
        per_domain_path = args.results_dir / f"router-eval-{d}{suffix}.json"
        reference: dict[str, Any] = {"lexical": ctx.lexical_report()}
        if per_domain_path.is_file():
            ref = json.loads(per_domain_path.read_text())
            reference["zero_shot_base"] = ref["encoders"].get(args.base, {})
            reference["per_domain_fine_tuned"] = ref.get("fine_tuned", {})
            reference["report"] = _rel(per_domain_path)
        report["domains"][d] = {
            "split_rows": {k: len(v) for k, v in ctx.splits.items()},
            "test_classes_unseen_in_train": len(
                {r.target_curie for r in ctx.splits["test"]} - {r.target_curie for r in ctx.splits["train"]}
            ),
            "joint": entry,
            "reference": reference,
        }
    report["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    report["runtime_s"] = round(time.perf_counter() - t_start, 1)
    out = args.results_dir / f"router-eval-{args.tag}{suffix}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.results_dir / f"router-eval-{args.tag}{suffix}.md").write_text(render(report), encoding="utf-8")
    say(f"wrote {out} ({report['runtime_s']} s)")


def _test(e: dict[str, Any], key: str) -> Any:
    try:
        return e["variants"]["cosine"]["test"][key]
    except (KeyError, TypeError):
        return None


def _engine(e: dict[str, Any], name: str, key: str = "top1") -> Any:
    try:
        return e["engine"][name][key]
    except (KeyError, TypeError):
        return None


def render(r: dict[str, Any]) -> str:
    ft = r["finetune"]
    unit = r["split_unit"]
    lines = [
        f"# Joint bi-encoder over the three LLM-distilled corpora ({'batch' if unit == 'batch' else 'class'}-disjoint split)",
        "",
        "Labels are LLM-distilled and filtered, **not human**. One encoder is fine-tuned on the union of the three",
        "domains' train splits and selected on the mean validation top-1 over domains; each domain's test split is",
        "scored once. Engine rows use `OLaya.system_one` over the full ontology (router top-10); `[routed]` scores every",
        "routed concept instead of the routed leaves only.",
        "",
        f"- Base `{ft['base_model']}`, {ft['train_pairs']} train pairs over {ft['classes']} classes, {ft['epochs_run']} epochs in {ft['runtime_s']} s (budget {ft['budget_minutes']} min), best epoch {ft['best_epoch']}",
        f"- {ft['loss']}",
        f"- Mean validation top-1: fine-tuned {ft['best_mean_validation_top1']:.3f} vs zero-shot base {ft['zero_shot_mean_validation_top1']:.3f}",
        f"- Checkpoint `{ft['checkpoint']}` sha256 `{ft['sha256'][:16]}` ({ft['size_bytes']} bytes; gitignored)",
        f"- Commit `{r['commit'][:12]}`, runtime {r['runtime_s']} s",
        "",
        "| domain | test rows | unseen test classes | system | ceiling@8 | top-1 | recall@5 | task top-1 | ECE | engine top-1 | engine top-1 [routed] | engine recall@5 [routed] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d, dom in r["domains"].items():
        rows = dom["split_rows"]["test"]
        unseen = dom["test_classes_unseen_in_train"]
        ref = dom["reference"]
        lex = ref["lexical"]["test"]
        lines.append(
            f"| {d} | {rows} | {unseen} | lexical | {lex['ceiling']['8']:.3f} | {lex['top1']:.3f} | {lex['recall_at_5']:.3f} | "
            f"{ref['lexical']['task']['calibrated']['accuracy']:.3f} | {ref['lexical']['task']['calibrated']['ece']:.3f} | - | - | - |"
        )
        systems = [
            (f"zero-shot {ft['base_key']}", ref.get("zero_shot_base") or {}),
            ("per-domain fine-tuned", ref.get("per_domain_fine_tuned") or {}),
            ("**joint fine-tuned**", dom["joint"]),
        ]
        for name, e in systems:
            if "variants" not in e:
                continue
            task = e.get("task", {}).get("calibrated", {})
            lines.append(
                f"| {d} | {rows} | {unseen} | {name} | {_f(_test(e, 'ceiling')['8'] if _test(e, 'ceiling') else None)} | {_f(_test(e, 'top1'))} | "
                f"{_f(_test(e, 'recall_at_5'))} | {_f(task.get('accuracy'))} | {_f(task.get('ece'))} | {_f(_engine(e, 'cosine'))} | "
                f"{_f(_engine(e, 'cosine [routed]'))} | {_f(_engine(e, 'cosine [routed]', 'recall_at_5'))} |"
            )
    lines += [
        "",
        "## History",
        "",
        "- "
        + "; ".join(f"epoch {h['epoch']} loss {h['train_loss']:.3f} val {h['validation']:.3f}" for h in ft["history"]),
        "",
        "## Limitations",
        "",
        "- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.",
        (
            "- The split holds out generation batches, not classes; the encoder has seen almost every test class in training."
            if unit == "batch"
            else "- The split holds out whole classes per domain; no test class has a training row."
        ),
        "- The per-domain fine-tuned reference rows come from `router-eval-<domain>.json` and were trained with the same budget but without mined negatives.",
        "- Nothing here changes the shipped Rust / Node runtimes, which remain lexical.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
