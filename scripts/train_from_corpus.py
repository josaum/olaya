"""Train, calibrate and evaluate the decision head on an LLM-distilled corpus (``olaya.corpus``).

Pipeline per domain: load ``data/corpora/<domain>.jsonl``, split group-disjoint by generation
batch (60/15/10/15), expand each partition into concept / subsumes / choice / score / noul tasks
with the existing ``real_ontology_train`` expansion, train the small BERT decision model on CPU
under a wall-clock budget, select on validation accuracy, fit a temperature on the calibration
partition, and evaluate once on test:

* task metrics (gold always among the candidates): accuracy, top-3, NLL, ECE before / after
  temperature, Brier, abstention curve, per-task metrics;
* full-ontology retrieval through the real engine (``OLaya.system_one`` with the saved
  checkpoint: lexical router top-k, then the model; gold not guaranteed in the candidates);
* the lexical baseline (``scripts/evaluate_lexical_baseline.evaluate_observations``) on the
  same test rows.

Writes ``models/checkpoints/olaya_<domain>.pt`` (gitignored), updates
``models/checkpoints/MANIFEST.json`` and ``benchmarks/results/decision-head-<domain>.{json,md}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OLAYA_TORCH_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn
from scripts.evaluate_lexical_baseline import evaluate_observations
from torch.utils.data import DataLoader

from olaya.calibration import compute_ece, fit_temperature, pad_and_cat_logits
from olaya.category_theory import CategoricalLoss
from olaya.corpus import (
    CorpusRow,
    corpus_manifest,
    read_jsonl,
    split_corpus,
    split_report,
    to_observations,
)
from olaya.engine import OLaya
from olaya.ontology import Ontology
from olaya.real_ontology_train import expand_observations
from olaya.tokenizer import load_local_tokenizer
from olaya.train import DecisionDataset, TrainingSample, collate_decision_batch, create_trainable_model

torch.set_num_threads(int(os.environ["OLAYA_TORCH_THREADS"]))

ONTOLOGIES = {
    "incident": ROOT / "src/olaya/data/incident_ontology.json",
    "gs1": ROOT / "src/olaya/data/gs1_ontology.json",
    "schemaorg": ROOT / "src/olaya/data/schemaorg_ontology.json",
}
CONCEPT_ROOT = {"incident": "INCIDENT:ROOT"}
INSTRUCTION = "Classify the entity into the {name} domain hierarchy."


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forward(model, batch, dev):
    return model(
        batch["input_ids"].to(dev),
        batch["attention_mask"].to(dev),
        batch["marker_pos"].to(dev),
        batch["marker_mask"].to(dev),
        batch["qtype"].to(dev),
        parent_adj=batch["parent_adj"].to(dev),
        child_adj=batch["child_adj"].to(dev),
        disjoint_adj=batch["disjoint_adj"].to(dev),
        return_marker_states=True,
    )


def _loader(samples, batch_size, pad_id, shuffle):
    return DataLoader(
        DecisionDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_decision_batch(b, pad_id=pad_id),
    )


def collect_logits(model, samples: list[TrainingSample], batch_size: int, pad_id: int, dev):
    outputs, targets, masks = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in _loader(samples, batch_size, pad_id, False):
            logits, _, _ = _forward(model, batch, dev)
            outputs.append(logits.cpu())
            masks.append(batch["marker_mask"].cpu())
            targets.append(batch["targets"].cpu())
    return pad_and_cat_logits(outputs), torch.cat(targets), pad_and_cat_logits(masks, pad_value=False)


def task_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, temperature: float) -> dict[str, Any]:
    probs = torch.softmax((logits / temperature).masked_fill(~mask, -torch.inf), dim=-1)
    m = compute_ece(probs, targets)
    one_hot = torch.nn.functional.one_hot(targets, num_classes=probs.shape[1])
    brier = float(((probs - one_hot) ** 2).sum(dim=1).mean())
    top3 = torch.topk(probs, k=min(3, probs.shape[1]), dim=-1).indices
    top3_acc = float((top3 == targets[:, None]).any(dim=1).float().mean())
    conf = probs.max(dim=-1).values
    correct = (probs.argmax(dim=-1) == targets).float()
    curve = []
    for thr in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        answered = conf >= thr
        n = int(answered.sum())
        curve.append(
            {
                "threshold": thr,
                "coverage": n / len(targets),
                "accuracy_when_answered": float(correct[answered].mean()) if n else None,
            }
        )
    return {
        "count": int(len(targets)),
        "accuracy": m.accuracy,
        "top3_accuracy": top3_acc,
        "nll": m.nll,
        "ece": m.ece,
        "mce": m.mce,
        "brier": brier,
        "avg_confidence": m.avg_confidence,
        "abstention_curve": curve,
    }


def retrieval_eval(engine: OLaya, rows: list[CorpusRow], domain: str, ontology: Ontology) -> dict[str, Any]:
    """Deployment path: router top-k, then the model; gold is not guaranteed to be a candidate."""
    root = CONCEPT_ROOT.get(domain)
    question = {"q": {"type": "concept", "root": root, "instructions": INSTRUCTION.format(name=ontology.name)}}
    top1 = recall5 = routed = scored = 0
    conf_correct: list[tuple[float, bool]] = []
    latencies = []
    for row in rows:
        result = engine.system_one({"domain": domain, "query": row.text}, question)
        answer = result.answers["q"]
        ranked = sorted(answer.candidate_probabilities.items(), key=lambda kv: -kv[1])
        cands = [c for c, _ in ranked]
        gold = row.target_curie
        # Gold may be an ancestor of a routed leaf: count as routed/correct only on exact match.
        routed += gold in cands or gold in answer.probabilities
        scored += gold in cands
        top1 += answer.curie == gold
        recall5 += gold in cands[:5]
        conf_correct.append((answer.confidence, answer.curie == gold))
        latencies.append(result.latency_ms)
    n = len(rows)
    latencies.sort()
    curve = []
    for thr in (0.0, 0.3, 0.5, 0.7, 0.9):
        answered = [c for conf, c in conf_correct if conf >= thr]
        curve.append(
            {
                "confidence_threshold": thr,
                "coverage": len(answered) / n,
                "accuracy_when_answered": sum(answered) / len(answered) if answered else None,
            }
        )
    return {
        "rows": n,
        "top1": top1 / n,
        "recall_at_5": recall5 / n,
        "gold_in_routed_subgraph": routed / n,
        "gold_among_scored_leaf_candidates": scored / n,
        "abstention_curve": curve,
        "latency_ms_median": latencies[(n - 1) // 2],
        "note": "OLaya.system_one over the full ontology; candidate_probabilities are over the routed leaves after axiom projection.",
    }


def train_domain(args: argparse.Namespace) -> dict[str, Any]:
    domain = args.domain
    ontology_path = ONTOLOGIES[domain]
    ontology = Ontology.from_json(ontology_path.read_text(encoding="utf-8"))
    corpus_path = args.corpus_dir / f"{domain}.jsonl"
    rows = read_jsonl(corpus_path)
    if not rows:
        raise SystemExit(f"empty corpus: {corpus_path}")
    splits = split_corpus(rows, seed=args.seed)
    manifest = corpus_manifest(ontology, ontology_path, rows, splits, None)
    print(f"{domain}: {len(rows)} rows; split report: {json.dumps(split_report(splits))}")

    tokenizer = load_local_tokenizer()
    special_ids = {
        "cls": tokenizer.cls_token_id,
        "sep": tokenizer.sep_token_id,
        "mask": tokenizer.mask_token_id,
        "pad": tokenizer.pad_token_id,
    }
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t_expand = time.perf_counter()
    samples = {
        name: expand_observations(
            ontology,
            to_observations(part),
            domain,
            lambda t: tokenizer.encode(t, add_special_tokens=False),
            special_ids,
            seed=args.seed,
        )
        for name, part in splits.items()
    }
    counts = {k: len(v) for k, v in samples.items()}
    print(f"expanded tasks: {counts} in {time.perf_counter() - t_expand:.1f}s")
    dev = torch.device("cpu")
    vocab_size = max(len(tokenizer), tokenizer.vocab_size) + 256
    model = create_trainable_model(vocab_size=vocab_size, hidden_size=256, backbone="custom_bert").to(dev)
    pad_id = special_ids["pad"]
    train_loader = _loader(samples["train"], args.batch_size, pad_id, True)
    val_loader = _loader(samples["validation"], args.batch_size, pad_id, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs * len(train_loader), eta_min=1e-5
    )
    criterion_dec = nn.CrossEntropyLoss()
    criterion_act = nn.NLLLoss()
    criterion_cat = CategoricalLoss(weight_mono=0.5, weight_disj=0.3, weight_cone=0.1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / f"olaya_{domain}.pt"
    config = {"vocab_size": vocab_size, "hidden_size": 256, "backbone": "custom_bert"}
    best_val = -1.0
    history = []
    t_start = time.perf_counter()
    budget = args.budget_minutes * 60
    epochs_run = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        tot_loss = tot_correct = tot_n = 0.0
        for batch in train_loader:
            targets = batch["targets"].to(dev)
            optimizer.zero_grad()
            logits, act_probs, marker_states = _forward(model, batch, dev)
            loss_cat, _ = criterion_cat(
                logits=logits,
                marker_states=marker_states,
                subsumption_mask=batch["subsumption_adj"].to(dev),
                disjoint_mask=batch["disjoint_adj"].to(dev),
                marker_mask=batch["marker_mask"].to(dev),
            )
            loss = (
                criterion_dec(logits, targets)
                + 0.2 * criterion_act(torch.log(act_probs + 1e-8), batch["act_targets"].to(dev))
                + loss_cat
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tot_loss += float(loss.detach()) * len(targets)
            tot_correct += int((logits.argmax(-1) == targets).sum())
            tot_n += len(targets)
            if time.perf_counter() - t_start > budget:
                break
        model.eval()
        v_correct = v_n = 0
        with torch.no_grad():
            for batch in val_loader:
                logits, _, _ = _forward(model, batch, dev)
                v_correct += int((logits.argmax(-1) == batch["targets"]).sum())
                v_n += len(batch["targets"])
        val_acc = v_correct / v_n
        epochs_run = epoch
        elapsed = time.perf_counter() - t_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": tot_loss / tot_n,
                "train_acc": tot_correct / tot_n,
                "val_acc": val_acc,
                "elapsed_s": round(elapsed, 1),
            }
        )
        print(
            f"epoch {epoch}: loss {tot_loss / tot_n:.3f} train acc {tot_correct / tot_n:.3f} val acc {val_acc:.3f} ({elapsed:.0f}s)"
        )
        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {"model_state_dict": model.state_dict(), "config": config, "val_acc": val_acc, "domain": domain},
                ckpt_path,
            )
        if elapsed > budget:
            print(f"time budget of {args.budget_minutes} min reached after epoch {epoch}")
            break

    # Reload the validation-selected checkpoint, then calibrate on the calibration partition only.
    checkpoint = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    cal_logits, cal_targets, cal_mask = collect_logits(model, samples["calibration"], args.batch_size, pad_id, dev)
    temperature, cal_before, cal_after = fit_temperature(cal_logits, cal_targets, candidate_mask=cal_mask)
    temperature = max(float(temperature), 1e-3)

    test_logits, test_targets, test_mask = collect_logits(model, samples["test"], args.batch_size, pad_id, dev)
    test_raw = task_metrics(test_logits, test_targets, test_mask, 1.0)
    test_cal = task_metrics(test_logits, test_targets, test_mask, temperature)
    per_task = {}
    for task in sorted({s.qtype_name for s in samples["test"]}):
        idx = [i for i, s in enumerate(samples["test"]) if s.qtype_name == task]
        per_task[task] = task_metrics(test_logits[idx], test_targets[idx], test_mask[idx], temperature)

    checkpoint.update(
        {
            "temperature": temperature,
            "corpus_sha256": manifest["corpus_sha256"],
            "dataset_manifest": {k: v for k, v in manifest.items() if k != "splits"},
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "commit": _git_commit(),
        }
    )
    torch.save(checkpoint, ckpt_path)
    ckpt_sha = _sha256(ckpt_path)
    train_s = time.perf_counter() - t_start

    # Deployment-path evaluation: strict load through the engine, full-ontology retrieval.
    t_eval = time.perf_counter()
    engine = OLaya(ontology=ontology, tokenizer=tokenizer, device="cpu", weights_path=ckpt_path)
    assert engine.model_id == f"checkpoint-sha256:{ckpt_sha[:12]}", engine.model_id
    assert engine.calibrated_temperature == temperature
    retrieval = retrieval_eval(engine, splits["test"], domain, ontology)
    lexical = evaluate_observations(ontology, to_observations(splits["test"]))
    lexical.pop("observations", None)
    lexical.pop("per_class", None)
    eval_s = time.perf_counter() - t_eval

    report = {
        "domain": domain,
        "seed": args.seed,
        "commit": _git_commit(),
        "corpus": {
            "path": _rel(corpus_path),
            "rows": len(rows),
            "classes": manifest["classes"],
            "corpus_sha256": manifest["corpus_sha256"],
            "ontology_file_sha256": manifest["ontology_file_sha256"],
            "generator_models": manifest["generator_models"],
            "labeller_models": manifest["labeller_models"],
            "label_provenance": manifest["label_provenance"],
        },
        "split": {
            "rows": {k: len(v) for k, v in splits.items()},
            "expanded_tasks": counts,
            "report": split_report(splits),
            "evidence": manifest["splits"],
        },
        "training": {
            "backbone": "custom_bert (4 layers, hidden 256, 4 heads)",
            "epochs_run": epochs_run,
            "max_epochs": args.max_epochs,
            "budget_minutes": args.budget_minutes,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "torch_threads": torch.get_num_threads(),
            "history": history,
            "selection_validation_accuracy": best_val,
            "runtime_s": round(train_s, 1),
        },
        "calibration": {
            "temperature": temperature,
            "fit_set_before": asdict(cal_before),
            "fit_set_after": asdict(cal_after),
            "fit_set_tasks": counts["calibration"],
        },
        "test_task_metrics": {
            "raw": test_raw,
            "calibrated": test_cal,
            "per_task_calibrated": per_task,
            "note": "gold is always among the candidates (5 hard negatives + gold for concept/choice); not retrieval",
        },
        "full_ontology_retrieval": {
            "model": retrieval,
            "lexical_baseline": lexical,
            "eval_runtime_s": round(eval_s, 1),
        },
        "checkpoint": {
            "path": _rel(ckpt_path),
            "sha256": ckpt_sha,
            "size_bytes": ckpt_path.stat().st_size,
            "model_id": engine.model_id,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "llm_cost_usd_estimate": 0.0,
        "limitations": [
            "Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.",
            "The split holds out generation batches (register x seed), not classes; some test classes may be unseen in training (see split.report).",
            "Task metrics have the gold candidate present; full_ontology_retrieval is the deployment path and the number to compare with the lexical baseline.",
        ],
    }
    out_json = args.results_dir / f"decision-head-{domain}.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.results_dir / f"decision-head-{domain}.md").write_text(render_markdown(report), encoding="utf-8")
    update_manifest(args.output_dir / "MANIFEST.json", domain, report)
    print(
        json.dumps(
            {
                "task_top1": test_cal["accuracy"],
                "ece_raw": test_raw["ece"],
                "ece_cal": test_cal["ece"],
                "retrieval_top1": retrieval["top1"],
                "lexical_top1": lexical["top1_accuracy_with_abstention"],
            }
        )
    )
    return report


def update_manifest(path: Path, domain: str, report: dict[str, Any]) -> None:
    data = (
        json.loads(path.read_text())
        if path.is_file()
        else {
            "note": "Checkpoints are not committed; rebuild with scripts/train_from_corpus.py. Entries record what the reported numbers came from.",
            "checkpoints": {},
        }
    )
    data["checkpoints"][domain] = {
        "file": Path(report["checkpoint"]["path"]).name,
        "sha256": report["checkpoint"]["sha256"],
        "size_bytes": report["checkpoint"]["size_bytes"],
        "corpus_sha256": report["corpus"]["corpus_sha256"],
        "ontology_file_sha256": report["corpus"]["ontology_file_sha256"],
        "commit": report["commit"],
        "temperature": report["calibration"]["temperature"],
        "metrics": {
            "validation_accuracy": report["training"]["selection_validation_accuracy"],
            "test_task_top1": report["test_task_metrics"]["calibrated"]["accuracy"],
            "test_task_ece": report["test_task_metrics"]["calibrated"]["ece"],
            "retrieval_top1": report["full_ontology_retrieval"]["model"]["top1"],
            "retrieval_recall_at_5": report["full_ontology_retrieval"]["model"]["recall_at_5"],
            "lexical_top1": report["full_ontology_retrieval"]["lexical_baseline"]["top1_accuracy_with_abstention"],
        },
        "report": f"benchmarks/results/decision-head-{domain}.json",
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def render_markdown(r: dict[str, Any]) -> str:
    t, raw, cal = r["test_task_metrics"], r["test_task_metrics"]["raw"], r["test_task_metrics"]["calibrated"]
    ret, lex = r["full_ontology_retrieval"]["model"], r["full_ontology_retrieval"]["lexical_baseline"]
    rows = r["split"]["rows"]
    lines = [
        f"# Decision head on the LLM-distilled {r['domain']} corpus",
        "",
        "Labels are LLM-distilled and filtered (generated for a class, label-substring filtered, kept only when an",
        "independent labeller call agreed). They are **not human labels**. The numbers below show whether the",
        "trained head recovers the class a language model wrote the text for; they do not establish real-world",
        "accuracy.",
        "",
        f"- Corpus: `{r['corpus']['path']}`, {r['corpus']['rows']} rows, {r['corpus']['classes']} classes, sha256 `{r['corpus']['corpus_sha256'][:16]}`",
        f"- Split by generation batch (not by class): train {rows['train']} / val {rows['validation']} / calib {rows['calibration']} / test {rows['test']} rows; expanded tasks {r['split']['expanded_tasks']}",
        f"- Test classes unseen in training: {len(r['split']['report']['test']['classes_unseen_in_train'])}",
        f"- Training: {r['training']['backbone']}, {r['training']['epochs_run']} epochs, {r['training']['runtime_s']} s, selected val acc {r['training']['selection_validation_accuracy']:.3f}",
        f"- Checkpoint sha256 `{r['checkpoint']['sha256'][:16]}` ({r['checkpoint']['size_bytes']} bytes), engine model id `{r['checkpoint']['model_id']}`",
        f"- Temperature {r['calibration']['temperature']:.3f} fitted on {r['calibration']['fit_set_tasks']} calibration tasks",
        "",
        "## Test task metrics (gold always among candidates)",
        "",
        "| | raw | temperature-scaled |",
        "|---|---|---|",
        f"| top-1 | {raw['accuracy']:.3f} | {cal['accuracy']:.3f} |",
        f"| top-3 | {raw['top3_accuracy']:.3f} | {cal['top3_accuracy']:.3f} |",
        f"| NLL | {raw['nll']:.3f} | {cal['nll']:.3f} |",
        f"| ECE | {raw['ece']:.3f} | {cal['ece']:.3f} |",
        f"| Brier | {raw['brier']:.3f} | {cal['brier']:.3f} |",
        "",
        "| task | n | top-1 | ECE |",
        "|---|---|---|---|",
    ]
    for task, m in t["per_task_calibrated"].items():
        lines.append(f"| {task} | {m['count']} | {m['accuracy']:.3f} | {m['ece']:.3f} |")
    lines += [
        "",
        "Abstention curve (temperature-scaled, all tasks):",
        "",
        "| threshold | coverage | accuracy when answered |",
        "|---|---|---|",
    ]
    for p in cal["abstention_curve"]:
        acc = "-" if p["accuracy_when_answered"] is None else f"{p['accuracy_when_answered']:.3f}"
        lines.append(f"| {p['threshold']} | {p['coverage']:.2f} | {acc} |")
    lines += [
        "",
        "## Full-ontology retrieval on the same test rows (deployment path)",
        "",
        "| system | top-1 | recall@5 |",
        "|---|---|---|",
        f"| lexical router (abstains on ties / no evidence) | {lex['top1_accuracy_with_abstention']:.3f} | {lex['recall_at_5']:.3f} |",
        f"| router top-k + trained model (`OLaya.system_one`) | {ret['top1']:.3f} | {ret['recall_at_5']:.3f} |",
        "",
        f"Gold was inside the routed subgraph for {ret['gold_in_routed_subgraph']:.3f} of test rows, which bounds the model's reachable top-1.",
    ]
    verdict = "beats" if ret["top1"] > lex["top1_accuracy_with_abstention"] else "does not beat"
    lines.append(
        f"The trained model **{verdict}** the lexical baseline on top-1 over the full ontology on this test set."
    )
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in r["limitations"]]
    return "\n".join(lines) + "\n"


def write_summary(results_dir: Path) -> None:
    """Cross-domain table from every ``decision-head-<domain>.json`` in ``results_dir``."""
    reports = [json.loads(p.read_text()) for p in sorted(results_dir.glob("decision-head-*.json"))]
    reports = [r for r in reports if "domain" in r]
    if not reports:
        return
    lines = [
        "# Decision head on LLM-distilled corpora: summary",
        "",
        "Labels are **LLM-distilled and filtered, not human**: one model wrote observations for a class without",
        "naming it, texts containing the label / CURIE were dropped, and a second model, shown only the text and a",
        "shuffled candidate list, had to pick the same class for the row to be kept. Agreement shows the text is",
        "recognisable as its generating class by a language model; it does not establish human-judged accuracy,",
        "real-world distribution, or performance on ambiguous / no-answer inputs. Splits hold out generation",
        "batches (register x seed), not classes. Task metrics have the gold candidate present; the retrieval",
        "columns are the deployment path (`OLaya.system_one`: lexical router top-k, then the model) over the",
        "full ontology, and the lexical baseline is scored on the same test rows. LLM cost is reported as 0",
        "because the prices of the models used are not pinned in this repository.",
        "",
        "| domain | rows (classes) | train/val/calib/test rows | lexical top-1 | model top-1 (retrieval) | retrieval recall@5 (lex / model) | task top-1 / top-3 | ECE raw -> scaled | runtime (train + eval) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        rows = r["split"]["rows"]
        ret, lex = r["full_ontology_retrieval"]["model"], r["full_ontology_retrieval"]["lexical_baseline"]
        raw, cal = r["test_task_metrics"]["raw"], r["test_task_metrics"]["calibrated"]
        lines.append(
            f"| {r['domain']} | {r['corpus']['rows']} ({r['corpus']['classes']}) | "
            f"{rows['train']}/{rows['validation']}/{rows['calibration']}/{rows['test']} | "
            f"{lex['top1_accuracy_with_abstention']:.3f} | {ret['top1']:.3f} | {lex['recall_at_5']:.3f} / {ret['recall_at_5']:.3f} | "
            f"{cal['accuracy']:.3f} / {cal['top3_accuracy']:.3f} | {raw['ece']:.3f} -> {cal['ece']:.3f} | "
            f"{r['training']['runtime_s']:.0f} s + {r['full_ontology_retrieval']['eval_runtime_s']:.0f} s |"
        )
    lines += ["", "Per-domain reports: " + ", ".join(f"`decision-head-{r['domain']}.md`" for r in reports) + ".", ""]
    lines += [
        "Checkpoints: "
        + ", ".join(f"{r['domain']} `{r['checkpoint']['sha256'][:12]}`" for r in reports)
        + " (gitignored; `models/checkpoints/MANIFEST.json` is committed).",
        "",
    ]
    (results_dir / "decision-head-summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=sorted(ONTOLOGIES), required=True)
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "data" / "corpora")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "checkpoints")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "benchmarks" / "results")
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--budget-minutes", type=float, default=25.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train_domain(args)
    write_summary(args.results_dir)


if __name__ == "__main__":
    main()
