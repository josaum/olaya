"""CLI script to train O-Laya decision model on Atlas, DOID, and Schema.org ontologies."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

# Add repo root and src to python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from scripts.export_onnx_ort import export_olaya
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from olaya.calibration import compute_ece, fit_temperature, pad_and_cat_logits
from olaya.category_theory import CategoricalLoss
from olaya.evaluation import dataset_manifest, split_observations
from olaya.real_ontology_train import expand_observations, generate_real_ontology_observations, load_ontology_file
from olaya.train import (
    DecisionDataset,
    TrainingSample,
    collate_decision_batch,
    create_trainable_model,
)

DATA_PATHS = {
    "atlas": ROOT / "data" / "custom_ontology.json",
    "doid": ROOT / "data" / "doid_ontology.json",
    "schemaorg": ROOT / "data" / "schemaorg_ontology.json",
    "gs1": ROOT / "data" / "gs1_ontology.json",
}


def train_single_domain(
    domain: str,
    ontology_path: Path,
    tokenizer: PreTrainedTokenizerFast | None = None,
    backbone: str = "custom_bert",
    use_real_abox: bool = True,
    calibrate: bool = True,
    num_samples: int = 1500,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-4,
    device: str | None = None,
    output_dir: Path = ROOT / "models",
    export_onnx: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    print("\n=======================================================")
    print(f"Training O-Laya on Domain: {domain.upper()} ({ontology_path})")
    print(f"Backbone: {backbone} | Real ABox: {use_real_abox} | Calibrate: {calibrate}")
    print("=======================================================")

    ontology = load_ontology_file(ontology_path)
    print(f"Loaded {ontology.name} with {len(ontology.concepts)} concepts.")

    # Determine device
    if device is None:
        if torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    # Tokenizer setup
    if tokenizer is None:
        if backbone != "custom_bert":
            tokenizer = AutoTokenizer.from_pretrained(backbone)
        else:
            from olaya.tokenizer import load_local_tokenizer

            tokenizer = load_local_tokenizer()

    special_ids = {
        "cls": getattr(tokenizer, "cls_token_id", 50281) or 50281,
        "sep": getattr(tokenizer, "sep_token_id", 50282) or 50282,
        "mask": getattr(tokenizer, "mask_token_id", 50284) or 50284,
        "pad": getattr(tokenizer, "pad_token_id", 50283) or 0,
    }

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    torch.manual_seed(seed)
    np.random.seed(seed)
    observations = generate_real_ontology_observations(
        ontology=ontology,
        domain=domain,
        num_samples=num_samples,
        use_real_abox=use_real_abox,
        seed=seed,
    )
    source_splits = split_observations(observations, seed=seed)
    manifest = dataset_manifest(ontology, source_splits)
    split_samples = {
        name: expand_observations(
            ontology=ontology,
            observations=records,
            domain=domain,
            tokenizer_encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
            special_ids=special_ids,
            seed=seed,
        )
        for name, records in source_splits.items()
    }
    if any(not samples for samples in split_samples.values()):
        raise ValueError("Every source split must expand into at least one task")
    train_samples = split_samples["train"]
    val_samples = split_samples["validation"]
    print("Disjoint grouped splits:", {name: len(samples) for name, samples in split_samples.items()})

    train_loader = DataLoader(
        DecisionDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_decision_batch(b, pad_id=special_ids["pad"]),
    )
    val_loader = DataLoader(
        DecisionDataset(val_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_decision_batch(b, pad_id=special_ids["pad"]),
    )

    vocab_size = max(len(tokenizer), getattr(tokenizer, "vocab_size", 30522)) + 256
    model = create_trainable_model(vocab_size=vocab_size, hidden_size=256, backbone=backbone)
    model.to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)
    criterion_dec = nn.CrossEntropyLoss()
    criterion_act = nn.NLLLoss()
    criterion_cat = CategoricalLoss(weight_mono=0.5, weight_disj=0.3, weight_cone=0.1)

    best_val_acc = -1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_modernbert" if "ModernBERT" in backbone else ""
    ckpt_path = output_dir / f"olaya_{domain}{suffix}.pt"

    print(f"Starting domain training on {dev} for {epochs} epochs with Category-Theoretic Loss...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_cat_loss = 0.0
        correct_dec = 0
        total_items = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            marker_pos = batch["marker_pos"].to(dev)
            marker_mask = batch["marker_mask"].to(dev)
            qtypes = batch["qtype"].to(dev)
            targets = batch["targets"].to(dev)
            act_targets = batch["act_targets"].to(dev)
            subsumption_adj = batch["subsumption_adj"].to(dev)
            parent_adj = batch["parent_adj"].to(dev)
            child_adj = batch["child_adj"].to(dev)
            disjoint_adj = batch["disjoint_adj"].to(dev)

            optimizer.zero_grad()
            logits, act_probs, marker_states = model(
                input_ids,
                attention_mask,
                marker_pos,
                marker_mask,
                qtypes,
                parent_adj=parent_adj,
                child_adj=child_adj,
                disjoint_adj=disjoint_adj,
                return_marker_states=True,
            )

            loss_dec = criterion_dec(logits, targets)
            loss_act = criterion_act(torch.log(act_probs + 1e-8), act_targets)
            loss_cat, _ = criterion_cat(
                logits=logits,
                marker_states=marker_states,
                subsumption_mask=subsumption_adj,
                disjoint_mask=disjoint_adj,
                marker_mask=marker_mask,
            )
            loss = loss_dec + 0.2 * loss_act + loss_cat

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item()) * len(targets)
            total_cat_loss += float(loss_cat.item()) * len(targets)
            preds = torch.argmax(logits, dim=-1)
            correct_dec += int((preds == targets).sum().item())
            total_items += len(targets)

        train_loss = total_loss / total_items
        train_cat_loss = total_cat_loss / total_items
        train_acc = correct_dec / total_items

        # Validation
        model.eval()
        val_loss = 0.0
        val_cat_loss = 0.0
        val_mono_viols: list[float] = []
        val_disj_colls: list[float] = []
        val_correct = 0
        val_items = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(dev)
                attention_mask = batch["attention_mask"].to(dev)
                marker_pos = batch["marker_pos"].to(dev)
                marker_mask = batch["marker_mask"].to(dev)
                qtypes = batch["qtype"].to(dev)
                targets = batch["targets"].to(dev)
                act_targets = batch["act_targets"].to(dev)
                subsumption_adj = batch["subsumption_adj"].to(dev)
                parent_adj = batch["parent_adj"].to(dev)
                child_adj = batch["child_adj"].to(dev)
                disjoint_adj = batch["disjoint_adj"].to(dev)

                logits, act_probs, marker_states = model(
                    input_ids,
                    attention_mask,
                    marker_pos,
                    marker_mask,
                    qtypes,
                    parent_adj=parent_adj,
                    child_adj=child_adj,
                    disjoint_adj=disjoint_adj,
                    return_marker_states=True,
                )
                loss_dec = criterion_dec(logits, targets)
                loss_act = criterion_act(torch.log(act_probs + 1e-8), act_targets)
                loss_cat, cat_metrics = criterion_cat(
                    logits=logits,
                    marker_states=marker_states,
                    subsumption_mask=subsumption_adj,
                    disjoint_mask=disjoint_adj,
                    marker_mask=marker_mask,
                )
                loss = loss_dec + 0.2 * loss_act + loss_cat

                val_loss += float(loss.item()) * len(targets)
                val_cat_loss += float(loss_cat.item()) * len(targets)
                val_mono_viols.append(cat_metrics["mono_violation_rate"])
                val_disj_colls.append(cat_metrics["disj_collision_rate"])
                preds = torch.argmax(logits, dim=-1)
                val_correct += int((preds == targets).sum().item())
                val_items += len(targets)

        val_loss /= val_items
        val_cat_loss /= val_items
        val_acc = val_correct / val_items
        avg_mono_viol = float(np.mean(val_mono_viols)) if val_mono_viols else 0.0
        avg_disj_coll = float(np.mean(val_disj_colls)) if val_disj_colls else 0.0

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} (Cat: {train_cat_loss:.4f}) Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | Mono Viol: {avg_mono_viol:.1%} Disj Coll: {avg_disj_coll:.1%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": vocab_size,
                    "hidden_size": model.hidden_size,
                    "backbone": backbone,
                },
                "val_acc": val_acc,
                "domain": domain,
            }
            torch.save(checkpoint, str(ckpt_path))
            print(f"  -> Saved best checkpoint (acc: {val_acc:.2%}) to {ckpt_path}")

    # Reload the checkpoint chosen ONLY on validation before touching calibration.
    checkpoint = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def collect_logits(samples: list[TrainingSample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loader = DataLoader(
            DecisionDataset(samples),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda items: collate_decision_batch(items, pad_id=special_ids["pad"]),
        )
        outputs, targets, masks = [], [], []
        with torch.no_grad():
            for batch in loader:
                logits, _ = model(
                    batch["input_ids"].to(dev),
                    batch["attention_mask"].to(dev),
                    batch["marker_pos"].to(dev),
                    batch["marker_mask"].to(dev),
                    batch["qtype"].to(dev),
                    parent_adj=batch["parent_adj"].to(dev),
                    child_adj=batch["child_adj"].to(dev),
                    disjoint_adj=batch["disjoint_adj"].to(dev),
                )
                outputs.append(logits.cpu())
                masks.append(batch["marker_mask"].cpu())
                targets.append(batch["targets"].cpu())
        return pad_and_cat_logits(outputs), torch.cat(targets), pad_and_cat_logits(masks, pad_value=False)

    temperature = 1.0
    calibration_fit = None
    if calibrate:
        calibration_logits, calibration_targets, calibration_mask = collect_logits(split_samples["calibration"])
        temperature, before, after = fit_temperature(
            calibration_logits, calibration_targets, candidate_mask=calibration_mask
        )
        temperature = max(temperature, 1e-3)
        calibration_fit = {"before": asdict(before), "after": asdict(after), "scope": "calibration_fit_set"}

    # Final test is evaluated once after checkpoint selection and calibration.
    test_logits, test_targets, test_mask = collect_logits(split_samples["test"])
    test_metrics = compute_ece(
        torch.softmax((test_logits / temperature).masked_fill(~test_mask, -torch.inf), dim=-1), test_targets
    )
    test_raw_metrics = compute_ece(torch.softmax(test_logits.masked_fill(~test_mask, -torch.inf), dim=-1), test_targets)
    probabilities = torch.softmax((test_logits / temperature).masked_fill(~test_mask, -torch.inf), dim=-1)
    test_one_hot = torch.nn.functional.one_hot(test_targets, num_classes=probabilities.shape[1])
    test_brier = float(((probabilities - test_one_hot) ** 2).sum(dim=1).mean())
    per_task_metrics = {}
    for task in sorted({sample.qtype_name for sample in split_samples["test"]}):
        indices = [i for i, sample in enumerate(split_samples["test"]) if sample.qtype_name == task]
        per_task_metrics[task] = {
            "count": len(indices),
            **asdict(compute_ece(probabilities[indices], test_targets[indices])),
        }
    manifest["expanded_task_counts"] = {name: len(samples) for name, samples in split_samples.items()}
    report = {
        "seed": seed,
        "domain": domain,
        "backbone": backbone,
        "training_parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "requested_sample_budget": num_samples,
            "calibrate": calibrate,
            "use_real_abox": use_real_abox,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": version("transformers"),
            "device": str(dev),
            "torch_cpu_threads": torch.get_num_threads(),
        },
        "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "special_token_ids": special_ids,
        "training_source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                ROOT / "src/olaya/evaluation.py",
                ROOT / "src/olaya/real_ontology_train.py",
                ROOT / "src/olaya/calibration.py",
                ROOT / "src/olaya/train.py",
                ROOT / "src/olaya/model.py",
                ROOT / "src/olaya/sequence.py",
            )
        },
        "exported_artifacts_evaluated": False,
        "ontology_file_sha256": hashlib.sha256(ontology_path.read_bytes()).hexdigest(),
        "dataset_manifest": manifest,
        "selection_validation_accuracy": best_val_acc,
        "temperature": temperature,
        "calibration_fit": calibration_fit,
        "test_raw_metrics": asdict(test_raw_metrics),
        "test_metrics": asdict(test_metrics),
        "test_brier_score": test_brier,
        "test_metrics_by_task": per_task_metrics,
        "limitations": "Template-derived supervision; not independently labeled real-world accuracy. "
        "Calibration metrics concern raw model logits, not ontology-projected deployed outputs. "
        "Concept/choice samples contain oracle-included gold candidates, not end-to-end retrieval.",
    }
    report_path = output_dir / f"olaya_{domain}{suffix}_evaluation.json"
    checkpoint.update({"temperature": temperature, "evaluation_report": report_path.name, "dataset_manifest": manifest})
    torch.save(checkpoint, str(ckpt_path))
    report["checkpoint_sha256"] = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Held-out test:", test_metrics.summary())

    print(f"\nCompleted {domain} training! Best Validation Accuracy: {best_val_acc:.2%}")

    if export_onnx:
        print(f"Exporting ONNX and ORT model for domain {domain}...")
        domain_models_dir = output_dir / f"{domain}{suffix}"
        domain_models_dir.mkdir(parents=True, exist_ok=True)
        export_olaya(
            output_dir=str(domain_models_dir),
            backbone_name=backbone if backbone != "custom_bert" else "answerdotai/ModernBERT-base",
            weights_path=str(ckpt_path),
        )

    return {
        "val_acc": best_val_acc,
        "domain": domain,
        "temperature": temperature,
        "ece": test_metrics.ece,
        "test_acc": test_metrics.accuracy,
        "evaluation_report": str(report_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train O-Laya on real ontologies: Atlas, DOID, Schema.org, GS1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ontology-path", type=Path, help="Override ontology file for one explicit domain")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--domain", choices=["atlas", "doid", "schemaorg", "gs1", "all"], default="all")
    parser.add_argument("--backbone", default="custom_bert", help="Backbone model name or 'custom_bert'")
    parser.add_argument("--samples", "--num-samples", dest="samples", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--export-onnx", action="store_true", help="Opt in to legacy export; tokenizer/calibration parity is unverified"
    )
    parser.add_argument("--no-export-onnx", action="store_true", help="Explicitly disable legacy export")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip temperature calibration")
    parser.add_argument("--no-real-abox", action="store_true", help="Skip real ABox evidence ingestion")
    args = parser.parse_args()

    if args.ontology_path and args.domain == "all":
        parser.error("--ontology-path requires a single --domain")
    domains = ["atlas", "doid", "schemaorg", "gs1"] if args.domain == "all" else [args.domain]

    results = {}

    for d in domains:
        path = args.ontology_path or DATA_PATHS[d]
        res = train_single_domain(
            domain=d,
            ontology_path=path,
            backbone=args.backbone,
            use_real_abox=not args.no_real_abox,
            calibrate=not args.no_calibrate,
            num_samples=args.samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            export_onnx=args.export_onnx and not args.no_export_onnx,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        results[d] = res

    print("\n=======================================================")
    print("ALL REAL ONTOLOGY TRAINING RUNS COMPLETED")
    print("=======================================================")
    for d, res in results.items():
        print(
            f"Domain: {d:<12} | Validation Accuracy: {res['val_acc']:.2%} | Held-out Test Acc: {res['test_acc']:.2%} | Test ECE: {res.get('ece', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
