"""CLI script to train O-Laya decision model on Atlas, DOID, and Schema.org ontologies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root and src to python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from scripts.export_onnx_ort import export_olaya
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from olaya.calibration import fit_temperature, pad_and_cat_logits
from olaya.real_ontology_train import generate_real_ontology_samples, load_ontology_file
from olaya.train import (
    DecisionDataset,
    collate_decision_batch,
    create_trainable_model,
)

DATA_PATHS = {
    "atlas": ROOT / "data" / "atlas_full_ontology.json",
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
    export_onnx: bool = True,
    seed: int = 42,
) -> dict[str, float]:
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
            from tests.conftest import load_local_tokenizer

            tokenizer = load_local_tokenizer()

    special_ids = {
        "cls": getattr(tokenizer, "cls_token_id", 50281) or 50281,
        "sep": getattr(tokenizer, "sep_token_id", 50282) or 50282,
        "mask": getattr(tokenizer, "mask_token_id", 50284) or 50284,
        "pad": getattr(tokenizer, "pad_token_id", 50283) or 0,
    }

    print(f"Generating {num_samples} domain-grounded training samples for {domain}...")
    samples = generate_real_ontology_samples(
        ontology=ontology,
        domain=domain,
        num_samples=num_samples,
        tokenizer_encode=lambda t: tokenizer.encode(t, add_special_tokens=False),
        special_ids=special_ids,
        use_real_abox=use_real_abox,
        seed=seed,
    )

    split = int(0.85 * len(samples))
    train_samples, val_samples = samples[:split], samples[split:]
    print(f"Dataset assembled: {len(train_samples)} training, {len(val_samples)} validation samples")

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

    best_val_acc = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_modernbert" if "ModernBERT" in backbone else ""
    ckpt_path = output_dir / f"olaya_{domain}{suffix}.pt"
    best_val_logits: list[torch.Tensor] = []
    best_val_targets: list[torch.Tensor] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
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

            optimizer.zero_grad()
            logits, act_probs = model(input_ids, attention_mask, marker_pos, marker_mask, qtypes)

            loss_dec = criterion_dec(logits, targets)
            loss_act = criterion_act(torch.log(act_probs + 1e-8), act_targets)
            loss = loss_dec + 0.2 * loss_act

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item()) * len(targets)
            preds = torch.argmax(logits, dim=-1)
            correct_dec += int((preds == targets).sum().item())
            total_items += len(targets)

        train_loss = total_loss / total_items
        train_acc = correct_dec / total_items

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_items = 0
        epoch_val_logits: list[torch.Tensor] = []
        epoch_val_targets: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(dev)
                attention_mask = batch["attention_mask"].to(dev)
                marker_pos = batch["marker_pos"].to(dev)
                marker_mask = batch["marker_mask"].to(dev)
                qtypes = batch["qtype"].to(dev)
                targets = batch["targets"].to(dev)
                act_targets = batch["act_targets"].to(dev)

                logits, act_probs = model(input_ids, attention_mask, marker_pos, marker_mask, qtypes)
                loss_dec = criterion_dec(logits, targets)
                loss_act = criterion_act(torch.log(act_probs + 1e-8), act_targets)
                loss = loss_dec + 0.2 * loss_act

                val_loss += float(loss.item()) * len(targets)
                preds = torch.argmax(logits, dim=-1)
                val_correct += int((preds == targets).sum().item())
                val_items += len(targets)

                epoch_val_logits.append(logits.cpu())
                epoch_val_targets.append(targets.cpu())

        val_loss /= val_items
        val_acc = val_correct / val_items

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_logits = epoch_val_logits
            best_val_targets = epoch_val_targets

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

    # Calibration & ECE computation on best validation outputs
    cal_results = {"temperature": 1.0, "ece": 0.0, "uncalibrated_ece": 0.0}
    if calibrate and best_val_logits:
        all_logits = pad_and_cat_logits(best_val_logits)
        all_targets = torch.cat(best_val_targets, dim=0)
        opt_temp, uncal_metrics, cal_metrics = fit_temperature(all_logits, all_targets)
        print(f"\n--- Model Calibration for {domain.upper()} ---")
        print(f"  Raw (uncalibrated):  {uncal_metrics.summary()}")
        print(f"  Fitted Temperature: {opt_temp:.4f}")
        print(f"  Calibrated:          {cal_metrics.summary()}")

        cal_results["temperature"] = opt_temp
        cal_results["ece"] = cal_metrics.ece
        cal_results["uncalibrated_ece"] = uncal_metrics.ece

        # Re-save best checkpoint with calibration metadata
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            ckpt["temperature"] = opt_temp
            ckpt["ece"] = cal_metrics.ece
            ckpt["uncalibrated_ece"] = uncal_metrics.ece
            torch.save(ckpt, str(ckpt_path))

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
        "temperature": cal_results["temperature"],
        "ece": cal_results["ece"],
    }


def main():
    parser = argparse.ArgumentParser(description="Train O-Laya on real ontologies: Atlas, DOID, Schema.org, GS1")
    parser.add_argument("--domain", choices=["atlas", "doid", "schemaorg", "gs1", "all"], default="all")
    parser.add_argument("--backbone", default="custom_bert", help="Backbone model name or 'custom_bert'")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-export-onnx", action="store_true")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip temperature calibration")
    parser.add_argument("--no-real-abox", action="store_true", help="Skip real ABox evidence ingestion")
    args = parser.parse_args()

    domains = ["atlas", "doid", "schemaorg", "gs1"] if args.domain == "all" else [args.domain]

    results = {}

    for d in domains:
        path = DATA_PATHS[d]
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
            export_onnx=not args.no_export_onnx,
        )
        results[d] = res

    print("\n=======================================================")
    print("ALL REAL ONTOLOGY TRAINING RUNS COMPLETED")
    print("=======================================================")
    for d, res in results.items():
        print(
            f"Domain: {d:<12} | Validation Accuracy: {res['val_acc']:.2%} | Calibrated ECE: {res.get('ece', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
