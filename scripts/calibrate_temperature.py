"""Standalone temperature calibration and Expected Calibration Error (ECE) inspection CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root and src to python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from olaya.calibration import CalibrationMetrics, compute_ece, fit_temperature, pad_and_cat_logits
from olaya.engine import OLaya
from olaya.ontology import Ontology, create_incident_ontology
from olaya.real_ontology_train import generate_real_ontology_samples, load_ontology_file
from olaya.train import DecisionDataset, collate_decision_batch

DOMAIN_ONTOLOGIES = {
    "atlas": ROOT / "data" / "atlas_full_ontology.json",
    "doid": ROOT / "data" / "doid_ontology.json",
    "schemaorg": ROOT / "data" / "schemaorg_ontology.json",
    "gs1": ROOT / "data" / "gs1_ontology.json",
}



def print_reliability_diagram(metrics: CalibrationMetrics, title: str = "Reliability Diagram", num_bins: int = 10):
    print(f"\n{'='*70}")
    print(f" {title.upper()} ")
    print(f"{'='*70}")
    print(f"Overall Metrics: {metrics.summary()}")
    print("-" * 70)
    print(f"{'Bin Range':<15} | {'Count':<7} | {'Accuracy':<10} | {'Confidence':<10} | {'Gap':<8} | {'Diagram'}")
    print("-" * 70)

    bin_width = 1.0 / num_bins
    for i in range(num_bins):
        low = i * bin_width
        high = (i + 1) * bin_width
        cnt = metrics.bin_counts[i]
        acc = metrics.bin_accuracies[i]
        conf = metrics.bin_confidences[i]
        gap = abs(acc - conf)

        bar_len = int(acc * 20)
        conf_marker = int(conf * 20)
        bar = [" "] * 21
        for b in range(min(bar_len, 21)):
            bar[b] = "█"
        if conf_marker < 21:
            bar[conf_marker] = "|" if bar[conf_marker] == " " else "◈"
        bar_str = "".join(bar)

        print(
            f"[{low:.2f} - {high:.2f}]   | {cnt:<7} | {acc:<9.2%} | {conf:<9.2%} | {gap:<7.4f} | [{bar_str}]"
        )
    print("-" * 70)
    print("Legend: [███] Accuracy, [◈] Calibration Target (Confidence), [|] Empty Conf Target\n")


def calibrate_checkpoint(
    checkpoint_path: Path,
    domain: str = "atlas",
    samples: int = 800,
    bins: int = 10,
    save: bool = True,
    device: str | None = None,
):
    print(f"Loading checkpoint: {checkpoint_path}...")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    cfg = ckpt.get("config", {})
    backbone = cfg.get("backbone", "custom_bert")

    # Load ontology
    if domain == "incident":
        ontology = create_incident_ontology()
    else:
        onto_file = DOMAIN_ONTOLOGIES.get(domain, DOMAIN_ONTOLOGIES["atlas"])
        ontology = load_ontology_file(onto_file)

    engine = OLaya(
        ontology=ontology,
        weights_path=checkpoint_path,
        device=device,
    )

    print(f"Engine initialized with {len(ontology.concepts)} concepts. Backbone: {backbone}")

    # Generate calibration evaluation samples
    print(f"Generating {samples} validation samples for {domain}...")
    tokenizer = engine.tokenizer
    special_ids = engine.special_ids
    val_samples = generate_real_ontology_samples(
        ontology=ontology,
        domain=domain,
        num_samples=samples,
        tokenizer_encode=lambda t: tokenizer.encode(t, add_special_tokens=False),
        special_ids=special_ids,
        use_real_abox=True,
        seed=1337,
    )

    loader = DataLoader(
        DecisionDataset(val_samples),
        batch_size=16,
        shuffle=False,
        collate_fn=lambda b: collate_decision_batch(b, pad_id=special_ids["pad"]),
    )

    engine.model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(engine.device)
            attention_mask = batch["attention_mask"].to(engine.device)
            marker_pos = batch["marker_pos"].to(engine.device)
            marker_mask = batch["marker_mask"].to(engine.device)
            qtypes = batch["qtype"].to(engine.device)
            targets = batch["targets"].to(engine.device)

            logits, _ = engine.model(input_ids, attention_mask, marker_pos, marker_mask, qtypes)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    cat_logits = pad_and_cat_logits(all_logits)
    cat_targets = torch.cat(all_targets, dim=0)

    opt_temp, uncal_metrics, cal_metrics = fit_temperature(cat_logits, cat_targets)

    print_reliability_diagram(uncal_metrics, title=f"Before Calibration ({domain})", num_bins=bins)
    print_reliability_diagram(cal_metrics, title=f"After Temperature Scaling (T={opt_temp:.4f})", num_bins=bins)

    improvement = (uncal_metrics.ece - cal_metrics.ece) / (uncal_metrics.ece + 1e-12)
    print(f"Calibration Summary for {domain.upper()}:")
    print(f"  Optimal Temperature (T): {opt_temp:.4f}")
    print(f"  Uncalibrated ECE:        {uncal_metrics.ece:.4f} ({uncal_metrics.ece:.2%})")
    print(f"  Calibrated ECE:          {cal_metrics.ece:.4f} ({cal_metrics.ece:.2%})")
    print(f"  ECE Error Reduction:     {improvement:.2%}")

    if save:
        ckpt["temperature"] = opt_temp
        ckpt["ece"] = cal_metrics.ece
        ckpt["uncalibrated_ece"] = uncal_metrics.ece
        torch.save(ckpt, str(checkpoint_path))
        print(f"  -> Successfully updated checkpoint with calibrated temperature: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate temperature and evaluate ECE on O-Laya checkpoints")
    parser.add_argument("--checkpoint", default="models/olaya_atlas.pt", help="Path to .pt checkpoint")
    parser.add_argument("--domain", choices=["atlas", "doid", "schemaorg", "incident"], default="atlas")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--no-save", action="store_true", help="Do not overwrite checkpoint with calibrated temperature")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    calibrate_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        domain=args.domain,
        samples=args.samples,
        bins=args.bins,
        save=not args.no_save,
        device=args.device,
    )


if __name__ == "__main__":
    main()
