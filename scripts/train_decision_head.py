"""CLI runner for training O-Laya decision model and optionally exporting to ONNX/ORT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root and src to python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.export_onnx_ort import export_olaya

from olaya.train import train_olaya


def main():
    parser = argparse.ArgumentParser(description="Train O-Laya Ontology Decision Model")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num-samples", type=int, default=1200, help="Number of synthetic incidents to generate")
    parser.add_argument("--output", default="models/olaya_decision_model.pt", help="Path to save PyTorch checkpoint")
    parser.add_argument("--device", default=None, help="Device to use (mps, cuda, cpu)")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX and ORT after training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"=== Training O-Laya Decision Head ({args.num_samples} incidents, {args.epochs} epochs) ===")
    model, metrics = train_olaya(
        num_samples=args.num_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_path=args.output,
        seed=args.seed,
    )

    print("\n=== Training Completed ===")
    print(f"Best Validation Accuracy: {metrics['best_val_acc']:.2%}")
    print(f"Saved Checkpoint: {args.output}")

    if args.export_onnx:
        print("\n=== Exporting to ONNX and ORT FlatBuffers ===")
        export_olaya(output_dir="models", weights_path=args.output)


if __name__ == "__main__":
    main()
