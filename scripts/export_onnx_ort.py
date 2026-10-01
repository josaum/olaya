"""Export O-Laya Decision Model to ONNX and ORT (FlatBuffers) formats."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from olaya.model import OntologyDecisionModel


def export_olaya(
    output_dir: str = "models",
    backbone_name: str = "answerdotai/ModernBERT-base",
    opset_version: int = 17,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    onnx_file = out_path / "olaya_decision_model.onnx"
    ort_file = out_path / "olaya_decision_model.ort"

    print(f"Loading backbone {backbone_name} and building OntologyDecisionModel...")
    model = OntologyDecisionModel(backbone_name_or_config=backbone_name)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    tokenizer_file = out_path / "tokenizer.json"
    tokenizer.backend_tokenizer.save(str(tokenizer_file))
    print(f"Saved fast tokenizer to {tokenizer_file}")

    # Dummy inputs: Batch=1, SeqLen=32, Candidates=4
    batch_size = 1
    seq_len = 32
    num_candidates = 4

    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    dummy_marker_pos = torch.tensor([[4, 8, 12, 16]], dtype=torch.long)
    dummy_marker_mask = torch.ones((batch_size, num_candidates), dtype=torch.bool)
    dummy_qtype = torch.zeros((batch_size,), dtype=torch.long)

    input_names = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
    output_names = ["logits", "act_probs"]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "marker_pos": {0: "batch_size", 1: "num_candidates"},
        "marker_mask": {0: "batch_size", 1: "num_candidates"},
        "qtype": {0: "batch_size"},
        "logits": {0: "batch_size", 1: "num_candidates"},
        "act_probs": {0: "batch_size"},
    }

    print(f"Exporting PyTorch model to ONNX: {onnx_file}...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_marker_pos, dummy_marker_mask, dummy_qtype),
        str(onnx_file),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Successfully generated ONNX model: {onnx_file} ({onnx_file.stat().st_size / 1e6:.1f} MB)")

    print(f"Converting ONNX to ORT FlatBuffers format in {out_path}...")
    conv_cmd = [
        sys.executable,
        "-m",
        "onnxruntime.tools.convert_onnx_models_to_ort",
        "--output_dir",
        str(out_path),
        str(onnx_file),
    ]
    subprocess.check_call(conv_cmd)
    if ort_file.exists():
        print(f"Successfully created ORT FlatBuffers model: {ort_file} ({ort_file.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"Conversion command ran. Files generated: {[p.name for p in out_path.glob('*')]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--backbone", default="answerdotai/ModernBERT-base", help="Backbone model name")
    args = parser.parse_args()
    export_olaya(output_dir=args.output_dir, backbone_name=args.backbone)
