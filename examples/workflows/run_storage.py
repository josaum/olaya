"""Run the local storage policy with an explicitly untrained Python engine."""

from __future__ import annotations

# ruff: noqa: E402 - bootstrap the local src tree before importing O-Laya.
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
from transformers import PreTrainedTokenizerFast

from olaya import OLaya, Ontology
from olaya.workflow import compile_workflow, run_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shipment", default="Frozen seafood requires cold storage.")
    parser.add_argument("--workflow", type=Path, default=Path(__file__).with_name("storage.json"))
    parser.add_argument("--tbox", type=Path, default=Path(__file__).with_name("storage-tbox.json"))
    args = parser.parse_args()
    document = json.loads(args.workflow.read_text(encoding="utf-8"))
    ontology = Ontology.from_json(args.tbox.read_text(encoding="utf-8"))
    compiled = compile_workflow(document, ontology)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(ROOT / "models/tokenizer.json"),
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    torch.manual_seed(42)
    engine = OLaya(ontology=ontology, tokenizer=tokenizer, device="cpu", weights_path=False)
    result = run_workflow(compiled, engine, {"shipment": args.shipment})
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
