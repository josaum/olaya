"""Hermetic tokenizer loading from the bundled ModernBERT ``tokenizer.json``.

Shared by the training pipelines and the test-suite so neither needs network access or a
``tests``-package import.
"""

from __future__ import annotations

from pathlib import Path

from transformers import PreTrainedTokenizerFast

TOKENIZER_FILE = Path(__file__).resolve().parents[2] / "models" / "tokenizer.json"


def load_local_tokenizer(tokenizer_file: str | Path = TOKENIZER_FILE) -> PreTrainedTokenizerFast:
    """Load the bundled fast tokenizer with the special tokens the sequence builder expects."""
    tokenizer_file = Path(tokenizer_file)
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"Bundled tokenizer not found at {tokenizer_file}")
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
