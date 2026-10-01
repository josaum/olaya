"""Shared fixtures: a hermetic tokenizer loaded from the bundled ModernBERT tokenizer.json."""

from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from olaya.engine import OLaya

TOKENIZER_FILE = Path(__file__).resolve().parent.parent / "models" / "tokenizer.json"


def load_local_tokenizer() -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER_FILE),
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerFast:
    return load_local_tokenizer()


@pytest.fixture(scope="session")
def olaya(tokenizer) -> OLaya:
    torch.manual_seed(0)
    return OLaya(tokenizer=tokenizer, device="cpu")
