"""Shared fixtures: a hermetic tokenizer loaded from the bundled ModernBERT tokenizer.json."""

import pytest
import torch

from olaya.engine import OLaya
from olaya.tokenizer import TOKENIZER_FILE, load_local_tokenizer

__all__ = ["TOKENIZER_FILE", "load_local_tokenizer"]


@pytest.fixture(scope="session")
def tokenizer():
    return load_local_tokenizer()


@pytest.fixture(scope="session")
def olaya(tokenizer) -> OLaya:
    torch.manual_seed(0)
    return OLaya(tokenizer=tokenizer, device="cpu")
