"""Minimal smoke tests for csi-models build_model."""
from __future__ import annotations

import torch

from csi.models.model_factory import build_model


def test_build_model_returns_module() -> None:
    model, epochs = build_model("NTU-Fi_HAR", "lenet")
    assert epochs > 0
    x = torch.randn(2, 3, 114, 500)
    logits = model(x)
    assert logits.shape[0] == 2
