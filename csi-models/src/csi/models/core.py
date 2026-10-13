"""Technique-aware facade for model training/evaluation APIs."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .autoencoders import training as _autoencoder_training
from .supervised import training as _supervised_training

try:
    from .archive import self_supervised as _ssl_training
except Exception:  # pragma: no cover - optional namespace surface
    _ssl_training = None

__all__ = [
    "train",
    "test",
    "evaluate_classifier",
    "train_autoencoder",
    "train_autoencoder_epoch",
    "evaluate_autoencoder",
    "train_simclr",
    "train_wimae",
    "save_ssl_checkpoint",
    "load_ssl_checkpoint",
    "train_with_technique",
    "evaluate_with_technique",
]


train = _supervised_training.train
test = _supervised_training.test
evaluate_classifier = _supervised_training.evaluate_classifier
train_autoencoder = _autoencoder_training.train_autoencoder
train_autoencoder_epoch = _autoencoder_training.train_autoencoder_epoch
evaluate_autoencoder = _autoencoder_training.evaluate_autoencoder


def _missing_ssl(*args: Any, **kwargs: Any):
    raise ImportError("Self-supervised training is not available in this build of csi.models.")


if _ssl_training is not None:
    train_simclr = _ssl_training.train_simclr
    train_wimae = _ssl_training.train_wimae
    save_ssl_checkpoint = _ssl_training.save_ssl_checkpoint
    load_ssl_checkpoint = _ssl_training.load_ssl_checkpoint
else:  # pragma: no cover - optional fallback
    train_simclr = _missing_ssl
    train_wimae = _missing_ssl
    save_ssl_checkpoint = _missing_ssl
    load_ssl_checkpoint = _missing_ssl


def _normalize_technique(technique: str) -> str:
    return str(technique).strip().lower().replace("-", "_")


def train_with_technique(technique: str, *args: Any, **kwargs: Any):
    """Dispatch to the technique-specific training implementation."""

    key = _normalize_technique(technique)
    if key in {"supervised", "sl", "classification"}:
        return train(*args, **kwargs)
    if key in {"autoencoder", "ae", "reconstruction"}:
        return train_autoencoder(*args, **kwargs)
    if key == "simclr":
        return train_simclr(*args, **kwargs)
    if key in {"wimae", "mae"}:
        return train_wimae(*args, **kwargs)
    raise ValueError(
        "Unsupported training technique "
        f"'{technique}'. Expected one of: supervised, autoencoder, simclr, wimae."
    )


def evaluate_with_technique(technique: str, *args: Any, **kwargs: Any):
    """Dispatch to the technique-specific evaluation implementation."""

    key = _normalize_technique(technique)
    if key in {"supervised", "sl", "classification"}:
        return evaluate_classifier(*args, **kwargs)
    if key in {"autoencoder", "ae", "reconstruction"}:
        return evaluate_autoencoder(*args, **kwargs)
    raise ValueError(
        "Unsupported evaluation technique "
        f"'{technique}'. Expected one of: supervised, autoencoder."
    )


def train_dispatch(
    *,
    technique: str,
    model: torch.nn.Module,
    tensor_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    criterion=None,
    **kwargs: Any,
):
    """Internal helper for callers that want a common keyword-based dispatch path."""

    key = _normalize_technique(technique)
    if key in {"supervised", "sl", "classification"}:
        if criterion is None:
            raise ValueError("criterion is required for supervised training")
        return train(
            model,
            tensor_loader,
            num_epochs,
            learning_rate,
            criterion,
            device,
            **kwargs,
        )
    if key in {"autoencoder", "ae", "reconstruction"}:
        return train_autoencoder(
            model,
            tensor_loader,
            num_epochs,
            learning_rate,
            device,
            **kwargs,
        )
    raise ValueError(f"Unsupported training technique '{technique}'.")
