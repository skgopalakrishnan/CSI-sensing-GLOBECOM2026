"""Factories for constructing autoencoder models from experiment configs."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple, Any

import torch
from torch import nn

from .models import TinyFStrideAutoencoder, TinyFStrideConfig

__all__ = [
    "build_autoencoder_from_config",
]

def _as_int_pair(values, *, length: int = 2, name: str = "value") -> Tuple[int, int]:
    seq = list(values) if isinstance(values, Sequence) else [values]
    if len(seq) != length:
        raise ValueError(f"{name} must provide {length} entries, got {len(seq)}")
    result = tuple(int(v) for v in seq)
    assert len(result) == 2, f"Expected exactly 2 elements, got {len(result)}"
    return result  # type: ignore[return-value]


def _as_kernel_sequence(values, *, name: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    seq = list(values)
    if len(seq) != 2:
        raise ValueError(f"{name} must contain exactly two kernel specifications")
    first = _as_int_pair(seq[0], name=f"{name}[0]")
    second = _as_int_pair(seq[1], name=f"{name}[1]")
    return first, second


def _as_dilation_sequence(values, *, name: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    seq = list(values)
    if len(seq) != 2:
        raise ValueError(f"{name} must contain exactly two dilation sequences")
    first = tuple(int(v) for v in seq[0])
    second = tuple(int(v) for v in seq[1])
    return (first, second)


def build_autoencoder_from_config(
    model_cfg: Mapping[str, Any],
    *,
    sample_shape: Sequence[int],
    device: Optional[torch.device | str] = None,
) -> nn.Module:
    """Instantiate an autoencoder described by ``model_cfg``."""

    if not sample_shape:
        raise ValueError("sample_shape must provide at least the channel dimension")
    channels = int(sample_shape[0])
    model_type = str(model_cfg.get("type", "tiny_f_stride")).lower()

    if model_type in {"tiny", "tiny_f_stride", "tiny-f-stride"}:
        hidden_dims = _as_int_pair(model_cfg.get("hidden_dims", (16, 32)), name="hidden_dims")
        encoder_kernels = _as_kernel_sequence(model_cfg.get("encoder_kernels", ((5, 3), (3, 3))), name="encoder_kernels")
        encoder_strides = _as_kernel_sequence(model_cfg.get("encoder_strides", ((2, 1), (2, 1))), name="encoder_strides")
        temporal_dilations = _as_dilation_sequence(
            model_cfg.get("temporal_dilations", ((1, 2), (2, 4))),
            name="temporal_dilations",
        )
        latent_channels = model_cfg.get("latent_channels")
        latent_freq = model_cfg.get("latent_freq")
        config = TinyFStrideConfig(
            in_channels=channels,
            hidden_dims=hidden_dims,
            encoder_kernels=encoder_kernels,
            encoder_strides=encoder_strides,
            temporal_kernel=int(model_cfg.get("temporal_kernel", 3)),
            temporal_dilations=temporal_dilations,
            dropout=float(model_cfg.get("dropout", 0.10)),
            output_activation=model_cfg.get("output_activation"),
            latent_channels=int(latent_channels) if latent_channels is not None else 6,
            latent_freq=int(latent_freq) if latent_freq is not None else 8,
        )
        model: nn.Module = TinyFStrideAutoencoder(config)
    else:
        raise ValueError(f"Unsupported autoencoder model type '{model_type}'")

    if device is not None:
        model = model.to(device)
    return model
