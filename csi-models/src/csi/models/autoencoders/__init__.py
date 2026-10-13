"""Autoencoder architectures and utilities tailored for CSI reconstruction."""

# from .freq_temporal import FreqTemporalConvAutoencoder, FreqTemporalAutoencoderConfig
from .classifier import (
    AutoencoderClassifier,
    TemporalGRUHead,
    TemporalSSMHead,
    TemporalTCNHead,
    build_autoencoder_classifier,
    HEAD_ALIASES,
    build_classifier_head,
)
from .models import TinyFStrideAutoencoder, TinyFStrideConfig
from .factory import build_autoencoder_from_config
from .training import evaluate_autoencoder, train_autoencoder, train_autoencoder_epoch

__all__ = [
    "TinyFStrideAutoencoder",
    "TinyFStrideConfig",
    "AutoencoderClassifier",
    "TemporalTCNHead",
    "TemporalGRUHead",
    "TemporalSSMHead",
    "build_classifier_head",
    "build_autoencoder_classifier",
    "HEAD_ALIASES",
    "build_autoencoder_from_config",
    "train_autoencoder",
    "train_autoencoder_epoch",
    "evaluate_autoencoder",
]
