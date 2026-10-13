"""Utility namespace for CSI data loaders."""

from __future__ import annotations

from .sensefi.common import DataBundle

__all__ = [
    "get_experiment_loaders",
    "DataBundle",
    "filter_none_collate",
    "load_csi_bench_supervised",
    "load_contrastive_benchmark_ssl",
    "load_contrastive_benchmark_ssl_loaders",
    "load_classification_benchmark_ssl",
    "load_classification_with_val_benchmark_ssl",
    "load_augmented_train_benchmark_ssl",
    "load_mae_benchmark_ssl",
]


def load_csi_bench_supervised(*args, **kwargs):
    from .csi_bench import load_benchmark_supervised

    return load_benchmark_supervised(*args, **kwargs)


def load_contrastive_benchmark_ssl(*args, **kwargs):
    from .csi_bench import load_contrastive_benchmark_ssl as _impl

    return _impl(*args, **kwargs)


def load_contrastive_benchmark_ssl_loaders(*args, **kwargs):
    from .csi_bench import load_contrastive_benchmark_ssl_loaders as _impl

    return _impl(*args, **kwargs)


def load_classification_benchmark_ssl(*args, **kwargs):
    from .csi_bench import load_classification_benchmark_ssl as _impl

    return _impl(*args, **kwargs)


def load_classification_with_val_benchmark_ssl(*args, **kwargs):
    from .csi_bench import load_classification_with_val_benchmark_ssl as _impl

    return _impl(*args, **kwargs)


def load_augmented_train_benchmark_ssl(*args, **kwargs):
    from .csi_bench import load_augmented_train_benchmark_ssl as _impl

    return _impl(*args, **kwargs)


def load_mae_benchmark_ssl(*args, **kwargs):
    from .csi_bench import load_mae_benchmark_ssl as _impl

    return _impl(*args, **kwargs)


def filter_none_collate(*args, **kwargs):
    from .common import filter_none_collate as _impl

    return _impl(*args, **kwargs)


def get_experiment_loaders(*args, **kwargs):
    from .sensefi.loaders import get_experiment_loaders as _impl

    return _impl(*args, **kwargs)
