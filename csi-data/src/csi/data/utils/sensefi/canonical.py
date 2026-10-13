"""Canonical CSI data loaders with unified shape, time resolution, and normalisation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import numpy as np
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from ...datasets.ntu_fi import NTU_Dataset
from ...datasets.transform import TransformDataset as TD
from .common import (
    _compute_dataset_stats,
    _compute_per_subcarrier_stats,
    _hashable_mapping,
    _resolve_dataset_root,
    _resolve_normalisation,
)


__all__ = [
    "get_canonical_dataset",
    "get_canonical_stats",
]


DEFAULT_NTU_SMOOTHING = {"method": "savgol", "window": 25, "polyorder": 3}
# DEFAULT_NTU_CLEANING = {"method": "moving_average", "window": 5}


@dataclass
class _CanonicalStats:
    pre_transform: Callable[[Tensor], Tensor]
    normaliser: Callable[[Tensor], Tensor]
    mean: Optional[Tensor]
    std: Optional[Tensor]


_STATS_CACHE: Dict[Tuple[str, str, int, str, str, Tuple[Tuple[str, Any], ...], str], _CanonicalStats] = {}


_CACHE_DIR_NAME = ".canonical_stats"
_CACHE_VERSION = 1


def _load_dataset_for_stats(dataset_key: str, root: Path, base_norm: str, modal: str, smoothing: Optional[Dict[str, Any]]):
    if dataset_key in _NTU_DATASET_KEYS:
        smoothing_cfg = smoothing if smoothing is not None else DEFAULT_NTU_SMOOTHING
        return _load_ntu_dataset(root, "train", base_norm, modal, smoothing_cfg)
    raise ValueError(f"Unsupported dataset '{dataset_key}' for canonical stats")


def _verify_normalisation(
    dataset: Dataset,
    pre: Callable[[Tensor], Tensor],
    normaliser: Callable[[Tensor], Tensor],
    max_samples: int = 32,
) -> Optional[Tuple[float, float, int]]:
    import math

    total = 0
    sum_ = 0.0
    sum_sq = 0.0
    samples = min(max_samples, len(dataset))  # type: ignore[arg-type]
    for idx in range(samples):
        sample, _ = dataset[idx]
        tensor = torch.as_tensor(sample).float()
        tensor = pre(tensor) if pre is not None else tensor
        tensor = normaliser(tensor)
        sum_ += tensor.sum().item()
        sum_sq += (tensor.double() ** 2).sum().item()
        total += tensor.numel()
    if total == 0:
        return None
    mean = sum_ / total
    var = max(sum_sq / total - mean ** 2, 0.0)
    std = math.sqrt(var)
    return mean, std, samples


def _log_canonical_stats(
    dataset_key: str,
    root_key: str,
    timesteps: int,
    canon_norm: str,
    stats: _CanonicalStats,
    verification: Optional[Tuple[float, float, int]] = None,
) -> None:
    mean, std = stats.mean, stats.std
    tag = f"[CANON-STATS] dataset={dataset_key} root={root_key} timesteps={timesteps} norm={canon_norm}:"
    if mean is None or std is None:
        print(f"{tag} mean/std not available.")
        return
    if canon_norm == "per_subcarrier":
        mean_arr = mean.detach().cpu().view(-1).numpy()
        std_arr = std.detach().cpu().view(-1).numpy()
        
        # Compute mean of means and stds for logging
        mean_mean = np.mean(mean_arr)
        mean_std = np.std(mean_arr)        
        std_mean = np.mean(std_arr)
        std_std = np.std(std_arr)
        # print(f"{tag} Per-subcarrier means={mean_mean:.6f} (std={mean_std:.6f})")
        # print(f"{tag} Per-subcarrier stds={std_mean:.6f} (std={std_std:.6f})")
        if verification is not None:
            norm_mean, norm_std, sample_count = verification
            # print(f"{tag} Normalised sample mean={norm_mean:.6f}, std={norm_std:.6f} (samples={sample_count})")
        
        # # Full arrays
        # mean_str = np.array2string(mean_arr, precision=4, suppress_small=True, separator=", ")
        # std_str = np.array2string(std_arr, precision=4, suppress_small=True, separator=", ")
        # print(f"{tag} per-subcarrier mean={mean_str}")
        # print(f"{tag} per-subcarrier std={std_str}")
    # else:
        # print(f"{tag} mean={float(mean.detach().cpu()):.6f} std={float(std.detach().cpu()):.6f}")


def _stats_cache_path(root: Path, cache_key: Tuple[str, str, int, str, str, Tuple[Tuple[str, Any], ...], str]) -> Path:
    dataset_key, _, timesteps, canon_norm, base_norm, smoothing_key, modal = cache_key
    digest_source = "|".join(
        (
            dataset_key,
            modal,
            str(timesteps),
            canon_norm,
            base_norm,
            repr(smoothing_key),
        )
    )
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:16]
    safe_dataset = dataset_key.replace(os.sep, "-").replace(" ", "_")
    safe_modal = modal.replace(os.sep, "-").replace(" ", "_")
    filename = f"{safe_dataset}_{safe_modal}_ts{timesteps}_{canon_norm}_{base_norm}_{digest}_v{_CACHE_VERSION}.pt"
    cache_dir = root / _CACHE_DIR_NAME
    return cache_dir / filename


def _load_cached_stats(path: Path) -> Optional[Tuple[Tensor, Tensor]]:
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"Warning: Failed to load cached canonical stats from {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version", -1)) != _CACHE_VERSION:
        return None
    mean = payload.get("mean")
    std = payload.get("std")
    if not isinstance(mean, Tensor) or not isinstance(std, Tensor):
        return None
    return mean.float(), std.float()


def _save_cached_stats(path: Path, mean: Tensor, std: Tensor, metadata: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _CACHE_VERSION,
            "mean": mean.detach().cpu().float(),
            "std": std.detach().cpu().float(),
            "metadata": metadata,
        }
        torch.save(payload, path)
    except Exception as exc:
        print(f"Warning: Failed to cache canonical stats at {path}: {exc}")


_NTU_DATASET_KEYS = {
    "ntu_fi",
    "ntu-fi",
    "ntu_fi_har",
    "ntu-fi_har",
    "ntu_fi_humanid",
    "ntu-fi_humanid",
    "ntu_fi-humanid",
    "ntu-fi-humanid",
    "ntu_fi_human_id",
    "ntu-fi_human_id",
}
def _load_ntu_dataset(
    root: Path,
    split: str,
    base_normalisation: str,
    modal: str,
    smoothing: Optional[Dict[str, Any]] = None,
) -> NTU_Dataset:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split '{split}' for NTU-Fi datasets")
    train_root = root / "train_amp"
    test_root = root / "test_amp"
    if not train_root.exists():
        train_root = root
    if not test_root.exists():
        test_root = root
    target_root = train_root if split == "train" else test_root
    smoothing_cfg = smoothing or {}
    method = smoothing_cfg.get("method")
    kwargs = {k: v for k, v in smoothing_cfg.items() if k != "method"}
    return NTU_Dataset(
        str(target_root),
        modal=modal,
        dtype="tensor",
        normalisation=base_normalisation,
        smoothing=method,
        smoothing_kwargs=kwargs or None,
    )


def _resolve_stats(
    dataset: str,
    root: Path,
    *,
    timesteps: int,
    canon_normalisation: str,
    base_normalisation: str,
    smoothing: Optional[Dict[str, Any]],
    modal: str,
    split: str = "train",
    log: bool = True,
) -> _CanonicalStats:
    canon_norm = _resolve_normalisation(canon_normalisation)
    base_norm = _resolve_normalisation(base_normalisation)
    
    dataset_key = dataset.lower()
    root_key = str(root.resolve())
    smoothing_key = _hashable_mapping(smoothing)
    cache_key = (
        dataset_key,
        root_key,
        timesteps,
        canon_norm,
        base_norm,
        smoothing_key,
        modal,
    )
    if cache_key in _STATS_CACHE:
        cached_stats = _STATS_CACHE[cache_key]
        verification = None
        if log:
            try:
                dataset_for_check = _load_dataset_for_stats(dataset_key, root, base_norm, modal, smoothing)
                verification = _verify_normalisation(dataset_for_check, cached_stats.pre_transform, cached_stats.normaliser)
            except Exception:
                verification = None
            _log_canonical_stats(dataset_key, root_key, timesteps, canon_norm, cached_stats, verification)
        return cached_stats

    if dataset_key in _NTU_DATASET_KEYS:
        pre = TD.compose_transforms(TD.create_channel_dim_transform, TD.downsample_time_transform(timesteps))
        smoothing_cfg = smoothing if smoothing is not None else DEFAULT_NTU_SMOOTHING
        base_dataset = _load_ntu_dataset(root, "train", base_norm, modal, smoothing_cfg)
    else:
        raise ValueError(f"Unsupported dataset '{dataset}' for canonical loader")

    cache_path = _stats_cache_path(Path(root_key), cache_key)
    cached_mean_std: Optional[Tuple[Tensor, Tensor]] = None
    if cache_path.exists():
        cached_mean_std = _load_cached_stats(cache_path)
        # if cached_mean_std is not None:
            # print(f"[CACHE] Loaded canonical stats from {cache_path}")

    metadata = {
        "dataset": dataset_key,
        "modal": modal,
        "timesteps": timesteps,
        "canonical_normalisation": canon_norm,
        "base_normalisation": base_norm,
        "smoothing": [(k, v) for k, v in smoothing_key],
    }

    norm_key = canon_normalisation.lower()
    if norm_key == "per_subcarrier":
        if cached_mean_std is None:
            mean, std = _compute_per_subcarrier_stats(base_dataset, pre)
            _save_cached_stats(cache_path, mean, std, metadata)
            # print(f"[CACHE] Stored canonical stats at {cache_path}")
        else:
            mean, std = cached_mean_std
        normaliser = TD.zscore_per_subc_transform(mean, std)
    elif norm_key == "dataset":
        if cached_mean_std is None:
            mean, std = _compute_dataset_stats(base_dataset, pre)
            _save_cached_stats(cache_path, mean, std, metadata)
            # print(f"[CACHE] Stored canonical stats at {cache_path}")
        else:
            mean, std = cached_mean_std
        normaliser = TD.zscore_dataset_transform(mean, std)
    else:
        raise ValueError(f"Unsupported canonical normalisation '{canon_normalisation}'")

    stats = _CanonicalStats(pre_transform=pre, normaliser=normaliser, mean=mean, std=std)
    _STATS_CACHE[cache_key] = stats
    if log:
        verification = _verify_normalisation(base_dataset, pre, normaliser)
        _log_canonical_stats(dataset_key, root_key, timesteps, canon_norm, stats, verification)
    return stats


def get_canonical_stats(
    dataset: str,
    root: str | Path,
    *,
    timesteps: int = 250,
    canon_normalisation: str = "per_subcarrier",
    base_normalisation: str = "none",
    smoothing: Optional[Dict[str, Any]],
    modal: str = "CSIamp",
) -> _CanonicalStats:
    root_path = _resolve_dataset_root(root)
    return _resolve_stats(
        dataset,
        root_path,
        timesteps=timesteps,
        canon_normalisation=canon_normalisation,
        base_normalisation=base_normalisation,
        smoothing=smoothing,
        modal=modal,
        log=True,
    )

def get_canonical_dataset(
    dataset: str,
    root: str | Path,
    split: str,
    task: str = "reconstruction",
    timesteps: int = 250,
    canon_normalisation: str = "per_subcarrier",
    base_normalisation: str = "none",
    smoothing: Optional[Dict[str, Any]] = None,
    modal: str = "CSIamp",
    augment: Mapping[str, float] = {},
) -> TD:
    root_path = _resolve_dataset_root(root)
    stats = _resolve_stats(
        dataset,
        root_path,
        timesteps=timesteps,
        canon_normalisation=canon_normalisation,
        base_normalisation=base_normalisation,
        smoothing=smoothing,
        modal=modal, 
        split=split,
        log=(split == "train"),
    )
    dataset_key = dataset.lower()
    if dataset_key in _NTU_DATASET_KEYS:
        smoothing_cfg = smoothing if smoothing is not None else DEFAULT_NTU_SMOOTHING
        base_dataset = _load_ntu_dataset(root_path, split, base_normalisation, modal, smoothing_cfg)
    else:
        raise ValueError(f"Unsupported dataset '{dataset}' for canonical loader")
    transform = TD.compose_transforms(stats.pre_transform, stats.normaliser)
    return TD(base_dataset, base_normalisation=base_normalisation, 
              transform=transform, task=task, augment=augment)
