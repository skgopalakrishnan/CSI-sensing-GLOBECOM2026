"""DataLoader helpers for SenseFi datasets with optional spectrogram features."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union
import logging

from torch import Generator
from torch import device as torch_device
from torch.cuda import is_available as is_cuda_available
from torch.utils.data import DataLoader, random_split

# from ...datasets.sensefi_spectrogram import (
#     SenseFiSpectrogramDataset,
#     resolve_sensefi_spectrogram_layout,
# )

from .canonical import get_canonical_dataset, get_canonical_stats
from .common import DataBundle, _resolve_three_way_split

VAL_RATIOS = {
    "NTU-Fi_HAR": 0.1,
    "NTU-Fi-HumanID": 0.1,
}

TRAIN_RATIOS = {
    "NTU-Fi_HAR": 0.7,
    "NTU-Fi-HumanID": 0.7,
}

__all__ = [
    "get_experiment_loaders"
    ]


def _default_num_workers(explicit_num_workers: Optional[int], *, use_gpu: bool) -> int:
    if explicit_num_workers is not None:
        return max(0, int(explicit_num_workers))
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 1 if use_gpu else 0
    if use_gpu:
        return max(2, min(12, cpu_count - 1))
    return max(0, min(4, cpu_count - 1))


def get_experiment_loaders(
    dataset: str,
    root: str,
    task: str, 
    timesteps: int = 250,
    batch_sizes: Optional[Dict[str, int]] = None,
    device: Optional[Union[str, torch_device]] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    base_normalisation: str = "none",
    canon_normalisation: str = "per_subcarrier",
    variant: str = "raw",
    smoothing: Optional[Dict[str, Any]] = None,
    augment_train: Mapping[str, float] = {},
    augment_eval: Mapping[str, float] = {},
    seed: int = 42,
):
    """Convenience wrapper to build train/test loaders from a dataset name and root.

    Parameters mirror the historical SenseFi helper while allowing a ``variant`` of
    ``\"spectrogram\"`` to load feature-engineered artifacts produced by the pipeline.

    - NTU-Fi_* use folders: <root>/<dataset>/(train_amp|test_amp)
    """

    project = dataset
    variant_key = variant.lower()

    if variant_key in {"raw", "amplitude"}:
        if project in ("NTU-Fi_HAR", "NTU-Fi-HumanID"):
            paths = {
                "data_train": os.path.join(root, project, "train_amp"),
                "data_test": os.path.join(root, project, "test_amp"),
            }
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    else:
        raise ValueError(f"Unknown SenseFi variant '{variant}'")

    return _get_loaders(
        project=project,
        paths=paths,
        task=task,
        batch_sizes=batch_sizes,
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        variant=variant,
        base_normalisation=base_normalisation,
        canon_normalisation=canon_normalisation,
        smoothing=smoothing,
        timesteps=timesteps,
        augment_train=augment_train,
        augment_eval=augment_eval,
        seed=seed
    )


def _get_loaders(
    project: str,
    paths: Mapping[str, str],
    task: str, 
    batch_sizes: Optional[Dict[str, int]] = None,
    device: Optional[Union[str, torch_device]] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    base_normalisation: str = "none",
    canon_normalisation: str = "per_subcarrier",
    variant: str = "raw",
    smoothing: Optional[Dict[str, Any]] = None,
    timesteps: int = 250,
    augment_train: Mapping[str, float] = {},
    augment_eval: Mapping[str, float] = {},
    seed: int = 42,
):
    """Create train/test DataLoaders for SenseFi datasets."""

    use_gpu = False
    if device is not None:
        if isinstance(device, torch_device):
            use_gpu = device.type == "cuda"
        else:
            use_gpu = str(device) == "cuda"
    else:
        use_gpu = is_cuda_available()

    pin_memory = use_gpu if pin_memory is None else bool(pin_memory)
    num_workers = _default_num_workers(num_workers, use_gpu=use_gpu)
    bs = batch_sizes or {}
    variant_key = variant.lower()

    if variant_key in {"raw", "amplitude"}:
        return _build_raw_loaders(
            project,
            task,
            paths,
            batch_sizes=bs,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            use_gpu=use_gpu,
            base_norm=base_normalisation,
            canon_norm=canon_normalisation,
            smoothing=smoothing,
            timesteps=timesteps,
            augment_train=augment_train,
            augment_eval=augment_eval,
            seed=seed)

    else:
        raise ValueError(f"Unknown dataset variant '{variant}'")

def _build_raw_loaders(
    project: str,
    task: str,
    paths: Mapping[str, str],
    batch_sizes: Mapping[str, int],
    pin_memory: bool = True,
    num_workers: int = 4,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    use_gpu: bool = False,
    timesteps: int = 250,
    base_norm: str = "none",
    canon_norm: str = "per_subcarrier",
    val_ratio: Optional[float] = None,
    train_ratio: Optional[float] = None,
    smoothing: Optional[Dict[str, Any]] = None,
    augment_train: Mapping[str, float] = {},
    augment_eval: Mapping[str, float] = {},
    seed: int = 42,
):
    """Return (train_loader, test_loader) for supported projects with minimal duplication."""

    bs_train = batch_sizes.get("train", 32)  # Low but keep it common
    bs_test = batch_sizes.get("test", 128)  # Keep it common

    # Basic config
    if project == "NTU-Fi_HAR" or project == "NTU-Fi-HumanID":
        data_root = Path(paths["data_train"]).parent        
        
    else:
        raise ValueError(f"Unknown SenseFi project '{project}'")
    
    # Common
    train_ds = get_canonical_dataset(
        project,
        data_root,
        "train",
        task,
        timesteps=timesteps,
        canon_normalisation=canon_norm,
        base_normalisation=base_norm,
        augment=augment_train,
        smoothing=smoothing,
    )

    val_dataset = None
    total_train = len(train_ds)
    if val_ratio is None or train_ratio is None:
        val_ratio = VAL_RATIOS[project]
        train_ratio = TRAIN_RATIOS[project]
        logging.info("Using default train and val ratios: %.3f, %.3f", train_ratio, val_ratio)

    train_count, val_count, _ = _resolve_three_way_split(total_train, train_ratio, val_ratio)
    val_count = max(1, val_count)
    train_count = max(1, total_train - val_count)
    generator = Generator().manual_seed(seed)
    train_ds, val_dataset = random_split(
        train_ds,
        [train_count, val_count],
        generator=generator,
    )
        
    # Create test set
    test_ds = get_canonical_dataset(
        project,
        data_root,
        "test",
        task,
        timesteps=timesteps,
        canon_normalisation=canon_norm,
        base_normalisation=base_norm,
        augment=augment_eval,
        smoothing=smoothing,
    )

    loader_kwargs: Dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = (
            bool(use_gpu) if persistent_workers is None else bool(persistent_workers)
        )
        if prefetch_factor is None:
            loader_kwargs["prefetch_factor"] = 4 if use_gpu else 2
        else:
            loader_kwargs["prefetch_factor"] = max(2, int(prefetch_factor))

    # Create dataloaders
    train_loader = DataLoader(train_ds, 
                              batch_size=bs_train,
                              shuffle=True,
                              drop_last=True,
                              **loader_kwargs)

    val_loader = None
    val_length = 0
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, 
                                batch_size=bs_test,
                                shuffle=False,
                                drop_last=False,
                                **loader_kwargs)
        val_length = len(val_dataset)
    
    test_loader = DataLoader(test_ds,
                             batch_size=bs_test,
                             shuffle=False,
                             drop_last=False,
                             **loader_kwargs)
        
    stats = get_canonical_stats(
        project,
        data_root,
        timesteps=timesteps,
        canon_normalisation=canon_norm,
        base_normalisation=base_norm,
        smoothing=smoothing)
    
    # Safely extract a sample tensor for shape info
    first_item = train_ds[0]
    if isinstance(first_item, (tuple, list)):
        sample_tensor = first_item[0]
    else:
        sample_tensor = first_item

    num_labels = 0
    if task == "classification":
        # Extract labels safely from the dataset, handling different tuple sizes
        try:
            base_labels = []
            for item in train_ds:
                if isinstance(item, (tuple, list)):
                    # Assume label is always the last element
                    label = item[-1]
                    base_labels.append(int(label))
            num_labels = len(set(base_labels))
        except Exception:
            base_labels = None
        
    info = {
        "dataset": project,
        "train_samples": len(train_ds),
        "val_samples": val_length,
        "test_samples": len(test_ds),
        "input_shape": tuple(sample_tensor.shape), # type: ignore
        "num_classes": num_labels,
        "canonical_normalization": canon_norm,
        "timesteps": timesteps,
        "zscore_mean_shape": tuple(stats.mean.shape) if stats.mean is not None else None,
        "zscore_std_shape": tuple(stats.std.shape) if stats.std is not None else None,
    }

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        info=info,
    )
