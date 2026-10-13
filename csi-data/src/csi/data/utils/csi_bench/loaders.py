import os
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader

from ...datasets.csi_bench import CSI_Bench_Dataset
from ..common.ssl_utils import (
    ApplyTransformDataset,
    ContrastiveTransformations,
    SkipNoneDataset,
    contrastive_collate,
    build_classification_transforms_timeseries,
    build_contrast_transforms_timeseries,
    build_mae_transforms_timeseries,
)
from .common import require_metadata_path, resolve_loader_name
from .label_utils import load_or_create_label_mapper


def _default_num_workers(explicit_num_workers: Optional[int], *, use_cuda: bool) -> int:
    if explicit_num_workers is not None:
        return max(0, int(explicit_num_workers))
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 1 if use_cuda else 0
    if use_cuda:
        return max(2, min(12, cpu_count - 1))
    return max(0, min(4, cpu_count - 1))


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_splits(only_splits: Optional[Sequence[str]]) -> Sequence[str]:
    if only_splits is None:
        return ["train_id", "val_id", "test_id"]
    if isinstance(only_splits, str):
        return [s.strip() for s in only_splits.split(",") if s.strip()]
    return list(only_splits)


def _build_loader_kwargs(
    *,
    num_workers: int,
    pin_memory: Optional[bool],
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    use_cuda: bool,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": (bool(pin_memory) if pin_memory is not None else bool(use_cuda)),
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = (
            _parse_bool(persistent_workers) if persistent_workers is not None else bool(use_cuda)
        )
        if prefetch_factor is None:
            kwargs["prefetch_factor"] = 4 if use_cuda else 2
        else:
            kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
    return kwargs


def _validate_task_dir(dataset_root: str, task_name: str, splits_dir_name: str) -> str:
    task_dir = os.path.join(dataset_root, task_name)
    has_metadata = os.path.exists(os.path.join(task_dir, "metadata"))
    has_splits = os.path.exists(os.path.join(task_dir, splits_dir_name))
    if not (has_metadata and has_splits):
        raise FileNotFoundError(
            f"Task directory is missing metadata or split files: {task_dir}. "
            f"Expected `metadata/` and `{splits_dir_name}/`."
        )
    return task_dir


def _build_dataset(
    dataset_root: str,
    task_name: str,
    split_name: str,
    normalisation: str,
    target_subcarriers: int,
    target_timesteps: int,
    splits_dir_name: str,
    channel_axis=None,
    verbose: bool = False,
) -> CSI_Bench_Dataset:
    return CSI_Bench_Dataset(
        dataset_root=dataset_root,
        task_name=task_name,
        split_name=split_name,
        splits_dir_name=splits_dir_name,
        normalisation=normalisation,
        target_subcarriers=target_subcarriers,
        target_timesteps=target_timesteps,
        channel_axis=channel_axis,
        verbose=verbose,
    )


def _wrap_dataset(base_dataset, transform, max_skip_retries: int):
    wrapped = ApplyTransformDataset(base_dataset, transform=transform)
    return SkipNoneDataset(wrapped, max_retries=max_skip_retries)


def _resolve_aligned_targets(
    tasks: Iterable[Dict[str, Any]],
    default_subcarriers: int,
    default_timesteps: int,
    align_subcarriers: str,
    align_timesteps: str,
) -> Tuple[int, int]:
    sub_list = [int(t.get("target_subcarriers", default_subcarriers)) for t in tasks]
    time_list = [int(t.get("target_timesteps", default_timesteps)) for t in tasks]
    aligned_sub = min(sub_list) if align_subcarriers == "min" else int(default_subcarriers)
    aligned_time = min(time_list) if align_timesteps == "min" else int(default_timesteps)
    return aligned_sub, aligned_time


def load_benchmark_supervised(
    dataset_root,
    task_name="FallDetection",
    batch_size=32,
    normalisation="zscore",
    target_subcarriers=232,
    target_timesteps=500,
    num_workers=None,
    pin_memory: Optional[bool] = None,
    only_splits=None,
    splits_dir_name="splits",
    collate_fn=None,
    verbose: bool = False,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
):
    use_cuda = torch.cuda.is_available()
    num_workers = _default_num_workers(num_workers, use_cuda=use_cuda)
    loader_kwargs = _build_loader_kwargs(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        use_cuda=use_cuda,
    )
    all_splits = _resolve_splits(only_splits)
    task_dir = _validate_task_dir(dataset_root, task_name, splits_dir_name)

    metadata_path = require_metadata_path(
        task_dir,
        allow_alternates=False,
        error_message=(
            f"Metadata file not found: {os.path.join(task_dir, 'metadata', 'sample_metadata.csv')}."
        ),
    )
    mapping_path = os.path.join(task_dir, "metadata", "label_mapping.json")
    label_mapper = load_or_create_label_mapper(
        metadata_path,
        label_column="label",
        mapping_path=mapping_path,
    )

    splits_dir = os.path.join(task_dir, splits_dir_name)
    missing_splits = [
        split for split in all_splits if not os.path.exists(os.path.join(splits_dir, f"{split}.json"))
    ]
    if missing_splits:
        raise FileNotFoundError(f"Missing split files in {splits_dir}: {missing_splits}")

    datasets = {}
    for split_name in all_splits:
        try:
            datasets[split_name] = _build_dataset(
                dataset_root=dataset_root,
                task_name=task_name,
                split_name=split_name,
                normalisation=normalisation,
                target_subcarriers=int(target_subcarriers),
                target_timesteps=int(target_timesteps),
                splits_dir_name=splits_dir_name,
                channel_axis=None,
                verbose=bool(verbose),
            )
        except Exception as exc:
            print(f"Error loading split '{split_name}': {exc}")
            datasets[split_name] = None

    loaders = {}
    for split_name, dataset in datasets.items():
        if dataset is None:
            continue
        loader_name = resolve_loader_name(split_name)
        loaders[loader_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train_id"),
            collate_fn=collate_fn,
            **loader_kwargs,
        )

    return {
        "loaders": loaders,
        "datasets": datasets,
        "num_classes": label_mapper.num_classes,
        "label_mapper": label_mapper,
    }


def load_contrastive_benchmark_ssl(
    dataset_root,
    dataset_config=None,
    n_views=2,
):
    config = dataset_config or {}
    normalisation = config.get("normalisation", "zscore")
    split_dir = config.get("split_dir", "splits")
    channel_axis = config.get("channel_axis")
    verbose = bool(config.get("verbose", False))
    max_skip_retries = int(config.get("max_skip_retries", 20))
    contrast_params = config.get("contrastive_transforms", {})
    contrast_transforms = build_contrast_transforms_timeseries(**contrast_params)
    contrast_wrapper = ContrastiveTransformations(contrast_transforms, n_views=n_views)

    ssl_tasks = config.get("ssl_tasks")
    if ssl_tasks:
        align_subcarriers = config.get("align_subcarriers", "min")
        align_timesteps = config.get("align_timesteps", "min")
        default_subcarriers = int(config.get("target_subcarriers", 232))
        default_timesteps = int(config.get("target_timesteps", 500))
        aligned_sub, aligned_time = _resolve_aligned_targets(
            ssl_tasks,
            default_subcarriers,
            default_timesteps,
            align_subcarriers,
            align_timesteps,
        )
        train_sets = []
        val_sets = []
        for task in ssl_tasks:
            task_name = task["task_name"]
            train_split = task.get("train_split", "train_id")
            val_split = task.get("val_split", "val_id")
            task_root = task.get("dataset_root", dataset_root)
            task_norm = task.get("normalisation", normalisation)
            task_split_dir = task.get("split_dir", split_dir)
            task_channel_axis = task.get("channel_axis", channel_axis)
            task_verbose = bool(task.get("verbose", verbose))

            train_base = _build_dataset(
                task_root,
                task_name,
                train_split,
                task_norm,
                aligned_sub,
                aligned_time,
                task_split_dir,
                task_channel_axis,
                task_verbose,
            )
            val_base = _build_dataset(
                task_root,
                task_name,
                val_split,
                task_norm,
                aligned_sub,
                aligned_time,
                task_split_dir,
                task_channel_axis,
                task_verbose,
            )
            train_sets.append(_wrap_dataset(train_base, contrast_wrapper, max_skip_retries))
            val_sets.append(_wrap_dataset(val_base, contrast_wrapper, max_skip_retries))
        return ConcatDataset(train_sets), ConcatDataset(val_sets)

    task_name = config.get("task_name", "FallDetection")
    train_split = config.get("train_split", "train_id")
    val_split = config.get("val_split", "val_id")
    target_subcarriers = int(config.get("target_subcarriers", 232))
    target_timesteps = int(config.get("target_timesteps", 500))

    train_base = _build_dataset(
        dataset_root,
        task_name,
        train_split,
        normalisation,
        target_subcarriers,
        target_timesteps,
        split_dir,
        channel_axis,
        verbose,
    )
    val_base = _build_dataset(
        dataset_root,
        task_name,
        val_split,
        normalisation,
        target_subcarriers,
        target_timesteps,
        split_dir,
        channel_axis,
        verbose,
    )
    return (
        _wrap_dataset(train_base, contrast_wrapper, max_skip_retries),
        _wrap_dataset(val_base, contrast_wrapper, max_skip_retries),
    )


def load_contrastive_benchmark_ssl_loaders(
    dataset_root,
    dataset_config=None,
    n_views=2,
    batch_size=32,
    num_workers=None,
    pin_memory: Optional[bool] = None,
    drop_last=True,
    collate_fn=None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
):
    config = dataset_config or {}
    train_dataset, val_dataset = load_contrastive_benchmark_ssl(
        dataset_root=dataset_root,
        dataset_config=config,
        n_views=n_views,
    )
    use_cuda = torch.cuda.is_available()
    num_workers = _default_num_workers(num_workers, use_cuda=use_cuda)
    pin_default = bool(use_cuda) if pin_memory is None else bool(pin_memory)
    pin_memory = _parse_bool(config.get("pin_memory", pin_memory), default=pin_default)
    if persistent_workers is None:
        raw_persistent = config.get("persistent_workers")
        persistent_workers = None if raw_persistent is None else _parse_bool(raw_persistent)
    if prefetch_factor is None:
        raw_prefetch = config.get("prefetch_factor")
        prefetch_factor = None if raw_prefetch is None else int(raw_prefetch)
    loader_kwargs = _build_loader_kwargs(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        use_cuda=use_cuda,
    )
    if collate_fn is None and config.get("contrastive_collate", True):
        collate_fn = contrastive_collate

    train_drop_last = config.get("train_drop_last", drop_last)
    val_drop_last = config.get("val_drop_last", False)
    train_shuffle = config.get("train_shuffle", True)
    val_shuffle = config.get("val_shuffle", False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        drop_last=train_drop_last,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=val_shuffle,
        drop_last=val_drop_last,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    return train_loader, val_loader


def load_classification_benchmark_ssl(dataset_root, dataset_config=None):
    config = dataset_config or {}
    task_name = config.get("task_name", "FallDetection")
    train_split = config.get("train_split", "train_id")
    test_split = config.get("test_split", "test_id")
    split_dir = config.get("split_dir", "splits")
    channel_axis = config.get("channel_axis")
    verbose = bool(config.get("verbose", False))
    normalisation = config.get("normalisation", "zscore")
    target_subcarriers = int(config.get("target_subcarriers", 232))
    target_timesteps = int(config.get("target_timesteps", 500))
    max_skip_retries = int(config.get("max_skip_retries", 20))
    transform = build_classification_transforms_timeseries()

    train_base = _build_dataset(
        dataset_root,
        task_name,
        train_split,
        normalisation,
        target_subcarriers,
        target_timesteps,
        split_dir,
        channel_axis,
        verbose,
    )
    test_base = _build_dataset(
        dataset_root,
        task_name,
        test_split,
        normalisation,
        target_subcarriers,
        target_timesteps,
        split_dir,
        channel_axis,
        verbose,
    )
    return (
        _wrap_dataset(train_base, transform, max_skip_retries),
        _wrap_dataset(test_base, transform, max_skip_retries),
    )


def load_classification_with_val_benchmark_ssl(dataset_root, dataset_config=None):
    config = dataset_config or {}
    task_name = config.get("task_name", "FallDetection")
    train_split = config.get("train_split", "train_id")
    val_split = config.get("val_split", "val_id")
    test_split = config.get("test_split", "test_id")
    split_dir = config.get("split_dir", "splits")
    channel_axis = config.get("channel_axis")
    verbose = bool(config.get("verbose", False))
    normalisation = config.get("normalisation", "zscore")
    target_subcarriers = int(config.get("target_subcarriers", 232))
    target_timesteps = int(config.get("target_timesteps", 500))
    max_skip_retries = int(config.get("max_skip_retries", 20))
    transform = build_classification_transforms_timeseries()

    train_base = _build_dataset(
        dataset_root,
        task_name,
        train_split,
        normalisation,
        target_subcarriers,
        target_timesteps,
        split_dir,
        channel_axis,
        verbose,
    )
    val_base = _build_dataset(
        dataset_root,
        task_name,
        val_split,
        normalisation,
        target_subcarriers,
        target_timesteps,
        split_dir,
        channel_axis,
        verbose,
    )
    test_base = _build_dataset(
        dataset_root,
        task_name,
        test_split,
        normalisation,
        target_subcarriers,
        target_timesteps,
        split_dir,
        channel_axis,
        verbose,
    )
    return (
        _wrap_dataset(train_base, transform, max_skip_retries),
        _wrap_dataset(val_base, transform, max_skip_retries),
        _wrap_dataset(test_base, transform, max_skip_retries),
    )


def load_augmented_train_benchmark_ssl(dataset_root, dataset_config=None):
    config = dataset_config or {}
    task_name = config.get("task_name", "FallDetection")
    train_split = config.get("train_split", "train_id")
    split_dir = config.get("split_dir", "splits")
    channel_axis = config.get("channel_axis")
    verbose = bool(config.get("verbose", False))
    normalisation = config.get("normalisation", "zscore")
    target_subcarriers = int(config.get("target_subcarriers", 232))
    target_timesteps = int(config.get("target_timesteps", 500))
    max_skip_retries = int(config.get("max_skip_retries", 20))
    contrast_params = config.get("contrastive_transforms", {})
    transform = build_contrast_transforms_timeseries(**contrast_params)

    train_base = _build_dataset(
        dataset_root,
        task_name,
        train_split,
        normalisation,
        target_subcarriers,
        target_timesteps,
        split_dir,
        channel_axis,
        verbose,
    )
    return _wrap_dataset(train_base, transform, max_skip_retries)


def load_mae_benchmark_ssl(dataset_root, dataset_config=None):
    config = dataset_config or {}
    normalisation = config.get("normalisation", "zscore")
    split_dir = config.get("split_dir", "splits")
    channel_axis = config.get("channel_axis")
    verbose = bool(config.get("verbose", False))
    max_skip_retries = int(config.get("max_skip_retries", 20))
    target_subcarriers = int(config.get("target_subcarriers", 232))
    target_timesteps = int(config.get("target_timesteps", 500))
    mae_cfg = config.get("mae_transforms", {})
    output_size = mae_cfg.get("output_size")
    transform = build_mae_transforms_timeseries(output_size=output_size)

    ssl_tasks = config.get("ssl_tasks")
    if ssl_tasks:
        align_subcarriers = config.get("align_subcarriers", "min")
        align_timesteps = config.get("align_timesteps", "min")
        aligned_sub, aligned_time = _resolve_aligned_targets(
            ssl_tasks,
            target_subcarriers,
            target_timesteps,
            align_subcarriers,
            align_timesteps,
        )
        train_sets = []
        val_sets = []
        for task in ssl_tasks:
            task_name = task["task_name"]
            train_split = task.get("train_split", "train_id")
            val_split = task.get("val_split", "val_id")
            task_root = task.get("dataset_root", dataset_root)
            task_norm = task.get("normalisation", normalisation)
            task_split_dir = task.get("split_dir", split_dir)
            task_channel_axis = task.get("channel_axis", channel_axis)
            task_verbose = bool(task.get("verbose", verbose))

            train_base = _build_dataset(
                task_root,
                task_name,
                train_split,
                task_norm,
                aligned_sub,
                aligned_time,
                task_split_dir,
                task_channel_axis,
                task_verbose,
            )
            val_base = _build_dataset(
                task_root,
                task_name,
                val_split,
                task_norm,
                aligned_sub,
                aligned_time,
                task_split_dir,
                task_channel_axis,
                task_verbose,
            )
            train_sets.append(_wrap_dataset(train_base, transform, max_skip_retries))
            val_sets.append(_wrap_dataset(val_base, transform, max_skip_retries))
        return ConcatDataset(train_sets), ConcatDataset(val_sets)

    task_name = config.get("task_name", "FallDetection")
    train_split = config.get("train_split", "train_id")
    val_split = config.get("val_split", "val_id")

    train_base = _build_dataset(
        dataset_root,
        task_name,
        train_split,
        normalisation,
        target_subcarriers,
        target_timesteps,
        split_dir,
        channel_axis,
        verbose,
    )
    val_base = _build_dataset(
        dataset_root,
        task_name,
        val_split,
        normalisation,
        target_subcarriers,
        target_timesteps,
        split_dir,
        channel_axis,
        verbose,
    )
    return (
        _wrap_dataset(train_base, transform, max_skip_retries),
        _wrap_dataset(val_base, transform, max_skip_retries),
    )


__all__ = [
    "load_benchmark_supervised",
    "load_contrastive_benchmark_ssl",
    "load_contrastive_benchmark_ssl_loaders",
    "load_classification_benchmark_ssl",
    "load_classification_with_val_benchmark_ssl",
    "load_augmented_train_benchmark_ssl",
    "load_mae_benchmark_ssl",
]
