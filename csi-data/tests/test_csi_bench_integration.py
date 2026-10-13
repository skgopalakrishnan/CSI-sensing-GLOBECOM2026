"""Integration smoke test for CSI-Bench real data (opt-in via env vars)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from csi.data.utils.csi_bench.loaders import load_benchmark_supervised


def _resolve_default_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "datasets" / "CSI-Bench"


def _resolve_task(root: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    tasks_dir = root / "tasks"
    if tasks_dir.exists():
        candidates = sorted(p.name for p in tasks_dir.iterdir() if p.is_dir())
    else:
        candidates = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not candidates:
        raise ValueError("No task directories found in CSI-Bench root.")
    if len(candidates) > 1:
        print(f"CSI_BENCH_TASK not set; defaulting to '{candidates[0]}' from {candidates}")
    print(candidates[0])
    return candidates[0]  # Pick first alphabetically


def _next_non_none(loader):
    for batch in loader:
        if batch is not None:
            return batch
    return None


def test_csi_bench_real_data_loader() -> None:
    root = Path(os.environ.get("CSI_BENCH_ROOT", _resolve_default_root()))
    if not root.exists():
        pytest.skip(f"CSI-Bench dataset root not found at {root}")

    task = _resolve_task(root, os.environ.get("CSI_BENCH_TASK"))

    file_format = os.environ.get("CSI_BENCH_FORMAT", "h5")
    task_dir = (root / "tasks" / task) if (root / "tasks" / task).exists() else (root / task)
    default_split = "data_test_id" if (task_dir / "splits" / "data_test_id.json").exists() else "test_id"
    test_splits = os.environ.get("CSI_BENCH_TEST_SPLITS", default_split)
    use_root_as_task_dir = os.environ.get("CSI_BENCH_USE_ROOT", "0") == "1"
    normalisation = os.environ.get("CSI_BENCH_NORM", "zscore")
    target_subcarriers = int(os.environ.get("CSI_BENCH_SUBCARRIERS", "232"))
    target_timesteps = int(os.environ.get("CSI_BENCH_TIMESTEPS", "500"))

    data = load_benchmark_supervised(
        dataset_root=root,
        task_name=task,
        batch_size=2,
        file_format=file_format,
        test_splits=test_splits,
        only_splits=test_splits,
        use_root_as_task_dir=use_root_as_task_dir,
        normalisation=normalisation,
        target_subcarriers=target_subcarriers,
        target_timesteps=target_timesteps,
        shuffle_train=False,
        num_workers=0,
        pin_memory=False,
        skip_errors=True,
    )

    loaders = data["loaders"]
    assert loaders, "Expected at least one loader."
    test_loaders = {k: v for k, v in loaders.items() if k.startswith("test")}
    assert test_loaders, "Expected at least one test split loader."

    first_key = next(iter(loaders))
    batch = _next_non_none(loaders[first_key])
    assert batch is not None, "All training batches were skipped."
    x, y = batch
    assert isinstance(x, torch.Tensor)
    assert x.dim() == 4
    assert x.shape[1:] == (1, target_subcarriers, target_timesteps)
    assert y is not None
