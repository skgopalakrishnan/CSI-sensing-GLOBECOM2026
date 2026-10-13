"""Minimal dataset smoke tests for csi-data."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from csi.data.datasets import CSI_Bench_Dataset
from csi.data.utils.csi_bench.loaders import load_benchmark_supervised


def _write_h5(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("CSI_amps", data=array)


def _build_csi_bench(tmp_path: Path) -> Path:
    root = tmp_path / "bench"
    task_dir = root / "motion_task"
    data_dir = task_dir / "data"
    metadata_dir = task_dir / "metadata"
    splits_dir = task_dir / "splits"

    rng = np.random.default_rng(1)
    entries = []
    split_ids = []
    for idx in range(3):
        sample = rng.standard_normal((232, 500), dtype=np.float32)
        file_path = data_dir / f"sample_{idx}.h5"
        _write_h5(file_path, sample)
        entries.append({"id": idx, "file_path": f"data/sample_{idx}.h5", "label": "walk"})
        split_ids.append(idx)

    metadata_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(entries).to_csv(metadata_dir / "sample_metadata.csv", index=False)
    with open(splits_dir / "train_id.json", "w", encoding="utf-8") as handle:
        import json
        json.dump(split_ids, handle)
    return root

def test_csi_bench_dataset_shapes_and_norm(tmp_path) -> None:
    root = _build_csi_bench(tmp_path)
    dataset = CSI_Bench_Dataset(
        dataset_root=str(root),
        task_name="motion_task",
        split_name="train_id",
        normalisation="zscore",
        target_subcarriers=232,
        target_timesteps=500,
    )

    x, y = dataset[0]
    assert x.shape == (1, 232, 500)
    assert isinstance(y, (int, np.integer))
    mean = x.mean(dim=-1)
    std = x.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-4)


def _resolve_csi_bench_root() -> Path:
    return Path(__file__).resolve().parents[2] / "datasets" / "CSI-Bench"


def _resolve_csi_bench_task(root: Path) -> str | None:
    candidates = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "ssl_split_reports")
    if not candidates:
        return None
    return candidates[0]


def test_csi_bench_real_data_loader() -> None:
    root = _resolve_csi_bench_root()
    if not root.exists():
        return
    task = _resolve_csi_bench_task(root)
    if task is None:
        return

    task_dir = root / task
    default_split = "data_test_id" if (task_dir / "splits" / "data_test_id.json").exists() else "test_id"

    data = load_benchmark_supervised(
        dataset_root=str(root),
        task_name=task,
        batch_size=1,
        only_splits=default_split,
        num_workers=0,
        pin_memory=False,
    )

    loaders = data["loaders"]
    assert loaders, "Expected at least one loader."

    loader = next(iter(loaders.values()))
    batch = None
    for item in loader:
        if item is not None:
            batch = item
            break
    assert batch is not None, "All batches were skipped."
    x, _ = batch
    mean = x.mean(dim=-1)
    std = x.std(dim=-1)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-1)
    assert torch.allclose(std, torch.ones_like(std), atol=2e-1)
