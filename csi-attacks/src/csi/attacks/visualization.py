from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def _prepare_heatmap_array(delta: np.ndarray) -> np.ndarray:
    if delta.ndim == 1:
        return delta[np.newaxis, :]
    if delta.ndim == 2:
        return delta
    time_axis = int(np.argmax(delta.shape))
    reshaped = np.moveaxis(delta, time_axis, -1)
    return reshaped.reshape(-1, reshaped.shape[-1])


def save_perturbation_heatmaps(
    differences: Any,
    out_dir: str,
    prefix: str,
    *,
    samples: int = 3,
    indices: Optional[Sequence[int]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    cmap: str = "viridis",
) -> Sequence[str]:
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    if differences is None:
        return []
    try:
        diffs_np = differences.detach().cpu().numpy()  # type: ignore[attr-defined]
    except AttributeError:
        diffs_np = np.asarray(differences)

    if diffs_np.size == 0:
        return []

    num_samples = diffs_np.shape[0]
    if indices is None:
        if num_samples <= samples:
            selected = list(range(num_samples))
        else:
            selected = np.linspace(0, num_samples - 1, num=samples, dtype=int).tolist()
    else:
        selected = [i for i in indices if 0 <= i < num_samples]
        if not selected:
            return []

    os.makedirs(out_dir, exist_ok=True)

    saved_paths = []
    meta_lines = []
    if metadata:
        for key, value in metadata.items():
            if value is None:
                continue
            meta_lines.append(f"{key}: {value}")
    meta_text = " | ".join(meta_lines)

    for idx in selected:
        delta = np.abs(diffs_np[idx])
        heatmap = _prepare_heatmap_array(delta)
        vmax = np.max(heatmap) if heatmap.size else 1.0
        vmin = 0.0

        fig, ax = plt.subplots(figsize=(max(6.0, heatmap.shape[1] / 40.0), 4.0))
        im = ax.imshow(heatmap, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_ylabel("Dimension")
        ax.set_xlabel("Time")
        title = f"{prefix} | sample {idx}"
        if meta_text:
            title = f"{title}\n{meta_text}"
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("|perturbation|", rotation=270, labelpad=15)
        fig.tight_layout()
        fname = f"{prefix}_sample{idx:03d}.png"
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=200)
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths


def truncate_loader(loader: DataLoader, max_batches: Optional[int]) -> DataLoader:
    if max_batches is None or int(max_batches) <= 0:
        return loader
    max_batches = int(max_batches)
    base_bs = getattr(loader, "batch_size", None) or 256
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return loader
    n_total = len(dataset)
    max_samples = min(max_batches * base_bs, n_total)
    if max_samples <= 0:
        return loader
    indices = list(range(max_samples))
    subset = Subset(dataset, indices)
    num_workers = int(getattr(loader, "num_workers", 0) or 0)
    pin_memory = bool(getattr(loader, "pin_memory", False))
    persistent_workers = bool(getattr(loader, "persistent_workers", False)) if num_workers > 0 else False
    dl_kwargs = {
        "batch_size": base_bs,
        "shuffle": False,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        prefetch = getattr(loader, "prefetch_factor", None)
        if prefetch is not None:
            dl_kwargs["prefetch_factor"] = prefetch
    return DataLoader(subset, **dl_kwargs)


__all__ = ["save_perturbation_heatmaps", "truncate_loader"]
