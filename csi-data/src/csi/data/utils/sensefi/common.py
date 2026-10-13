from pathlib import Path
from typing import Mapping, Tuple, Optional, Callable, Any, Union, Dict
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import torch
from torch import Tensor


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader]
    info: Dict[str, Any]


def _resolve_dataset_root(path: str | Path, 
                          repo_root: Optional[Union[str, Path]] = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if repo_root is not None:
        root_candidate = repo_root / candidate
        if root_candidate.exists():
            return root_candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    print("Warning: Unable to resolve dataset root; using given path as-is.")
    return Path(path)


def _resolve_normalisation(value: Optional[str]) -> str:
    if value is None:
        return "per_subcarrier"
    key = str(value).lower()
    if key in {"per_subcarrier", "canonical", "canonical_per_subcarrier"}:
        return "per_subcarrier"
    if key in {"none", "identity"}:
        return "none"
    if key in {"zscore", "minmax", "min-max", "z-score"}:
        if key == "zscore" or key == "z-score":
            return "zscore"
        return "minmax"
    raise ValueError(f"Unsupported canonical normalisation '{value}'")


def _resolve_batch_sizes(cfg: Mapping[str, int]) -> Tuple[int, int, int]:
    train = int(cfg.get("train", cfg.get("train_batch_size", 32)))
    val = int(cfg.get("val", cfg.get("val_batch_size", cfg.get("eval", train))))
    test = int(cfg.get("test", cfg.get("test_batch_size", val)))
    return train, val, test


def _resolve_three_way_split(total: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    if total <= 0:
        raise ValueError("Dataset must contain at least one sample")
    train_len = max(1, int(round(total * train_ratio)))
    val_len = max(1, int(round(total * val_ratio)))
    test_len = total - train_len - val_len

    if test_len < 0:
        test_len = 0
    # Reduce lengths if they exceed the dataset size
    while train_len + val_len + test_len > total:
        if train_len >= val_len and train_len > 1:
            train_len -= 1
        elif val_len > 1:
            val_len -= 1
        else:
            test_len -= 1
            if test_len < 0:
                test_len = 0
                break
    # Ensure the sum matches the dataset size
    while train_len + val_len + test_len < total:
        test_len += 1

    return train_len, val_len, test_len


def _compute_per_subcarrier_stats(dataset: Dataset, transform: Optional[Callable[[Tensor], Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function assumes the shape of the dataset is (C, F, T) or (B, C, F, T). Summing with dim=-1 (time)."""
    sum_ = None
    sum_sq = None
    count = 0
    for sample, _ in dataset:
        tensor = sample.float() if isinstance(sample, torch.Tensor) else torch.as_tensor(sample, dtype=torch.float32)
        if transform is not None:
            tensor = transform(tensor)
        if tensor.dim() < 3:
            continue
        tensor = tensor.detach().cpu()
        c, f, cur_t = tensor.shape[0], tensor.shape[1], tensor.shape[2]
        values = tensor.view(c, f, cur_t)
        if sum_ is None:
            sum_ = torch.zeros(c, f, dtype=torch.float64)
            sum_sq = torch.zeros(c, f, dtype=torch.float64)
        sum_ += values.sum(dim=-1)
        assert sum_ is not None and sum_sq is not None  # preventing warnings
        sum_sq += (values ** 2).sum(dim=-1)
        count += cur_t
    if sum_ is None or count == 0:
        raise RuntimeError("Unable to compute per-subcarrier statistics; dataset produced no samples.")
    mean = (sum_ / count).to(torch.float32)
    assert sum_sq is not None  # preventing warnings
    var = torch.clamp((sum_sq / count) - mean.double() ** 2, min=1e-6)
    std = torch.sqrt(var).to(torch.float32)
    return mean, std


def _compute_dataset_stats(dataset: Dataset, transform: Optional[Callable[[Tensor], Tensor]]) -> Tuple[Tensor, Tensor]:
    total_sum = 0.0
    total_sq = 0.0
    total_count = 0
    for sample, _ in dataset:
        tensor = sample if isinstance(sample, Tensor) else torch.as_tensor(sample)
        if transform is not None:
            tensor = transform(tensor.float())
        tensor = tensor.detach().cpu()
        total_sum += tensor.sum().item()
        total_sq += (tensor.double() ** 2).sum().item()
        total_count += tensor.numel()
    if total_count == 0:
        raise RuntimeError("Unable to compute dataset statistics; dataset produced no samples.")
    mean = torch.tensor(total_sum / total_count, dtype=torch.float32)
    var = max((total_sq / total_count) - float(mean) ** 2, 1e-6)
    std = torch.tensor(var, dtype=torch.float32).sqrt()
    return mean, std

def _hashable_mapping(mapping: Optional[Mapping[str, Any]]) -> Tuple[Tuple[str, Any], ...]:
    if not mapping:
        return ()
    return tuple(sorted((str(k), mapping[k]) for k in mapping))
