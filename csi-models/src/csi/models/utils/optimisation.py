"""Utils for Optuna-based hyperparameter tuning of supervised models."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol

import torch
from torch import nn

from .datasets import DatasetSpec
from ..supervised.models import MODEL_BUILDERS


__all__ = [
    "suggest_hparams_for_model",
    "build_model_from_hparams",
    "CUDATrialOOMError",
    "run_with_oom_guard",
]

####################################################################################################

class TrialLike(Protocol):
    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1, log: bool = False) -> int:
        ...

    def suggest_float(self, name: str, low: float, high: float, *, step: Optional[float] = None, log: bool = False) -> float:
        ...

    def suggest_categorical(self, name: str, choices: List[Any]) -> Any:
        ...


def _normalise_model_name(name: str) -> str:
    key = str(name).lower().replace(" ", "").replace("-", "").replace("_", "")
    aliases = {
        "timesformer": "csitimesformer",
        "timesformer1d": "csitimesformer",
    }
    return aliases.get(key, key)


def _divisor_choices(value: int, *, min_divisor: int = 2, max_divisor: Optional[int] = None) -> List[int]:
    upper = value if max_divisor is None else min(value, max_divisor)
    choices = [d for d in range(max(1, min_divisor), upper + 1) if value % d == 0]
    if not choices:
        return [1]
    return choices


class CUDATrialOOMError(RuntimeError):
    """Raised when a CUDA OOM is detected during a train/eval step."""


def _is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "cudnn" in message)


def run_with_oom_guard(step_fn: Callable[[], Any]) -> Any:
    """Run one training/evaluation step and convert CUDA OOM to a typed error."""

    try:
        return step_fn()
    except RuntimeError as exc:
        if not _is_cuda_oom_error(exc):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise CUDATrialOOMError("CUDA OOM during trial step; cleared CUDA cache.") from exc


def suggest_hparams_for_model(
    trial: TrialLike,
    model_name: str,
    spec: DatasetSpec,
) -> Dict[str, Any]:
    """Return Optuna-suggested hparams for one supervised model."""

    key = _normalise_model_name(model_name)
    if key == "mlp":
        return {
            "hidden_dim": trial.suggest_int("hidden_dim", 128, 1024, step=128),
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.6, step=0.05),
        }
    if key == "lenet":
        return {
            "base_filters": trial.suggest_int("base_filters", 16, 128, step=16),
            "fc_hidden_dim": trial.suggest_int("fc_hidden_dim", 128, 768, step=64),
            "pool_size": trial.suggest_categorical("pool_size", [2, 3, 4]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.05),
        }
    if key == "resnet18":
        return {
            "pretrained": False,
        }
    if key == "bilstm":
        return {
            "hidden_dim": trial.suggest_int("hidden_dim", 64, 512, step=64),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.05),
        }
    if key == "csitimesformer":
        num_heads = int(trial.suggest_categorical("num_heads", [2, 4, 8]))
        emb_choices = [dim for dim in [64, 96, 128, 160, 192, 256] if dim % num_heads == 0]
        return {
            "patch_f": int(trial.suggest_categorical("patch_f", _divisor_choices(spec.freq_bins, min_divisor=2, max_divisor=32))),
            "patch_t": int(trial.suggest_categorical("patch_t", _divisor_choices(spec.time_steps, min_divisor=10, max_divisor=128))),
            "emb_dim": int(trial.suggest_categorical("emb_dim", emb_choices)),
            "depth": trial.suggest_int("depth", 2, 6),
            "num_heads": num_heads,
            "mlp_ratio": float(trial.suggest_categorical("mlp_ratio", [2.0, 3.0, 4.0])),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4, step=0.05),
        }
    raise ValueError(f"Unsupported model for tuning: {model_name}")


def build_model_from_hparams(
    spec: DatasetSpec,
    model_name: str,
    num_classes: int,
    hparams: Dict[str, Any],
) -> nn.Module:
    """Build one supervised model from a sampled Optuna hparam dict."""
    key = _normalise_model_name(model_name)
    if key not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model '{model_name}'. Available: {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[key](spec, num_classes=num_classes, **hparams)

####################################################################################################
