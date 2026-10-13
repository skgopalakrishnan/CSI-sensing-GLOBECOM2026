from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader

from torch.nn.functional import mse_loss, l1_loss


__all__ = [
    "TrainingEpochResult",
    "TrainingSummary",
    "AutoencoderEpochResult",
    "AutoencoderTrainingSummary",
    "AutoencoderLossStats",
    "reconstruction_loss",
    "set_requires_grad",
    "count_trainable_parameters",
    "_resolve_augment",
    "build_scheduler",
    "train_classifier_epoch",
    "maybe_tqdm",
]


@dataclass
class TrainingEpochResult:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None


@dataclass
class TrainingSummary:
    history: List[TrainingEpochResult]
    best_epoch: int
    best_state: Optional[Dict[str, Tensor]]
    best_metric: Optional[float]
    best_val_accuracy: Optional[float]
    best_val_loss: Optional[float]
    final_train_accuracy: float
    final_train_loss: float


@dataclass
class AutoencoderEpochResult:
    epoch: int
    train_loss: float
    train_mse: float
    train_mae: float
    train_contractive: float = 0.0
    val_loss: Optional[float] = None
    val_mse: Optional[float] = None
    val_mae: Optional[float] = None


@dataclass
class AutoencoderTrainingSummary:
    history: List[AutoencoderEpochResult]
    best_epoch: int
    best_state: Optional[Dict[str, Tensor]]
    best_metric: Optional[float]
    best_val_loss: Optional[float]
    final_train_loss: float
    final_train_mse: float
    final_train_mae: float
    final_train_contractive: float = 0.0


@dataclass
class AutoencoderLossStats:
    """Aggregated metrics for a single autoencoder training epoch."""

    loss: float
    mse: float
    mae: float
    contractive: float = 0.0


def _maybe_tqdm(iterable, use_tqdm: bool, desc: Optional[str] = None):
    if use_tqdm:
        try:
            from tqdm import tqdm  # type: ignore

            return tqdm(iterable, desc=desc)
        except Exception:
            pass
    return iterable


def maybe_tqdm(iterable, use_tqdm: bool, desc: Optional[str] = None):
    """Public alias for optional tqdm wrapping."""

    return _maybe_tqdm(iterable, use_tqdm, desc)


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _resolve_amp_dtype(amp_dtype: Optional[str]) -> torch.dtype:
    if amp_dtype is None:
        return torch.float16
    key = str(amp_dtype).strip().lower()
    if key in {"float16", "fp16", "half"}:
        return torch.float16
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported amp_dtype '{amp_dtype}'. Use 'float16' or 'bfloat16'.")


def _autocast_context(*, device: torch.device, enabled: bool, dtype: torch.dtype):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    from contextlib import nullcontext

    return nullcontext()


def _augment_batch(batch: torch.Tensor, aug_params: Optional[Dict[str, Union[int, float]]]) -> torch.Tensor:
    if aug_params is None or not aug_params:
        return batch

    x = batch.clone()
    if float(aug_params.get("gaussian_std", 0.0)) > 0.0:
        x = x + torch.randn_like(x) * float(aug_params["gaussian_std"])
    if (
        float(aug_params.get("time_mask_fraction", 0.0)) > 0.0
        and int(aug_params.get("time_mask_count", 0)) > 0
    ):
        B, C, S, T = x.shape
        width = max(1, int(T * float(aug_params["time_mask_fraction"])))
        for b in range(B):
            for _ in range(int(aug_params["time_mask_count"])):
                if T - width <= 0:
                    continue
                start = np.random.randint(0, T - width + 1)
                x[b, :, :, start : start + width] = 0
    if float(aug_params.get("subcarrier_mask_fraction", 0.0)) > 0.0:
        B, C, S, T = x.shape
        mask_count = max(1, int(S * float(aug_params["subcarrier_mask_fraction"])))
        for b in range(B):
            actual_mask_count = min(mask_count, S)
            idx = np.random.choice(S, actual_mask_count, replace=False)
            x[b, :, idx, :] = 0
    if float(aug_params.get("antenna_dropout_prob", 0.0)) > 0.0:
        dropout_mask = (
            torch.rand(x.shape[:2], device=x.device) > float(aug_params["antenna_dropout_prob"])
        ).float()
        mask_shape = list(x.shape[:2]) + [1] * (x.dim() - 2)
        dropout_mask = dropout_mask.view(*mask_shape)
        x = x * dropout_mask
    return x


def _resolve_augment(aug_cfg: Dict) -> Optional[Callable[[torch.Tensor, Dict], torch.Tensor]]:
    aug_params: Dict[str, Union[int, float]] = {
        "time_mask_fraction": float(aug_cfg.get("time_mask_fraction", 0.0)),
        "time_mask_count": int(aug_cfg.get("time_mask_count", 0)),
        "subcarrier_mask_fraction": float(aug_cfg.get("subcarrier_mask_fraction", 0.0)),
        "antenna_dropout_prob": float(aug_cfg.get("antenna_dropout", 0.0)),
        "gaussian_std": float(aug_cfg.get("gaussian_std", 0.0)),
    }
    if any(
        [
            aug_params["time_mask_fraction"] > 0.0,
            aug_params["time_mask_count"] > 0,
            aug_params["subcarrier_mask_fraction"] > 0.0,
            aug_params["antenna_dropout_prob"] > 0.0,
            aug_params["gaussian_std"] > 0.0,
        ]
    ):
        return _augment_batch
    return None


def reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    mask_focus: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return weighted reconstruction loss along with MSE/MAE diagnostics."""

    mse_map = mse_loss(prediction, target, reduction="none")
    mae_map = l1_loss(prediction, target, reduction="none")
    weights = 1.0 + mask_focus * (1.0 - mask)
    weighted = mse_map * weights
    return weighted.mean(), mse_map.mean(), mae_map.mean()


def set_requires_grad(module: nn.Module, value: bool) -> None:
    """Toggle the ``requires_grad`` flag for every parameter in a module."""

    for param in module.parameters():
        param.requires_grad = value


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the number of parameters that will receive gradients."""

    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Optional[Mapping[str, Any]],
    epochs: int,
) -> Optional[torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """Instantiate a ReduceLROnPlateau scheduler from configuration."""

    if not cfg:
        return None
    sched_type = str(cfg.get("type", "reduce_on_plateau")).lower()
    if sched_type in {"cosine_warmup", "warmup_cosine", "cosine"}:
        warmup_epochs = int(cfg.get("warmup_epochs", 0))

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if sched_type in {"reduce_on_plateau", "reduceonplateau", "rop"}:
        mode = str(cfg.get("mode", "max")).lower()
        if mode not in {"min", "max"}:
            raise ValueError(f"Unsupported ReduceLROnPlateau mode '{mode}'. Expected 'min' or 'max'.")
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(cfg.get("factor", 0.5)),
            patience=int(cfg.get("patience", 5)),
            threshold=float(cfg.get("threshold", 1e-4)),
            min_lr=float(cfg.get("min_lr", 0.0)),
        )
    raise ValueError(
        f"Unsupported scheduler type '{sched_type}'. Only 'reduce_on_plateau' is supported."
    )


def train_classifier_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    grad_clip: Optional[float] = None,
    use_tqdm: bool = False,
    freeze_encoder: bool = False,
    progress_desc: str = "Train",
    extra_forward_args: Optional[Sequence[object]] = None,
    extra_forward_kwargs: Optional[Mapping[str, object]] = None,
    amp: Optional[bool] = None,
    amp_dtype: str = "float16",
) -> Dict[str, float]:
    """Run one supervised training epoch for a classifier style model."""

    if loader is None:
        raise ValueError("Loader must be provided for classifier training")

    model.train()
    if freeze_encoder and hasattr(model, "encoder"):
        encoder = getattr(model, "encoder")
        if isinstance(encoder, nn.Module):
            encoder.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    iterator = _maybe_tqdm(loader, use_tqdm, desc=progress_desc)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    forward_args: Sequence[object] = tuple(extra_forward_args or ())
    forward_kwargs = dict(extra_forward_kwargs or {})
    resolved_amp_dtype = _resolve_amp_dtype(amp_dtype)
    amp_enabled = bool(amp) if amp is not None else (device.type == "cuda")
    scaler = _make_grad_scaler(
        enabled=(amp_enabled and device.type == "cuda" and resolved_amp_dtype == torch.float16)
    )

    for batch in iterator:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            inputs, labels = batch[0], batch[1]
        else:
            raise ValueError("Classifier training expects batches of (inputs, labels)")
        non_blocking = device.type == "cuda"
        inputs = inputs.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking).long()
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device=device, enabled=amp_enabled, dtype=resolved_amp_dtype):
            outputs = model(inputs, *forward_args, **forward_kwargs)
            loss = criterion(outputs, labels)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0.0 and trainable_params:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0.0 and trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
        total_samples += batch_size

        if use_tqdm:
            postfix = getattr(iterator, "set_postfix", None)
            if callable(postfix):
                try:
                    postfix(loss=f"{loss.item():.4f}")
                except Exception:
                    pass

    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }
