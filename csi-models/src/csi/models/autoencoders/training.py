"""Autoencoder training and evaluation helpers."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..utils.training import (
    AutoencoderEpochResult,
    AutoencoderLossStats,
    AutoencoderTrainingSummary,
    _autocast_context,
    _make_grad_scaler,
    _resolve_amp_dtype,
    build_scheduler,
    reconstruction_loss,
    maybe_tqdm,
)

__all__ = [
    "train_autoencoder",
    "train_autoencoder_epoch",
    "evaluate_autoencoder",
]


def _coerce_autoencoder_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalise differing batch structures for autoencoder training."""

    if isinstance(batch, dict):
        noisy = next(
            (
                batch[key]
                for key in ("data", "noisy", "input", "inputs")
                if isinstance(batch.get(key), torch.Tensor)
            ),
            None,
        )
        if noisy is None:
            raise ValueError("Autoencoder batch dict must include a tensor under 'data' or 'noisy'")
        target = batch.get("target")
        if not isinstance(target, torch.Tensor):
            target = noisy
        mask = batch.get("mask")
        if not isinstance(mask, torch.Tensor):
            mask = torch.ones_like(target)
        return noisy, target, mask

    if isinstance(batch, (tuple, list)):
        tensors = [item for item in batch if isinstance(item, torch.Tensor)]
        if not tensors:
            raise ValueError("Autoencoder batch tuple/list must contain tensors")
        noisy = tensors[0]
        target = noisy
        if len(tensors) >= 2 and tensors[1].shape == noisy.shape:
            target = tensors[1]
        mask = torch.ones_like(target)
        if len(tensors) >= 3 and tensors[2].shape == noisy.shape:
            mask = tensors[2]
        return noisy, target, mask

    if isinstance(batch, torch.Tensor):
        return batch, batch, torch.ones_like(batch)

    raise ValueError("Unsupported autoencoder batch structure; expected tensor, tuple/list, or dict.")


def train_autoencoder_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    mask_focus: float,
    grad_clip: Optional[float] = None,
    contractive_lambda: float = 0.0,
    use_tqdm: bool = False,
    contractive_interval: int = 1,
    contractive_sample_ratio: float = 1.0,
    amp: Optional[bool] = None,
    amp_dtype: str = "float16",
) -> AutoencoderLossStats:
    """Train a single epoch and return aggregated metrics."""

    model.train()
    iterator = maybe_tqdm(loader, use_tqdm=use_tqdm, desc="Training")

    loss_accum = 0.0
    mse_accum = 0.0
    mae_accum = 0.0
    contractive_accum = 0.0
    samples = 0
    interval = max(1, contractive_interval)
    sample_ratio = max(0.0, min(1.0, contractive_sample_ratio))
    batch_idx = 0
    resolved_amp_dtype = _resolve_amp_dtype(amp_dtype)
    amp_enabled = bool(amp) if amp is not None else (device.type == "cuda")
    scaler = _make_grad_scaler(
        enabled=(amp_enabled and device.type == "cuda" and resolved_amp_dtype == torch.float16)
    )

    for batch in iterator:
        noisy, target, mask = _coerce_autoencoder_batch(batch)
        non_blocking = device.type == "cuda"
        noisy = noisy.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)
        mask = mask.to(device=device, dtype=target.dtype, non_blocking=non_blocking)

        compute_contractive = contractive_lambda > 0.0 and (batch_idx % interval == 0)
        if compute_contractive:
            noisy.requires_grad_(True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device=device, enabled=amp_enabled, dtype=resolved_amp_dtype):
            reconstruction, latent = model(noisy)
            loss, mse_val, mae_val = reconstruction_loss(
                reconstruction,
                target,
                mask,
                mask_focus=mask_focus,
            )

        contractive_value = 0.0
        if compute_contractive:
            flat_latent = latent.view(latent.size(0), -1)
            if sample_ratio < 1.0 and flat_latent.size(1) > 1:
                feature_count = max(1, int(flat_latent.size(1) * sample_ratio))
                if feature_count < flat_latent.size(1):
                    indices = torch.randperm(flat_latent.size(1), device=flat_latent.device)[:feature_count]
                    latent_sum = flat_latent[:, indices].sum(dim=1)
                else:
                    latent_sum = flat_latent.sum(dim=1)
            else:
                latent_sum = flat_latent.sum(dim=1)
            jacobian = torch.autograd.grad(
                latent_sum.sum(),
                noisy,
                create_graph=True,
                retain_graph=True,
            )[0]
            contractive_term = contractive_lambda * jacobian.pow(2).sum(dim=[1, 2, 3]).mean()
            loss = loss + contractive_term
            contractive_value = contractive_term.detach().item()

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = noisy.size(0)
        loss_accum += loss.item() * batch_size
        mse_accum += mse_val.item() * batch_size
        mae_accum += mae_val.item() * batch_size
        contractive_accum += contractive_value * batch_size
        samples += batch_size

        if use_tqdm:
            postfix = getattr(iterator, "set_postfix", None)
            if callable(postfix):
                try:
                    postfix(loss=f"{loss.item():.4f}", mse=f"{mse_val.item():.4f}")
                except Exception:
                    pass

        batch_idx += 1

    if samples == 0:
        return AutoencoderLossStats(loss=0.0, mse=0.0, mae=0.0, contractive=0.0)
    return AutoencoderLossStats(
        loss=loss_accum / samples,
        mse=mse_accum / samples,
        mae=mae_accum / samples,
        contractive=contractive_accum / samples,
    )


def train_autoencoder(
    model: torch.nn.Module,
    tensor_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    *,
    use_tqdm: bool = False,
    weight_decay: float = 0.0,
    val_loader: Optional[DataLoader] = None,
    patience: Optional[int] = None,
    min_epochs: int = 0,
    monitor_metric: str = "loss",
    monitor_mode: Optional[str] = None,
    min_delta: float = 0.0,
    restore_best: bool = True,
    grad_clip: Optional[float] = None,
    optimizer_cfg: Optional[Dict[str, Any]] = None,
    scheduler_cfg: Optional[Dict[str, Any]] = None,
    mask_focus: float = 0.0,
    contractive_lambda: float = 0.0,
    contractive_interval: int = 1,
    contractive_sample_ratio: float = 1.0,
    amp: Optional[bool] = None,
    amp_dtype: str = "float16",
) -> AutoencoderTrainingSummary:
    model = model.to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable autoencoder parameters found (all parameters are frozen).")

    optimizer_type = str((optimizer_cfg or {}).get("type", "adamw")).lower()
    if optimizer_type in {"adamw", "adam_w"}:
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type in {"adam"}:
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    scheduler = build_scheduler(optimizer, scheduler_cfg, num_epochs)

    metric_key = str(monitor_metric).strip().lower()
    resolved_mode = monitor_mode.lower() if monitor_mode else None
    if resolved_mode not in {"max", "min", None}:
        raise ValueError(f"monitor_mode must be 'max', 'min', or None, got {monitor_mode}")
    if resolved_mode is None:
        resolved_mode = "min"

    if val_loader is None:
        patience = None

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss: Optional[float] = None
    best_epoch = 0
    history: List[AutoencoderEpochResult] = []
    best_metric = float("-inf") if resolved_mode == "max" else float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        train_stats = train_autoencoder_epoch(
            model,
            tensor_loader,
            optimizer,
            device,
            mask_focus=mask_focus,
            grad_clip=grad_clip,
            contractive_lambda=contractive_lambda,
            use_tqdm=use_tqdm,
            contractive_interval=contractive_interval,
            contractive_sample_ratio=contractive_sample_ratio,
            amp=amp,
            amp_dtype=amp_dtype,
        )
        val_metrics = evaluate_autoencoder(model, val_loader, device, amp=amp, amp_dtype=amp_dtype)

        history.append(
            AutoencoderEpochResult(
                epoch=epoch + 1,
                train_loss=train_stats.loss,
                train_mse=train_stats.mse,
                train_mae=train_stats.mae,
                train_contractive=train_stats.contractive,
                val_loss=val_metrics.get("loss"),
                val_mse=val_metrics.get("mse"),
                val_mae=val_metrics.get("mae"),
            )
        )

        log_parts = [
            f"epoch={epoch + 1}/{num_epochs}",
            f"train_loss={train_stats.loss:.4f}",
            f"train_mse={train_stats.mse:.4f}",
            f"train_mae={train_stats.mae:.4f}",
        ]
        if train_stats.contractive > 0.0:
            log_parts.append(f"train_contractive={train_stats.contractive:.4f}")
        if val_metrics:
            log_parts.append(f"val_loss={val_metrics.get('loss', 0.0):.4f}")
            log_parts.append(f"val_mse={val_metrics.get('mse', 0.0):.4f}")
            log_parts.append(f"val_mae={val_metrics.get('mae', 0.0):.4f}")
        print("[AE-TRAIN] " + " | ".join(log_parts))

        monitored_value = val_metrics.get(metric_key)
        if val_loader is not None and monitored_value is not None:
            improved = (
                monitored_value > (best_metric + min_delta)
                if resolved_mode == "max"
                else monitored_value < (best_metric - min_delta)
            )
            if improved:
                best_metric = float(monitored_value)
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                best_val_loss = float(val_metrics.get("loss", monitored_value))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_target = val_metrics.get(metric_key, train_stats.loss)
                scheduler.step(float(scheduler_target))
            else:
                scheduler.step()

        if (
            patience is not None
            and val_loader is not None
            and epoch + 1 >= max(1, min_epochs)
            and epochs_no_improve >= patience
        ):
            print(f"[AE-EARLY-STOP] Stopping at epoch {epoch + 1} (best epoch {best_epoch or epoch + 1})")
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = history[-1].epoch if history else 0
        best_metric = None if val_loader is None else best_metric

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    final_epoch = history[-1] if history else None
    return AutoencoderTrainingSummary(
        history=history,
        best_epoch=best_epoch,
        best_state=copy.deepcopy(best_state) if best_state is not None else None,
        best_metric=best_metric if val_loader is not None else None,
        best_val_loss=best_val_loss,
        final_train_loss=final_epoch.train_loss if final_epoch is not None else 0.0,
        final_train_mse=final_epoch.train_mse if final_epoch is not None else 0.0,
        final_train_mae=final_epoch.train_mae if final_epoch is not None else 0.0,
        final_train_contractive=final_epoch.train_contractive if final_epoch is not None else 0.0,
    )


def evaluate_autoencoder(
    model: torch.nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    *,
    amp: Optional[bool] = None,
    amp_dtype: str = "float16",
) -> Dict[str, float]:
    """Evaluate reconstruction metrics on a dataloader."""

    if loader is None:
        return {}
    was_training = model.training
    model.eval()
    resolved_amp_dtype = _resolve_amp_dtype(amp_dtype)
    amp_enabled = bool(amp) if amp is not None else (device.type == "cuda")
    loss_accum = 0.0
    mse_accum = 0.0
    mae_accum = 0.0
    samples = 0
    with torch.no_grad():
        for batch in loader:
            noisy, target, mask = _coerce_autoencoder_batch(batch)
            non_blocking = device.type == "cuda"
            noisy = noisy.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            mask = mask.to(device=device, dtype=target.dtype, non_blocking=non_blocking)
            with _autocast_context(device=device, enabled=amp_enabled, dtype=resolved_amp_dtype):
                reconstruction, _ = model(noisy)
                _, mse_val, mae_val = reconstruction_loss(
                    reconstruction,
                    target,
                    mask,
                    mask_focus=0.0,
                )
            batch_size = noisy.size(0)
            loss_accum += mse_val.item() * batch_size
            mse_accum += mse_val.item() * batch_size
            mae_accum += mae_val.item() * batch_size
            samples += batch_size
    if samples == 0:
        return {}
    if was_training:
        model.train()
    return {
        "loss": loss_accum / samples,
        "mse": mse_accum / samples,
        "mae": mae_accum / samples,
    }
