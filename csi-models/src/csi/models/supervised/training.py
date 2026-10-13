"""Supervised training and evaluation helpers."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..metrics import compute_f1_scores
from ..utils.training import TrainingEpochResult, TrainingSummary, maybe_tqdm

__all__ = [
    "train",
    "test",
    "evaluate_classifier",
]


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _should_log_epoch(
    *,
    epoch_index: int,
    total_epochs: int,
    use_tqdm: bool,
    log_every_epochs: Optional[int],
) -> bool:
    epoch_num = epoch_index + 1
    if log_every_epochs is None:
        interval = 1 if use_tqdm else 5
    else:
        interval = max(1, int(log_every_epochs))
    return epoch_num == 1 or epoch_num == total_epochs or (epoch_num % interval == 0)


def train(
    model: torch.nn.Module,
    tensor_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    criterion,
    device: torch.device,
    *,
    use_tqdm: bool = False,
    weight_decay: float = 0.0,
    val_loader: Optional[DataLoader] = None,
    patience: Optional[int] = None,
    min_epochs: int = 0,
    monitor_metric: str = "accuracy",
    monitor_mode: Optional[str] = None,
    min_delta: float = 0.0,
    restore_best: bool = True,
    grad_clip: Optional[float] = None,
    augment_fn: Optional[Callable[[torch.Tensor, Dict], torch.Tensor]] = None,
    aug_cfg: Optional[Dict] = None,
    optimizer_cfg: Optional[Dict] = None,
    scheduler_cfg: Optional[Dict] = None,
    amp: Optional[bool] = None,
    amp_dtype: str = "float16",
    log_every_epochs: Optional[int] = None,
) -> TrainingSummary:
    model = model.to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable model parameters found (all parameters are frozen).")

    optimizer_type = str((optimizer_cfg or {}).get("type", "adam")).lower()
    if optimizer_type in {"adamw", "adam_w"}:
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type in {"adam"}:
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    scheduler = _build_scheduler_from_cfg(optimizer, scheduler_cfg, num_epochs)
    resolved_amp_dtype = _resolve_amp_dtype(amp_dtype)
    amp_enabled = bool(amp) if amp is not None else (device.type == "cuda")
    scaler = _make_grad_scaler(
        enabled=(amp_enabled and device.type == "cuda" and resolved_amp_dtype == torch.float16)
    )

    metric_key = monitor_metric.lower()
    resolved_mode = monitor_mode.lower() if monitor_mode else None
    if resolved_mode not in {"max", "min", None}:
        raise ValueError(f"monitor_mode must be 'max', 'min', or None, got {monitor_mode}")
    if resolved_mode is None:
        resolved_mode = "max" if metric_key in {"accuracy", "acc"} else "min"

    if val_loader is None:
        patience = None

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_accuracy: Optional[float] = None
    best_val_loss: Optional[float] = None
    best_epoch = 0
    history: List[TrainingEpochResult] = []

    best_metric = float("-inf") if resolved_mode == "max" else float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        iterator = maybe_tqdm(tensor_loader, use_tqdm=use_tqdm, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for inputs, labels in iterator:
            if inputs is None or labels is None:
                continue
            if hasattr(inputs, "size") and inputs.size(0) == 0:
                continue
            non_blocking = device.type == "cuda"
            inputs = inputs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking).long()

            if augment_fn is not None:
                inputs = augment_fn(inputs, aug_cfg or {})

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device=device, enabled=amp_enabled, dtype=resolved_amp_dtype):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
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

            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            epoch_samples += batch_size

            if use_tqdm:
                postfix = getattr(iterator, "set_postfix", None)
                if callable(postfix):
                    try:
                        postfix(loss=f"{loss.item():.4f}")
                    except Exception:
                        pass

        train_loss = epoch_loss / max(1, epoch_samples)
        train_accuracy = epoch_correct / max(1, epoch_samples)

        val_accuracy: Optional[float] = None
        val_loss: Optional[float] = None
        if val_loader is not None:
            val_accuracy, val_loss = _evaluate(
                model,
                val_loader,
                criterion,
                device,
                use_tqdm=False,
                desc="Validation",
                amp_enabled=amp_enabled,
                amp_dtype=resolved_amp_dtype,
            )

        history.append(
            TrainingEpochResult(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )
        )

        log_parts = [
            f"epoch={epoch + 1}/{num_epochs}",
            f"train_acc={train_accuracy:.4f}",
            f"train_loss={train_loss:.4f}",
        ]
        if val_accuracy is not None and val_loss is not None:
            log_parts.append(f"val_acc={val_accuracy:.4f}")
            log_parts.append(f"val_loss={val_loss:.4f}")
        if _should_log_epoch(
            epoch_index=epoch,
            total_epochs=num_epochs,
            use_tqdm=use_tqdm,
            log_every_epochs=log_every_epochs,
        ):
            print("[TRAIN] " + " | ".join(log_parts))

        if val_loader is not None and val_accuracy is not None and val_loss is not None:
            if metric_key in {"accuracy", "acc"}:
                monitored_value = val_accuracy
            elif metric_key == "loss":
                monitored_value = val_loss
            else:
                raise ValueError(f"Unsupported monitor_metric: {monitor_metric}")

            if resolved_mode == "max":
                improved = monitored_value > (best_metric + min_delta)
            else:
                improved = monitored_value < (best_metric - min_delta)
            if improved:
                best_metric = monitored_value
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if (
            patience is not None
            and val_loader is not None
            and epoch + 1 >= max(min_epochs, 1)
            and epochs_no_improve >= patience
        ):
            print(f"[EARLY-STOP] Stopping at epoch {epoch + 1} (best epoch {best_epoch or epoch + 1})")
            break

        if scheduler is not None:
            scheduler.step()

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = history[-1].epoch if history else 0
        best_metric = None if val_loader is None else best_metric

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    final_train_accuracy = history[-1].train_accuracy if history else 0.0
    final_train_loss = history[-1].train_loss if history else 0.0

    return TrainingSummary(
        history=history,
        best_epoch=best_epoch,
        best_state=copy.deepcopy(best_state) if best_state is not None else None,
        best_metric=best_metric if val_loader is not None else None,
        best_val_accuracy=best_val_accuracy,
        best_val_loss=best_val_loss,
        final_train_accuracy=final_train_accuracy,
        final_train_loss=final_train_loss,
    )


def _build_scheduler_from_cfg(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: Optional[Dict],
    num_epochs: int,
):
    if not scheduler_cfg:
        return None
    sched_type = str(scheduler_cfg.get("type", "")).lower()
    if sched_type in {"cosine_warmup", "warmup_cosine", "cosine"}:
        warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if sched_type in {"reduce_on_plateau", "reduceonplateau", "rop"}:
        mode = str(scheduler_cfg.get("mode", "min")).lower()
        if mode not in {"min", "max"}:
            raise ValueError(f"Unsupported ReduceLROnPlateau mode '{mode}'. Expected 'min' or 'max'.")
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 5)),
            threshold=float(scheduler_cfg.get("threshold", 1e-4)),
            min_lr=float(scheduler_cfg.get("min_lr", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler type: {sched_type}")


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
    return nullcontext()


def test(
    model: torch.nn.Module,
    tensor_loader: DataLoader,
    criterion,
    device: torch.device,
    *,
    use_tqdm: bool = False,
) -> Tuple[float, float]:
    metrics, _ = evaluate_classifier(
        model,
        tensor_loader,
        criterion,
        device,
        use_tqdm=use_tqdm,
    )
    accuracy = metrics.get("accuracy", 0.0)
    loss = metrics.get("loss", 0.0)
    print(f"validation accuracy:{accuracy:.4f}, loss:{loss:.5f}")
    return accuracy, loss


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    *,
    use_tqdm: bool = False,
    desc: Optional[str] = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        iterator = maybe_tqdm(loader, use_tqdm=use_tqdm, desc=desc or "Evaluating")
        for inputs, labels in iterator:
            if inputs is None or labels is None:
                continue
            if hasattr(inputs, "size") and inputs.size(0) == 0:
                continue
            non_blocking = device.type == "cuda"
            inputs = inputs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking).long()
            with _autocast_context(device=device, enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total_samples += labels.size(0)
    average_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    if was_training:
        model.train()
    return accuracy, average_loss


def evaluate_classifier(
    model: torch.nn.Module,
    loader: Optional[DataLoader],
    criterion,
    device: torch.device,
    *,
    use_tqdm: bool = False,
    collect_probabilities: bool = False,
    collect_predictions: bool = False,
    desc: Optional[str] = None,
    amp: Optional[bool] = None,
    amp_dtype: str = "float16",
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    """Evaluate classification metrics, optionally collecting probabilities."""

    if loader is None:
        return {}, {}
    was_training = model.training
    model.eval()
    resolved_amp_dtype = _resolve_amp_dtype(amp_dtype)
    amp_enabled = bool(amp) if amp is not None else (device.type == "cuda")
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_probs: List[torch.Tensor] = []
    all_prob_labels: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    with torch.no_grad():
        iterator = maybe_tqdm(loader, use_tqdm=use_tqdm, desc=desc or "Evaluating")
        for inputs, labels in iterator:
            if inputs is None or labels is None:
                continue
            if hasattr(inputs, "size") and inputs.size(0) == 0:
                continue
            non_blocking = device.type == "cuda"
            inputs = inputs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking).long()
            with _autocast_context(device=device, enabled=amp_enabled, dtype=resolved_amp_dtype):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            all_preds.append(preds.detach().cpu())
            all_targets.append(labels.detach().cpu())
            if collect_probabilities:
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu())
                all_prob_labels.append(labels.cpu())
    metrics: Dict[str, float] = {}
    extras: Dict[str, torch.Tensor] = {}
    if total_samples > 0:
        metrics["loss"] = total_loss / total_samples
        metrics["accuracy"] = total_correct / total_samples
        if all_preds and all_targets:
            y_pred = torch.cat(all_preds, dim=0).numpy()
            y_true = torch.cat(all_targets, dim=0).numpy()
            f1_scores = compute_f1_scores(y_true, y_pred)
            metrics["f1_macro"] = float(f1_scores["macro"])
            metrics["f1_weighted"] = float(f1_scores["weighted"])
            metrics["f1_micro"] = float(f1_scores["micro"])
            if collect_predictions:
                extras["preds"] = torch.as_tensor(y_pred)
                extras["targets"] = torch.as_tensor(y_true)
        if collect_probabilities and all_probs:
            extras["probabilities"] = torch.cat(all_probs, dim=0)
            extras["labels"] = torch.cat(all_prob_labels, dim=0)
    if was_training:
        model.train()
    return metrics, extras
