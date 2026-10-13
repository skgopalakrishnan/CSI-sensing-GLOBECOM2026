"""Two-stage autoencoder trials for NTU and CSI-Bench.

This runner supports:
1. Reconstruction pretraining of a shared encoder.
2. Classifier fine-tuning with AE heads (TCN / GRU / SSM).

The same entry point can execute either stage independently or both in sequence.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from csi.data import get_experiment_loaders
from csi.data.utils import filter_none_collate, load_csi_bench_supervised
from csi.models.autoencoders import build_autoencoder_from_config, evaluate_autoencoder, train_autoencoder_epoch
from csi.models.model_factory import build_model, infer_model_technique
from csi.models.supervised import evaluate_classifier
from csi.models.utils.experiment import load_config, resolve_config_path_from_arg, sanitize_token
from csi.models.utils.gpu_utils import _resolve_device
from csi.models.utils.results import (
    prepare_results_dir,
    save_config,
    save_results_rows,
    save_run_meta,
)
from csi.models.utils.seeding import configure_from_env_and_config
from csi.models.utils.training import (
    build_scheduler,
    count_trainable_parameters,
    set_requires_grad,
    train_classifier_epoch,
)
from ..common.shared_utils import (
    collect_csi_bench_task_metadata,
    infer_input_shape_from_loader,
    move_incomplete_run,
    normalize_csi_bench_task_name,
    parse_bool,
    parse_splits,
    parse_token_list,
    print_training_start,
    save_training_metadata,
)
from ..common.train_models_utils import (
    DEFAULT_CSI_BENCH_SUBCARRIERS,
    DEFAULT_CSI_BENCH_TIMESTEPS,
    as_list,
    configure_torch_performance,
    evaluate_splits,
    infer_ntu_shape,
    resolve_ntu_num_classes,
    resolve_project_path,
    resolve_supervised_models,
    save_model_weights,
)


def _resolve_autoencoder_classifier_models(requested: Sequence[str]) -> Sequence[str]:
    model_names = resolve_supervised_models(requested)
    invalid: list[str] = []
    for model_name in model_names:
        if infer_model_technique(model_name) != "autoencoder_classifier":
            invalid.append(model_name)
    if invalid:
        raise ValueError(
            f"ae_trial_training only supports AE classifier heads. Unsupported entries: {invalid}"
        )
    return model_names


def _resolve_stages(value: Any) -> List[str]:
    if value is None:
        tokens = ["both"]
    elif isinstance(value, str):
        tokens = [token.strip().lower() for token in value.split(",") if token.strip()]
    elif isinstance(value, (list, tuple)):
        tokens = [str(token).strip().lower() for token in value if str(token).strip()]
    else:
        tokens = ["both"]

    stages: List[str] = []
    for token in tokens:
        key = token.replace("-", "_")
        if key in {"both", "all"}:
            for stage in ("pretrain", "classify"):
                if stage not in stages:
                    stages.append(stage)
            continue
        if key in {"pretrain", "reconstruction", "autoencoder"}:
            if "pretrain" not in stages:
                stages.append("pretrain")
            continue
        if key in {"classify", "classifier", "transfer", "finetune", "fine_tune"}:
            if "classify" not in stages:
                stages.append("classify")
            continue
        raise ValueError(f"Unsupported AE stage '{token}'. Expected pretrain, classify, or both.")
    if not stages:
        stages = ["pretrain", "classify"]
    return stages


def _head_type_from_model_name(model_name: str) -> str:
    key = sanitize_token(model_name).lower().replace("-", "_")
    if key.endswith("tcn"):
        return "tcn"
    if key.endswith("gru"):
        return "gru"
    if key.endswith("ssm"):
        return "ssm"
    raise ValueError(f"Unable to infer AE head type from model name '{model_name}'")


def _encoder_model_config(model_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    encoder_cfg = model_cfg.get("encoder")
    if not isinstance(encoder_cfg, Mapping):
        return {"type": "tiny_f_stride"}
    encoder_type = str(encoder_cfg.get("type", "tiny_f_stride"))
    config = encoder_cfg.get("config")
    if isinstance(config, Mapping):
        resolved = dict(config)
    else:
        resolved = {key: value for key, value in encoder_cfg.items() if key not in {"type", "config"}}
    resolved["type"] = str(resolved.get("type", encoder_type))
    return resolved


def _encoder_type_from_cfg(model_cfg: Mapping[str, Any]) -> str:
    encoder_cfg = model_cfg.get("encoder")
    if isinstance(encoder_cfg, Mapping):
        return str(encoder_cfg.get("type", "tiny_f_stride"))
    return "tiny_f_stride"


def _head_metadata_from_cfg(model_cfg: Mapping[str, Any], head_type: str) -> Dict[str, Any]:
    heads_cfg = model_cfg.get("heads")
    if not isinstance(heads_cfg, Mapping):
        return {"type": head_type}
    selected = heads_cfg.get(head_type)
    if isinstance(selected, Mapping):
        payload = dict(selected)
    else:
        payload = {}
    payload["type"] = str(payload.get("type", head_type))
    return payload


def _build_encoder_arch_metadata(
    *,
    dataset_name: str,
    encoder_name: str,
    encoder_model_cfg: Mapping[str, Any],
    channels: int,
    freq_bins: int,
    time_steps: int,
    task_name: Optional[str] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "technique": "autoencoder",
        "dataset": dataset_name,
        "encoder_name": encoder_name,
        "channels": int(channels),
        "freq_bins": int(freq_bins),
        "time_steps": int(time_steps),
        "input_shape": [int(channels), int(freq_bins), int(time_steps)],
        "encoder": {
            "type": str(encoder_model_cfg.get("type", encoder_name)),
            "config": dict(encoder_model_cfg),
        },
    }
    if task_name:
        metadata["task_name"] = task_name
    return metadata


def _build_classifier_arch_metadata(
    *,
    dataset_name: str,
    model_name: str,
    model_cfg: Mapping[str, Any],
    channels: int,
    freq_bins: int,
    time_steps: int,
    num_classes: int,
    task_name: Optional[str] = None,
) -> Dict[str, Any]:
    head_type = _head_type_from_model_name(model_name)
    metadata: Dict[str, Any] = {
        "technique": "autoencoder_classifier",
        "dataset": dataset_name,
        "num_classes": int(num_classes),
        "channels": int(channels),
        "freq_bins": int(freq_bins),
        "time_steps": int(time_steps),
        "input_shape": [int(channels), int(freq_bins), int(time_steps)],
        "head": _head_metadata_from_cfg(model_cfg, head_type),
        "encoder": {
            "type": _encoder_type_from_cfg(model_cfg),
            "config": _encoder_model_config(model_cfg),
        },
    }
    if task_name:
        metadata["task_name"] = task_name
    return metadata


def _extract_state_dict_from_payload(payload: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint payload must be a mapping, got {type(payload)!r}")
    if "model_state_dict" in payload and isinstance(payload["model_state_dict"], Mapping):
        return payload["model_state_dict"]  # type: ignore[return-value]
    if "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
        return payload["state_dict"]  # type: ignore[return-value]
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload  # type: ignore[return-value]
    raise ValueError("Unsupported checkpoint format (expected state_dict or model_state_dict).")


def _load_state_dict_file(path: str) -> Mapping[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return _extract_state_dict_from_payload(payload)


def _metadata_path_for(checkpoint_path: Path) -> Path:
    return Path(f"{checkpoint_path}.meta.json")


def _write_checkpoint_metadata(checkpoint_path: Path, payload: Mapping[str, Any]) -> None:
    meta_path = _metadata_path_for(checkpoint_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)


def _history_rows_autoencoder(history: Sequence[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in history:
        rows.append(
            {
                "epoch": int(item.get("epoch", 0)),
                "train_loss": float(item.get("train_loss", 0.0)),
                "train_mse": float(item.get("train_mse", 0.0)),
                "train_mae": float(item.get("train_mae", 0.0)),
                "train_contractive": float(item.get("train_contractive", 0.0)),
                "val_loss": float(item["val_loss"]) if item.get("val_loss") is not None else None,
                "val_mse": float(item["val_mse"]) if item.get("val_mse") is not None else None,
                "val_mae": float(item["val_mae"]) if item.get("val_mae") is not None else None,
                "mask_focus": float(item.get("mask_focus", 0.0)),
            }
        )
    return rows


def _evaluate_autoencoder_splits(
    *,
    model: nn.Module,
    device: torch.device,
    split_loaders: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split_name, loader in split_loaders.items():
        if loader is None:
            continue
        metrics = evaluate_autoencoder(model, loader, device)
        rows.append(
            {
                "split": split_name,
                "loss": float(metrics.get("loss", 0.0)),
                "mse": float(metrics.get("mse", 0.0)),
                "mae": float(metrics.get("mae", 0.0)),
            }
        )
    return rows


def _build_optimizer(
    params: Any,
    *,
    learning_rate: float,
    weight_decay: float,
    optimizer_cfg: Optional[Mapping[str, Any]],
):
    optimizer_type = str((optimizer_cfg or {}).get("type", "adamw")).lower()
    if optimizer_type in {"adamw", "adam_w"}:
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_type == "adam":
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def _monitor_improved(
    *,
    monitored_value: float,
    best_metric: float,
    mode: str,
    min_delta: float,
) -> bool:
    if mode == "max":
        return monitored_value > (best_metric + min_delta)
    return monitored_value < (best_metric - min_delta)


def _encoder_export_dir(
    *,
    domain: str,
    datasets_root: str,
    weights_dir_name: str,
    dataset_name: str,
    task_name: Optional[str],
) -> Path:
    root = resolve_project_path(datasets_root)
    if domain == "ntu":
        return root / dataset_name / weights_dir_name
    csi_root = root if root.name == "CSI-Bench" else (root / "CSI-Bench")
    if not task_name:
        raise ValueError("task_name is required for CSI-Bench encoder exports.")
    return csi_root / weights_dir_name / task_name


def _export_encoder_checkpoint(
    *,
    domain: str,
    datasets_root: str,
    weights_dir_name: str,
    dataset_name: str,
    task_name: Optional[str],
    encoder_name: str,
    source_checkpoint: str,
    source_out_dir: str,
    config_path: Optional[str],
    run_timestamp: Optional[str],
    arch_metadata: Mapping[str, Any],
) -> str:
    export_dir = _encoder_export_dir(
        domain=domain,
        datasets_root=datasets_root,
        weights_dir_name=weights_dir_name,
        dataset_name=dataset_name,
        task_name=task_name,
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    stem_prefix = dataset_name if domain == "ntu" else "CSI-Bench"
    stem = f"{stem_prefix}_{encoder_name}"
    ts_token = sanitize_token(run_timestamp or "") or sanitize_token(Path(source_out_dir).name) or "run"
    export_path = export_dir / f"{stem}__{ts_token}.pt"
    latest_path = export_dir / f"{stem}.pt"

    shutil.copy2(source_checkpoint, export_path)
    shutil.copy2(source_checkpoint, latest_path)
    common_payload = {
        "stage": "ae_trial_pretrain_export",
        "domain": domain,
        "dataset": dataset_name,
        "task_name": task_name,
        "encoder_name": encoder_name,
        "run_timestamp": run_timestamp,
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "source_run_dir": str(Path(source_out_dir).resolve()),
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        **dict(arch_metadata),
    }
    _write_checkpoint_metadata(export_path, {**common_payload, "variant": "timestamped", "latest_path": str(latest_path)})
    _write_checkpoint_metadata(latest_path, {**common_payload, "variant": "latest_alias", "timestamped_path": str(export_path)})
    return str(export_path)


def _export_classifier_checkpoint(
    *,
    domain: str,
    datasets_root: str,
    weights_dir_name: str,
    dataset_name: str,
    task_name: Optional[str],
    model_name: str,
    source_checkpoint: str,
    source_out_dir: str,
    config_path: Optional[str],
    run_timestamp: Optional[str],
    arch_metadata: Mapping[str, Any],
) -> str:
    root = resolve_project_path(datasets_root)
    if domain == "ntu":
        export_dir = root / dataset_name / weights_dir_name
        stem = f"{dataset_name}_{model_name}"
    else:
        csi_root = root if root.name == "CSI-Bench" else (root / "CSI-Bench")
        if not task_name:
            raise ValueError("task_name is required for CSI-Bench classifier exports.")
        export_dir = csi_root / weights_dir_name / task_name
        stem = f"CSI-Bench_{model_name}"
    export_dir.mkdir(parents=True, exist_ok=True)
    ts_token = sanitize_token(run_timestamp or "") or sanitize_token(Path(source_out_dir).name) or "run"
    export_path = export_dir / f"{stem}__{ts_token}.pt"
    latest_path = export_dir / f"{stem}.pt"

    shutil.copy2(source_checkpoint, export_path)
    shutil.copy2(source_checkpoint, latest_path)
    common_payload = {
        "stage": "ae_trial_classifier_export",
        "domain": domain,
        "dataset": dataset_name,
        "task_name": task_name,
        "model": model_name,
        "run_timestamp": run_timestamp,
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "source_run_dir": str(Path(source_out_dir).resolve()),
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        **dict(arch_metadata),
    }
    _write_checkpoint_metadata(export_path, {**common_payload, "variant": "timestamped", "latest_path": str(latest_path)})
    _write_checkpoint_metadata(latest_path, {**common_payload, "variant": "latest_alias", "timestamped_path": str(export_path)})
    return str(export_path)


def _resolve_encoder_checkpoint(
    *,
    domain: str,
    datasets_root: str,
    weights_dir_name: str,
    dataset_name: str,
    task_name: Optional[str],
    encoder_name: str,
    explicit_checkpoint: Optional[str],
) -> str:
    if explicit_checkpoint:
        path = resolve_project_path(str(explicit_checkpoint))
        if not path.exists():
            raise FileNotFoundError(f"Explicit encoder checkpoint does not exist: {path}")
        return str(path)

    export_dir = _encoder_export_dir(
        domain=domain,
        datasets_root=datasets_root,
        weights_dir_name=weights_dir_name,
        dataset_name=dataset_name,
        task_name=task_name,
    )
    stem_prefix = dataset_name if domain == "ntu" else "CSI-Bench"
    alias_path = export_dir / f"{stem_prefix}_{encoder_name}.pt"
    if alias_path.exists():
        return str(alias_path)

    if not export_dir.exists():
        raise FileNotFoundError(f"Encoder checkpoint directory does not exist: {export_dir}")

    token = sanitize_token(encoder_name).lower()
    candidates = [path for path in export_dir.glob("*.pt") if token in sanitize_token(path.stem).lower()]
    if not candidates:
        raise FileNotFoundError(
            f"No encoder checkpoint found in {export_dir} for encoder '{encoder_name}'."
        )
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def _lookup_checkpoint_override(mapping: Any, *keys: str) -> Optional[str]:
    current = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    if current is None:
        return None
    return str(current)


def _train_autoencoder_model(
    *,
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    pretrain_cfg: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    learning_rate = float(pretrain_cfg.get("learning_rate", 1e-3))
    weight_decay = float(pretrain_cfg.get("weight_decay", 0.0))
    epochs = int(pretrain_cfg.get("epochs", 100))
    grad_clip_raw = pretrain_cfg.get("grad_clip")
    grad_clip = float(grad_clip_raw) if grad_clip_raw is not None else None
    mask_focus_start = float(pretrain_cfg.get("mask_focus", 1.0))
    mask_focus_final = pretrain_cfg.get("mask_focus_final")
    mask_focus_final = float(mask_focus_final) if mask_focus_final is not None else None
    mask_focus_decay_portion = float(pretrain_cfg.get("mask_focus_decay_portion", 0.0))
    contractive_lambda = float(pretrain_cfg.get("contractive_lambda", 0.0))
    contractive_interval = int(pretrain_cfg.get("contractive_interval", 1))
    contractive_sample_ratio = float(pretrain_cfg.get("contractive_sample_ratio", 1.0))
    progress_bar = parse_bool(pretrain_cfg.get("progress_bar"), default=False)
    patience_raw = pretrain_cfg.get("patience", pretrain_cfg.get("early_stopping_patience"))
    patience = int(patience_raw) if patience_raw is not None else None
    min_epochs = int(pretrain_cfg.get("min_epochs", 1))
    monitor_metric = str(pretrain_cfg.get("monitor_metric", "loss")).lower()
    min_delta = float(pretrain_cfg.get("min_delta", 0.0))
    optimizer_cfg = pretrain_cfg.get("optimizer")
    scheduler_cfg = pretrain_cfg.get("scheduler")
    amp = pretrain_cfg.get("amp")
    amp_dtype = str(pretrain_cfg.get("amp_dtype", "float16"))

    model = model.to(device)
    optimizer = _build_optimizer(
        [param for param in model.parameters() if param.requires_grad],
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_cfg=optimizer_cfg if isinstance(optimizer_cfg, Mapping) else None,
    )
    scheduler = build_scheduler(optimizer, scheduler_cfg if isinstance(scheduler_cfg, Mapping) else None, epochs)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metric = float("inf")
    best_epoch = 0
    best_val_loss: Optional[float] = None
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    decay_steps = 0
    if mask_focus_final is not None and mask_focus_decay_portion > 0.0:
        decay_steps = max(1, int(epochs * mask_focus_decay_portion))

    for epoch_idx in range(epochs):
        if decay_steps > 0 and mask_focus_final is not None and epoch_idx < decay_steps:
            progress = float(epoch_idx) / float(max(1, decay_steps - 1))
            mask_focus_current = mask_focus_start + (mask_focus_final - mask_focus_start) * progress
        elif mask_focus_final is not None and decay_steps > 0:
            mask_focus_current = mask_focus_final
        else:
            mask_focus_current = mask_focus_start

        train_stats = train_autoencoder_epoch(
            model,
            train_loader,
            optimizer,
            device,
            mask_focus=mask_focus_current,
            grad_clip=grad_clip,
            contractive_lambda=contractive_lambda,
            use_tqdm=progress_bar,
            contractive_interval=contractive_interval,
            contractive_sample_ratio=contractive_sample_ratio,
            amp=amp,
            amp_dtype=amp_dtype,
        )
        val_metrics = evaluate_autoencoder(model, val_loader, device, amp=amp, amp_dtype=amp_dtype)
        row = {
            "epoch": epoch_idx + 1,
            "train_loss": train_stats.loss,
            "train_mse": train_stats.mse,
            "train_mae": train_stats.mae,
            "train_contractive": train_stats.contractive,
            "val_loss": val_metrics.get("loss"),
            "val_mse": val_metrics.get("mse"),
            "val_mae": val_metrics.get("mae"),
            "mask_focus": mask_focus_current,
        }
        history.append(row)

        log_parts = [
            f"epoch={epoch_idx + 1}/{epochs}",
            f"train_loss={train_stats.loss:.4f}",
            f"train_mse={train_stats.mse:.4f}",
            f"train_mae={train_stats.mae:.4f}",
            f"mask_focus={mask_focus_current:.4f}",
        ]
        if train_stats.contractive > 0.0:
            log_parts.append(f"train_contractive={train_stats.contractive:.4f}")
        if val_metrics:
            log_parts.append(f"val_loss={val_metrics.get('loss', 0.0):.4f}")
            log_parts.append(f"val_mse={val_metrics.get('mse', 0.0):.4f}")
            log_parts.append(f"val_mae={val_metrics.get('mae', 0.0):.4f}")
        print("[AE-PRETRAIN] " + " | ".join(log_parts))

        monitored_value = float(val_metrics.get(monitor_metric, train_stats.loss))
        improved = _monitor_improved(
            monitored_value=monitored_value,
            best_metric=best_metric,
            mode="min",
            min_delta=min_delta,
        )
        if improved:
            best_metric = monitored_value
            best_epoch = epoch_idx + 1
            best_state = copy.deepcopy(model.state_dict())
            best_val_loss = float(val_metrics.get("loss", monitored_value)) if val_metrics else monitored_value
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_target = float(val_metrics.get(monitor_metric, train_stats.loss))
                scheduler.step(scheduler_target)
            else:
                scheduler.step()

        if (
            patience is not None
            and val_loader is not None
            and epoch_idx + 1 >= max(1, min_epochs)
            and epochs_no_improve >= patience
        ):
            print(
                f"[AE-EARLY-STOP] Stopping at epoch {epoch_idx + 1} "
                f"(best epoch {best_epoch or epoch_idx + 1})"
            )
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(history)
        best_metric = history[-1]["train_loss"] if history else 0.0

    model.load_state_dict(best_state)
    summary = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_val_loss": best_val_loss,
        "final_train_loss": float(history[-1]["train_loss"]) if history else 0.0,
        "final_train_mse": float(history[-1]["train_mse"]) if history else 0.0,
        "final_train_mae": float(history[-1]["train_mae"]) if history else 0.0,
        "final_train_contractive": float(history[-1]["train_contractive"]) if history else 0.0,
    }
    return history, summary


def _train_classifier_with_phases(
    *,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    transfer_cfg: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    progress_bar = parse_bool(transfer_cfg.get("progress_bar"), default=False)
    grad_clip_raw = transfer_cfg.get("grad_clip")
    grad_clip = float(grad_clip_raw) if grad_clip_raw is not None else None
    label_smoothing = float(transfer_cfg.get("label_smoothing", 0.0))
    patience_raw = transfer_cfg.get("patience")
    patience = int(patience_raw) if patience_raw is not None else None
    min_epochs = int(transfer_cfg.get("min_epochs", 0))
    monitor_metric = str(transfer_cfg.get("monitor_metric", "accuracy")).lower()
    if monitor_metric not in {"accuracy", "loss"}:
        raise ValueError("transfer.monitor_metric must be 'accuracy' or 'loss'")
    monitor_mode = "max" if monitor_metric == "accuracy" else "min"
    min_delta = float(transfer_cfg.get("min_delta", 0.0))
    optimizer_cfg = transfer_cfg.get("optimizer")
    amp = transfer_cfg.get("amp")
    amp_dtype = str(transfer_cfg.get("amp_dtype", "float16"))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metric = float("-inf") if monitor_mode == "max" else float("inf")
    best_epoch = 0
    best_val_metrics: Optional[Dict[str, float]] = None
    epochs_no_improve = 0
    global_epoch = 0
    history_rows: List[Dict[str, Any]] = []

    phases: Sequence[Tuple[str, Mapping[str, Any]]] = [
        ("phase_a", dict(transfer_cfg.get("phase_a", {}) or {})),
        ("phase_b", dict(transfer_cfg.get("phase_b", {}) or {})),
    ]

    for phase_name, phase_cfg in phases:
        if not phase_cfg:
            continue
        freeze_encoder = parse_bool(phase_cfg.get("freeze_encoder"), default=(phase_name == "phase_a"))
        epochs = int(phase_cfg.get("epochs", 0))
        if epochs <= 0:
            continue

        head_lr = float(phase_cfg.get("head_lr", phase_cfg.get("lr", 1e-3)))
        weight_decay = float(phase_cfg.get("weight_decay", transfer_cfg.get("weight_decay", 0.0)))
        encoder_lr_value: Optional[float] = None

        if freeze_encoder:
            set_requires_grad(model.encoder, False)
            set_requires_grad(model.head, True)
            trainable_params = [param for param in model.head.parameters() if param.requires_grad]
            if not trainable_params:
                raise RuntimeError("No trainable head parameters found during phase A.")
            optimizer = _build_optimizer(
                trainable_params,
                learning_rate=head_lr,
                weight_decay=weight_decay,
                optimizer_cfg=optimizer_cfg if isinstance(optimizer_cfg, Mapping) else None,
            )
        else:
            set_requires_grad(model.encoder, True)
            set_requires_grad(model.head, True)
            encoder_lr_value = float(phase_cfg.get("encoder_lr", head_lr * 0.1))
            head_params = [param for param in model.head.parameters() if param.requires_grad]
            encoder_params = [param for param in model.encoder.parameters() if param.requires_grad]
            optimizer = _build_optimizer(
                [
                    {"params": head_params, "lr": head_lr},
                    {"params": encoder_params, "lr": encoder_lr_value},
                ],
                learning_rate=head_lr,
                weight_decay=weight_decay,
                optimizer_cfg=optimizer_cfg if isinstance(optimizer_cfg, Mapping) else None,
            )

        scheduler_cfg = phase_cfg.get("scheduler")
        scheduler = build_scheduler(optimizer, scheduler_cfg if isinstance(scheduler_cfg, Mapping) else None, epochs)

        print(
            f"[AE-TRANSFER] phase={phase_name} epochs={epochs} freeze_encoder={freeze_encoder} "
            f"head_lr={head_lr:.2e} encoder_lr="
            f"{encoder_lr_value:.2e}" if encoder_lr_value is not None else
            f"[AE-TRANSFER] phase={phase_name} epochs={epochs} freeze_encoder={freeze_encoder} head_lr={head_lr:.2e}"
        )

        for _ in range(epochs):
            global_epoch += 1
            train_metrics = train_classifier_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                grad_clip=grad_clip,
                use_tqdm=progress_bar,
                freeze_encoder=freeze_encoder,
                progress_desc=f"{phase_name}:{global_epoch}",
                amp=amp,
                amp_dtype=amp_dtype,
            )
            val_metrics: Dict[str, float] = {}
            if val_loader is not None:
                evaluated, _ = evaluate_classifier(
                    model,
                    val_loader,
                    criterion,
                    device,
                    use_tqdm=False,
                    collect_predictions=False,
                    collect_probabilities=False,
                    amp=amp,
                    amp_dtype=amp_dtype,
                )
                val_metrics.update(evaluated)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler_target = val_metrics.get(monitor_metric, train_metrics[monitor_metric])
                    scheduler.step(float(scheduler_target))
                else:
                    scheduler.step()

            head_lr_current = float(optimizer.param_groups[0]["lr"])
            encoder_lr_current = (
                float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else None
            )
            row: Dict[str, Any] = {
                "epoch": global_epoch,
                "phase": phase_name,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_loss": float(val_metrics["loss"]) if "loss" in val_metrics else None,
                "val_accuracy": float(val_metrics["accuracy"]) if "accuracy" in val_metrics else None,
                "head_lr": head_lr_current,
                "encoder_lr": encoder_lr_current,
            }
            history_rows.append(row)

            log_parts = [
                f"epoch={global_epoch}",
                f"phase={phase_name}",
                f"train_loss={train_metrics['loss']:.4f}",
                f"train_acc={train_metrics['accuracy']:.4f}",
                f"head_lr={head_lr_current:.2e}",
            ]
            if encoder_lr_current is not None:
                log_parts.append(f"enc_lr={encoder_lr_current:.2e}")
            if val_metrics:
                log_parts.append(f"val_loss={val_metrics.get('loss', 0.0):.4f}")
                log_parts.append(f"val_acc={val_metrics.get('accuracy', 0.0):.4f}")
            print("[AE-TRANSFER] " + " | ".join(log_parts))

            monitored_value = float(val_metrics.get(monitor_metric, train_metrics[monitor_metric]))
            improved = _monitor_improved(
                monitored_value=monitored_value,
                best_metric=best_metric,
                mode=monitor_mode,
                min_delta=min_delta,
            )
            if improved:
                best_metric = monitored_value
                best_epoch = global_epoch
                best_state = copy.deepcopy(model.state_dict())
                best_val_metrics = dict(val_metrics) if val_metrics else dict(train_metrics)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (
                patience is not None
                and global_epoch >= max(1, min_epochs)
                and epochs_no_improve >= patience
            ):
                print(
                    f"[AE-EARLY-STOP] Stopping classifier training at epoch {global_epoch} "
                    f"(best epoch {best_epoch or global_epoch})"
                )
                break
        else:
            continue
        break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = global_epoch
        best_val_metrics = None

    model.load_state_dict(best_state)
    final_eval_rows = evaluate_splits(
        model=model,
        criterion=criterion,
        device=device,
        use_tqdm=False,
        split_loaders={"val": val_loader, "test": test_loader},
        amp=amp,
        amp_dtype=amp_dtype,
    )
    summary = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_val_metrics": best_val_metrics,
        "final_eval_rows": final_eval_rows,
    }
    return history_rows, summary


def _run_ntu(
    *,
    cfg: Mapping[str, Any],
    config_path: Optional[str],
    experiment_name: str,
    stages: Sequence[str],
    model_names: Sequence[str],
    model_cfg: Mapping[str, Any],
    save_encoder_weights: bool,
    save_classifier_weights: bool,
    encoder_weights_dir_name: str,
    classifier_weights_dir_name: str,
    device: torch.device,
    data_seed: int,
) -> None:
    ntu_cfg = dict(cfg.get("ntu", {}) or {})
    datasets = as_list(ntu_cfg.get("datasets") or ntu_cfg.get("dataset"), default=["NTU-Fi_HAR", "NTU-Fi-HumanID"])
    root = str(ntu_cfg.get("root", cfg.get("common", {}).get("root", "datasets")))
    timesteps = int(ntu_cfg.get("timesteps", 250))
    batch_sizes = ntu_cfg.get("batch_sizes")
    base_normalisation = str(ntu_cfg.get("base_normalisation", "none"))
    canon_normalisation = str(ntu_cfg.get("canon_normalisation", "per_subcarrier"))
    variant = str(ntu_cfg.get("variant", "raw"))
    smoothing = ntu_cfg.get("smoothing")
    num_workers_raw = ntu_cfg.get("num_workers")
    num_workers = int(num_workers_raw) if num_workers_raw is not None else None
    pin_memory_raw = ntu_cfg.get("pin_memory")
    pin_memory = None if pin_memory_raw is None else parse_bool(pin_memory_raw, default=(device.type == "cuda"))
    persistent_workers_raw = ntu_cfg.get("persistent_workers")
    persistent_workers = None if persistent_workers_raw is None else parse_bool(persistent_workers_raw, default=True)
    prefetch_factor_raw = ntu_cfg.get("prefetch_factor")
    prefetch_factor = int(prefetch_factor_raw) if prefetch_factor_raw is not None else None
    augment_train = dict(ntu_cfg.get("augment", {}) or {})
    augment_eval = dict(ntu_cfg.get("augment_eval", augment_train) or augment_train)
    ntu_seed = int(ntu_cfg.get("seed", data_seed))

    pretrain_cfg = dict(cfg.get("training", {}).get("pretrain", {}) or {})
    transfer_cfg = dict(cfg.get("training", {}).get("transfer", {}) or {})
    encoder_name = _encoder_type_from_cfg(model_cfg)
    encoder_checkpoints = transfer_cfg.get("encoder_checkpoints")
    explicit_encoder_checkpoint = transfer_cfg.get("encoder_checkpoint")

    for dataset_name in datasets:
        encoder_checkpoint_path: Optional[str] = None

        if "pretrain" in stages:
            out_dir, ts, cfg_name = prepare_results_dir(
                experiment_name=experiment_name,
                cfg_path=config_path,
                relative_parts=["ntu", dataset_name, "pretrain", sanitize_token(encoder_name)],
            )
            completed = False
            try:
                save_run_meta(
                    out_dir,
                    {
                        "domain": "ntu",
                        "dataset": dataset_name,
                        "stage": "pretrain",
                        "encoder": encoder_name,
                        "timestamp": ts,
                        "config_name": cfg_name,
                        "root": root,
                        "timesteps": timesteps,
                        "device": str(device),
                    },
                )
                save_config(out_dir, dict(cfg))
                print_training_start(
                    f"AE pretraining NTU ({dataset_name})",
                    {
                        "output_dir": out_dir,
                        "dataset": dataset_name,
                        "encoder": encoder_name,
                        "device": str(device),
                    },
                )

                bundle = get_experiment_loaders(
                    dataset_name,
                    root,
                    task="reconstruction",
                    timesteps=timesteps,
                    batch_sizes=batch_sizes,
                    device=device,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    prefetch_factor=prefetch_factor,
                    base_normalisation=base_normalisation,
                    canon_normalisation=canon_normalisation,
                    variant=variant,
                    smoothing=smoothing,
                    augment_train=augment_train,
                    augment_eval=augment_eval,
                    seed=ntu_seed,
                )
                train_loader = bundle.train_loader
                val_loader = bundle.val_loader
                test_loader = bundle.test_loader
                if train_loader is None:
                    raise RuntimeError(f"Train loader missing for dataset {dataset_name}")

                shape = infer_ntu_shape(bundle.info or {}, train_loader)
                if shape is None:
                    raise RuntimeError(f"Unable to infer NTU input shape for {dataset_name}")
                channels, freq_bins, time_steps = shape
                encoder_model_cfg = _encoder_model_config(model_cfg)
                model = build_autoencoder_from_config(encoder_model_cfg, sample_shape=shape, device=device).to(device)
                print(f"[MODEL] AE params={count_trainable_parameters(model):,}")

                history_rows, summary = _train_autoencoder_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    pretrain_cfg=pretrain_cfg,
                )
                arch_metadata = _build_encoder_arch_metadata(
                    dataset_name=dataset_name,
                    encoder_name=encoder_name,
                    encoder_model_cfg=encoder_model_cfg,
                    channels=channels,
                    freq_bins=freq_bins,
                    time_steps=time_steps,
                )
                save_results_rows(out_dir, _history_rows_autoencoder(history_rows), base_name="train_history")
                save_results_rows(
                    out_dir,
                    _evaluate_autoencoder_splits(
                        model=model,
                        device=device,
                        split_loaders={"val": val_loader, "test": test_loader},
                    ),
                    base_name="summary_rows",
                )

                local_checkpoint = save_model_weights(out_dir=out_dir, filename_stem=encoder_name, model=model)
                _write_checkpoint_metadata(
                    Path(local_checkpoint),
                    {
                        "stage": "ae_trial_pretrain_local_checkpoint",
                        "domain": "ntu",
                        "dataset": dataset_name,
                        "encoder_name": encoder_name,
                        "run_timestamp": ts,
                        "config_path": str(Path(config_path).resolve()) if config_path else None,
                        **arch_metadata,
                    },
                )
                encoder_checkpoint_path = local_checkpoint
                exported_checkpoint = None
                if save_encoder_weights:
                    exported_checkpoint = _export_encoder_checkpoint(
                        domain="ntu",
                        datasets_root=root,
                        weights_dir_name=encoder_weights_dir_name,
                        dataset_name=dataset_name,
                        task_name=None,
                        encoder_name=encoder_name,
                        source_checkpoint=local_checkpoint,
                        source_out_dir=out_dir,
                        config_path=config_path,
                        run_timestamp=ts,
                        arch_metadata=arch_metadata,
                    )

                save_training_metadata(
                    out_dir,
                    {
                        "stage": "ae_pretrain_ntu",
                        "dataset": dataset_name,
                        "encoder_name": encoder_name,
                        "root": root,
                        "device": str(device),
                        "input_shape": [channels, freq_bins, time_steps],
                        "weights_path": local_checkpoint,
                        "dataset_export_weights_path": exported_checkpoint,
                        "dataset_export_weights_dir_name": encoder_weights_dir_name,
                        "arch_metadata": arch_metadata,
                        "summary": summary,
                    },
                )
                completed = True
                print(f"[RUN] Completed AE pretraining: {out_dir}")
            except BaseException as exc:
                print(f"[ERROR] NTU AE pretraining failed for {dataset_name}: {exc}")
                traceback.print_exc()
                raise
            finally:
                if not completed:
                    target_dir = move_incomplete_run(out_dir)
                    if target_dir:
                        print(f"[INFO] Moved incomplete run to: {target_dir}")

        if "classify" not in stages:
            continue

        bundle = get_experiment_loaders(
            dataset_name,
            root,
            task="classification",
            timesteps=timesteps,
            batch_sizes=batch_sizes,
            device=device,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            base_normalisation=base_normalisation,
            canon_normalisation=canon_normalisation,
            variant=variant,
            smoothing=smoothing,
            augment_train=augment_train,
            augment_eval=augment_eval,
            seed=ntu_seed,
        )
        train_loader = bundle.train_loader
        val_loader = bundle.val_loader
        test_loader = bundle.test_loader
        if train_loader is None:
            raise RuntimeError(f"Train loader missing for dataset {dataset_name}")

        bundle_info = bundle.info or {}
        shape = infer_ntu_shape(bundle_info, train_loader)
        if shape is None:
            raise RuntimeError(f"Unable to infer NTU input shape for {dataset_name}")
        channels, freq_bins, time_steps = shape
        num_classes = resolve_ntu_num_classes(bundle_info, train_loader)

        explicit_encoder = (
            str(explicit_encoder_checkpoint)
            if explicit_encoder_checkpoint
            else _lookup_checkpoint_override(encoder_checkpoints, dataset_name)
        )
        resolved_encoder_checkpoint = encoder_checkpoint_path or _resolve_encoder_checkpoint(
            domain="ntu",
            datasets_root=root,
            weights_dir_name=encoder_weights_dir_name,
            dataset_name=dataset_name,
            task_name=None,
            encoder_name=encoder_name,
            explicit_checkpoint=explicit_encoder,
        )

        for model_name in model_names:
            out_dir, ts, cfg_name = prepare_results_dir(
                experiment_name=experiment_name,
                cfg_path=config_path,
                relative_parts=["ntu", dataset_name, sanitize_token(model_name)],
            )
            completed = False
            try:
                save_run_meta(
                    out_dir,
                    {
                        "domain": "ntu",
                        "dataset": dataset_name,
                        "stage": "classify",
                        "model": model_name,
                        "encoder_checkpoint": resolved_encoder_checkpoint,
                        "timestamp": ts,
                        "config_name": cfg_name,
                    },
                )
                save_config(out_dir, dict(cfg))
                print_training_start(
                    f"AE transfer NTU ({dataset_name}/{model_name})",
                    {
                        "output_dir": out_dir,
                        "dataset": dataset_name,
                        "model": model_name,
                        "encoder_checkpoint": resolved_encoder_checkpoint,
                        "device": str(device),
                    },
                )

                arch_metadata = _build_classifier_arch_metadata(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    model_cfg=model_cfg,
                    channels=channels,
                    freq_bins=freq_bins,
                    time_steps=time_steps,
                    num_classes=num_classes,
                )
                model, _ = build_model(
                    dataset_name,
                    model_name,
                    num_classes=num_classes,
                    arch_metadata=arch_metadata,
                )
                encoder_state = _load_state_dict_file(resolved_encoder_checkpoint)
                model.encoder.load_state_dict(encoder_state, strict=True)
                model = model.to(device)

                history_rows, summary = _train_classifier_with_phases(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    transfer_cfg=transfer_cfg,
                )
                save_results_rows(out_dir, history_rows, base_name="train_history")
                save_results_rows(out_dir, summary["final_eval_rows"], base_name="summary_rows")

                local_checkpoint = None
                exported_checkpoint = None
                if save_classifier_weights:
                    local_checkpoint = save_model_weights(out_dir=out_dir, filename_stem=model_name, model=model)
                    _write_checkpoint_metadata(
                        Path(local_checkpoint),
                        {
                            "stage": "ae_trial_classifier_local_checkpoint",
                            "domain": "ntu",
                            "dataset": dataset_name,
                            "model": model_name,
                            "run_timestamp": ts,
                            "config_path": str(Path(config_path).resolve()) if config_path else None,
                            "encoder_checkpoint": str(Path(resolved_encoder_checkpoint).resolve()),
                            **arch_metadata,
                        },
                    )
                    exported_checkpoint = _export_classifier_checkpoint(
                        domain="ntu",
                        datasets_root=root,
                        weights_dir_name=classifier_weights_dir_name,
                        dataset_name=dataset_name,
                        task_name=None,
                        model_name=model_name,
                        source_checkpoint=local_checkpoint,
                        source_out_dir=out_dir,
                        config_path=config_path,
                        run_timestamp=ts,
                        arch_metadata=arch_metadata,
                    )

                save_training_metadata(
                    out_dir,
                    {
                        "stage": "ae_transfer_ntu",
                        "dataset": dataset_name,
                        "model_name": model_name,
                        "encoder_name": encoder_name,
                        "encoder_checkpoint": resolved_encoder_checkpoint,
                        "input_shape": [channels, freq_bins, time_steps],
                        "num_classes": num_classes,
                        "weights_path": local_checkpoint,
                        "dataset_export_weights_path": exported_checkpoint,
                        "dataset_export_weights_dir_name": classifier_weights_dir_name,
                        "arch_metadata": arch_metadata,
                        "summary": summary,
                    },
                )
                completed = True
                print(f"[RUN] Completed AE transfer: {out_dir}")
            except BaseException as exc:
                print(f"[ERROR] NTU AE transfer failed for {dataset_name}/{model_name}: {exc}")
                traceback.print_exc()
                raise
            finally:
                if not completed:
                    target_dir = move_incomplete_run(out_dir)
                    if target_dir:
                        print(f"[INFO] Moved incomplete run to: {target_dir}")


def _run_csi_bench(
    *,
    cfg: Mapping[str, Any],
    config_path: Optional[str],
    experiment_name: str,
    stages: Sequence[str],
    model_names: Sequence[str],
    model_cfg: Mapping[str, Any],
    save_encoder_weights: bool,
    save_classifier_weights: bool,
    encoder_weights_dir_name: str,
    classifier_weights_dir_name: str,
    device: torch.device,
    data_seed: int,
) -> None:
    csi_cfg = dict(cfg.get("csi_bench", {}) or {})
    dataset_root = str(csi_cfg.get("dataset_root", cfg.get("common", {}).get("dataset_root", "datasets/CSI-Bench")))
    task_names = parse_token_list(
        csi_cfg.get("task_names")
        or csi_cfg.get("tasks")
        or csi_cfg.get("task_name")
        or "HumanActivityRecognition,HumanIdentification"
    )
    task_names = [normalize_csi_bench_task_name(name) for name in task_names]
    if not task_names:
        task_names = ["HumanActivityRecognition", "HumanIdentification"]
    batch_size = int(csi_cfg.get("batch_size", 256))
    num_workers_raw = csi_cfg.get("num_workers")
    num_workers = int(num_workers_raw) if num_workers_raw is not None else None
    pin_memory_raw = csi_cfg.get("pin_memory")
    pin_memory = None if pin_memory_raw is None else parse_bool(pin_memory_raw, default=(device.type == "cuda"))
    persistent_workers_raw = csi_cfg.get("persistent_workers")
    persistent_workers = None if persistent_workers_raw is None else parse_bool(persistent_workers_raw, default=True)
    prefetch_factor_raw = csi_cfg.get("prefetch_factor")
    prefetch_factor = int(prefetch_factor_raw) if prefetch_factor_raw is not None else None
    normalisation = str(csi_cfg.get("normalisation", "zscore"))
    splits_dir_name = str(csi_cfg.get("splits_dir", "splits"))
    target_subcarriers = int(csi_cfg.get("target_subcarriers", DEFAULT_CSI_BENCH_SUBCARRIERS))
    target_timesteps = int(csi_cfg.get("target_timesteps", DEFAULT_CSI_BENCH_TIMESTEPS))
    only_splits = parse_splits(csi_cfg.get("only_splits") or csi_cfg.get("splits"), default=["train_id", "val_id", "test_id"])

    pretrain_cfg = dict(cfg.get("training", {}).get("pretrain", {}) or {})
    transfer_cfg = dict(cfg.get("training", {}).get("transfer", {}) or {})
    encoder_name = _encoder_type_from_cfg(model_cfg)
    encoder_checkpoints = transfer_cfg.get("encoder_checkpoints")
    explicit_encoder_checkpoint = transfer_cfg.get("encoder_checkpoint")
    _ = data_seed

    for task_name in task_names:
        encoder_checkpoint_path: Optional[str] = None

        if "pretrain" in stages:
            out_dir, ts, cfg_name = prepare_results_dir(
                experiment_name=experiment_name,
                cfg_path=config_path,
                relative_parts=["csi_bench", task_name, "pretrain", sanitize_token(encoder_name)],
            )
            completed = False
            try:
                save_run_meta(
                    out_dir,
                    {
                        "domain": "csi_bench",
                        "dataset": "CSI-Bench",
                        "task_name": task_name,
                        "stage": "pretrain",
                        "encoder": encoder_name,
                        "timestamp": ts,
                        "config_name": cfg_name,
                        "dataset_root": dataset_root,
                    },
                )
                save_config(out_dir, dict(cfg))
                print_training_start(
                    f"AE pretraining CSI-Bench ({task_name})",
                    {
                        "output_dir": out_dir,
                        "task": task_name,
                        "encoder": encoder_name,
                        "device": str(device),
                    },
                )

                bundle = load_csi_bench_supervised(
                    dataset_root=dataset_root,
                    task_name=task_name,
                    batch_size=batch_size,
                    normalisation=normalisation,
                    target_subcarriers=target_subcarriers,
                    target_timesteps=target_timesteps,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    prefetch_factor=prefetch_factor,
                    only_splits=only_splits,
                    splits_dir_name=splits_dir_name,
                    collate_fn=filter_none_collate,
                )
                if bundle is False or bundle is None:
                    raise RuntimeError(f"Failed to load CSI-Bench task {task_name}")

                loaders = bundle["loaders"]
                train_loader = loaders.get("train")
                val_loader = loaders.get("val")
                test_loader = loaders.get("test")
                if train_loader is None:
                    raise RuntimeError(f"Training loader missing for task {task_name}")

                shape = infer_input_shape_from_loader(train_loader)
                if shape is None:
                    raise RuntimeError(f"Unable to infer CSI-Bench input shape for task {task_name}")
                channels, freq_bins, time_steps = shape
                encoder_model_cfg = _encoder_model_config(model_cfg)
                model = build_autoencoder_from_config(encoder_model_cfg, sample_shape=shape, device=device).to(device)
                print(f"[MODEL] AE params={count_trainable_parameters(model):,}")

                history_rows, summary = _train_autoencoder_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    pretrain_cfg=pretrain_cfg,
                )
                arch_metadata = _build_encoder_arch_metadata(
                    dataset_name="CSI-Bench",
                    encoder_name=encoder_name,
                    encoder_model_cfg=encoder_model_cfg,
                    channels=channels,
                    freq_bins=freq_bins,
                    time_steps=time_steps,
                    task_name=task_name,
                )
                save_results_rows(out_dir, _history_rows_autoencoder(history_rows), base_name="train_history")
                save_results_rows(
                    out_dir,
                    _evaluate_autoencoder_splits(
                        model=model,
                        device=device,
                        split_loaders={"val": val_loader, "test": test_loader},
                    ),
                    base_name="summary_rows",
                )

                local_checkpoint = save_model_weights(out_dir=out_dir, filename_stem=encoder_name, model=model)
                _write_checkpoint_metadata(
                    Path(local_checkpoint),
                    {
                        "stage": "ae_trial_pretrain_local_checkpoint",
                        "domain": "csi_bench",
                        "dataset": "CSI-Bench",
                        "task_name": task_name,
                        "encoder_name": encoder_name,
                        "run_timestamp": ts,
                        "config_path": str(Path(config_path).resolve()) if config_path else None,
                        **arch_metadata,
                    },
                )
                encoder_checkpoint_path = local_checkpoint
                exported_checkpoint = None
                if save_encoder_weights:
                    exported_checkpoint = _export_encoder_checkpoint(
                        domain="csi_bench",
                        datasets_root=dataset_root,
                        weights_dir_name=encoder_weights_dir_name,
                        dataset_name="CSI-Bench",
                        task_name=task_name,
                        encoder_name=encoder_name,
                        source_checkpoint=local_checkpoint,
                        source_out_dir=out_dir,
                        config_path=config_path,
                        run_timestamp=ts,
                        arch_metadata=arch_metadata,
                    )

                save_training_metadata(
                    out_dir,
                    {
                        "stage": "ae_pretrain_csi_bench",
                        "dataset": "CSI-Bench",
                        "task_name": task_name,
                        "encoder_name": encoder_name,
                        "dataset_root": dataset_root,
                        "device": str(device),
                        "input_shape": [channels, freq_bins, time_steps],
                        "weights_path": local_checkpoint,
                        "dataset_export_weights_path": exported_checkpoint,
                        "dataset_export_weights_dir_name": encoder_weights_dir_name,
                        "arch_metadata": arch_metadata,
                        "task_metadata": collect_csi_bench_task_metadata(
                            dataset_root=dataset_root,
                            task_name=task_name,
                            splits_dir_name=splits_dir_name,
                            split_names=only_splits,
                        ),
                        "summary": summary,
                    },
                )
                completed = True
                print(f"[RUN] Completed AE pretraining: {out_dir}")
            except BaseException as exc:
                print(f"[ERROR] CSI-Bench AE pretraining failed for {task_name}: {exc}")
                traceback.print_exc()
                raise
            finally:
                if not completed:
                    target_dir = move_incomplete_run(out_dir)
                    if target_dir:
                        print(f"[INFO] Moved incomplete run to: {target_dir}")

        if "classify" not in stages:
            continue

        bundle = load_csi_bench_supervised(
            dataset_root=dataset_root,
            task_name=task_name,
            batch_size=batch_size,
            normalisation=normalisation,
            target_subcarriers=target_subcarriers,
            target_timesteps=target_timesteps,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            only_splits=only_splits,
            splits_dir_name=splits_dir_name,
            collate_fn=filter_none_collate,
        )
        if bundle is False or bundle is None:
            raise RuntimeError(f"Failed to load CSI-Bench task {task_name}")

        loaders = bundle["loaders"]
        train_loader = loaders.get("train")
        val_loader = loaders.get("val")
        test_loader = loaders.get("test")
        if train_loader is None:
            raise RuntimeError(f"Training loader missing for task {task_name}")

        num_classes = int(bundle["num_classes"])
        shape = infer_input_shape_from_loader(train_loader)
        if shape is None:
            raise RuntimeError(f"Unable to infer CSI-Bench input shape for task {task_name}")
        channels, freq_bins, time_steps = shape

        explicit_encoder = (
            str(explicit_encoder_checkpoint)
            if explicit_encoder_checkpoint
            else _lookup_checkpoint_override(encoder_checkpoints, task_name)
        )
        resolved_encoder_checkpoint = encoder_checkpoint_path or _resolve_encoder_checkpoint(
            domain="csi_bench",
            datasets_root=dataset_root,
            weights_dir_name=encoder_weights_dir_name,
            dataset_name="CSI-Bench",
            task_name=task_name,
            encoder_name=encoder_name,
            explicit_checkpoint=explicit_encoder,
        )

        for model_name in model_names:
            out_dir, ts, cfg_name = prepare_results_dir(
                experiment_name=experiment_name,
                cfg_path=config_path,
                relative_parts=["csi_bench", task_name, sanitize_token(model_name)],
            )
            completed = False
            try:
                save_run_meta(
                    out_dir,
                    {
                        "domain": "csi_bench",
                        "dataset": "CSI-Bench",
                        "task_name": task_name,
                        "stage": "classify",
                        "model": model_name,
                        "encoder_checkpoint": resolved_encoder_checkpoint,
                        "timestamp": ts,
                        "config_name": cfg_name,
                    },
                )
                save_config(out_dir, dict(cfg))
                print_training_start(
                    f"AE transfer CSI-Bench ({task_name}/{model_name})",
                    {
                        "output_dir": out_dir,
                        "task": task_name,
                        "model": model_name,
                        "encoder_checkpoint": resolved_encoder_checkpoint,
                        "device": str(device),
                    },
                )

                arch_metadata = _build_classifier_arch_metadata(
                    dataset_name="CSI-Bench",
                    model_name=model_name,
                    model_cfg=model_cfg,
                    channels=channels,
                    freq_bins=freq_bins,
                    time_steps=time_steps,
                    num_classes=num_classes,
                    task_name=task_name,
                )
                model, _ = build_model(
                    "csi-bench",
                    model_name,
                    num_classes=num_classes,
                    arch_metadata=arch_metadata,
                )
                encoder_state = _load_state_dict_file(resolved_encoder_checkpoint)
                model.encoder.load_state_dict(encoder_state, strict=True)
                model = model.to(device)

                history_rows, summary = _train_classifier_with_phases(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    transfer_cfg=transfer_cfg,
                )
                save_results_rows(out_dir, history_rows, base_name="train_history")
                save_results_rows(out_dir, summary["final_eval_rows"], base_name="summary_rows")

                local_checkpoint = None
                exported_checkpoint = None
                if save_classifier_weights:
                    local_checkpoint = save_model_weights(out_dir=out_dir, filename_stem=model_name, model=model)
                    _write_checkpoint_metadata(
                        Path(local_checkpoint),
                        {
                            "stage": "ae_trial_classifier_local_checkpoint",
                            "domain": "csi_bench",
                            "dataset": "CSI-Bench",
                            "task_name": task_name,
                            "model": model_name,
                            "run_timestamp": ts,
                            "config_path": str(Path(config_path).resolve()) if config_path else None,
                            "encoder_checkpoint": str(Path(resolved_encoder_checkpoint).resolve()),
                            **arch_metadata,
                        },
                    )
                    exported_checkpoint = _export_classifier_checkpoint(
                        domain="csi_bench",
                        datasets_root=dataset_root,
                        weights_dir_name=classifier_weights_dir_name,
                        dataset_name="CSI-Bench",
                        task_name=task_name,
                        model_name=model_name,
                        source_checkpoint=local_checkpoint,
                        source_out_dir=out_dir,
                        config_path=config_path,
                        run_timestamp=ts,
                        arch_metadata=arch_metadata,
                    )

                save_training_metadata(
                    out_dir,
                    {
                        "stage": "ae_transfer_csi_bench",
                        "dataset": "CSI-Bench",
                        "task_name": task_name,
                        "model_name": model_name,
                        "encoder_name": encoder_name,
                        "encoder_checkpoint": resolved_encoder_checkpoint,
                        "input_shape": [channels, freq_bins, time_steps],
                        "num_classes": num_classes,
                        "weights_path": local_checkpoint,
                        "dataset_export_weights_path": exported_checkpoint,
                        "dataset_export_weights_dir_name": classifier_weights_dir_name,
                        "arch_metadata": arch_metadata,
                        "task_metadata": collect_csi_bench_task_metadata(
                            dataset_root=dataset_root,
                            task_name=task_name,
                            splits_dir_name=splits_dir_name,
                            split_names=only_splits,
                        ),
                        "summary": summary,
                    },
                )
                completed = True
                print(f"[RUN] Completed AE transfer: {out_dir}")
            except BaseException as exc:
                print(f"[ERROR] CSI-Bench AE transfer failed for {task_name}/{model_name}: {exc}")
                traceback.print_exc()
                raise
            finally:
                if not completed:
                    target_dir = move_incomplete_run(out_dir)
                    if target_dir:
                        print(f"[INFO] Moved incomplete run to: {target_dir}")


def main(config_path: Optional[str] = None) -> None:
    cfg = load_config(config_path)
    configure_from_env_and_config(cfg)
    configure_torch_performance(cfg)

    common_cfg = cfg.get("common", {})
    experiment_name = str(common_cfg.get("experiment_name", "clean-ae-trials"))
    model_cfg = dict(cfg.get("model", {}) or {})
    model_names = _resolve_autoencoder_classifier_models(
        as_list(
            model_cfg.get("architectures") or model_cfg.get("models"),
            default=["AEHead_TCN", "AEHead_GRU", "AEHead_SSM"],
        )
    )
    save_encoder_weights = parse_bool(model_cfg.get("save_encoder_weights"), default=True)
    save_classifier_weights = parse_bool(
        model_cfg.get("save_classifier_weights", model_cfg.get("save_weights")),
        default=True,
    )
    encoder_weights_dir_name = str(
        model_cfg.get("encoder_weights_dir_name") or model_cfg.get("pretrain_weights_dir_name") or "autoencoder_params_ae_trials"
    )
    classifier_weights_dir_name = str(
        model_cfg.get("dataset_weights_dir_name")
        or model_cfg.get("classifier_weights_dir_name")
        or model_cfg.get("weights_dir_name")
        or "model_params_ae_trials"
    )

    data_cfg = cfg.get("data", {})
    device = _resolve_device(data_cfg.get("device"))
    data_seed = int(data_cfg.get("seed", 42))

    run_cfg = cfg.get("run", {})
    targets = {token.strip().lower().replace("-", "_") for token in as_list(run_cfg.get("targets"), default=["ntu", "csi_bench"])}
    stages = _resolve_stages(run_cfg.get("stages", run_cfg.get("stage")))

    if "ntu" in targets:
        _run_ntu(
            cfg=cfg,
            config_path=config_path,
            experiment_name=experiment_name,
            stages=stages,
            model_names=model_names,
            model_cfg=model_cfg,
            save_encoder_weights=save_encoder_weights,
            save_classifier_weights=save_classifier_weights,
            encoder_weights_dir_name=encoder_weights_dir_name,
            classifier_weights_dir_name=classifier_weights_dir_name,
            device=device,
            data_seed=data_seed,
        )
    if "csi_bench" in targets:
        _run_csi_bench(
            cfg=cfg,
            config_path=config_path,
            experiment_name=experiment_name,
            stages=stages,
            model_names=model_names,
            model_cfg=model_cfg,
            save_encoder_weights=save_encoder_weights,
            save_classifier_weights=save_classifier_weights,
            encoder_weights_dir_name=encoder_weights_dir_name,
            classifier_weights_dir_name=classifier_weights_dir_name,
            device=device,
            data_seed=data_seed,
        )


if __name__ == "__main__":
    cfg_path = os.environ.get("EXP_TRAIN_CONFIG")
    if cfg_path is None and len(sys.argv) > 1:
        cfg_path = resolve_config_path_from_arg(sys.argv[1])
    main(cfg_path)
