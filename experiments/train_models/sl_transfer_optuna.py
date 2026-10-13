"""Optuna hyperparameter tuning for two-stage supervised transfer (HAR <-> HID)."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from csi.data import get_experiment_loaders
from csi.data.utils import filter_none_collate, load_csi_bench_supervised
from csi.models.supervised import evaluate_classifier, train
from csi.models.utils.datasets import DatasetSpec
from csi.models.utils.experiment import load_config, resolve_config_path_from_arg, sanitize_token
from csi.models.utils.gpu_utils import _resolve_device
from csi.models.utils.optimisation import (
    CUDATrialOOMError,
    build_model_from_hparams,
    run_with_oom_guard,
    suggest_hparams_for_model,
)
from csi.models.utils.results import (
    prepare_results_dir,
    save_config,
    save_results_rows,
    save_run_meta,
)
from csi.models.utils.seeding import configure_from_env_and_config
from ..common.shared_utils import (
    collect_csi_bench_task_metadata,
    export_best_weight_to_datasets,
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
    infer_spec_from_bundle,
    json_dump,
    resolve_ntu_num_classes,
    resolve_project_path,
    resolve_supervised_models,
    state_name,
)

try:
    import optuna
    from optuna.exceptions import TrialPruned
    from optuna.trial import Trial
except Exception as exc:  # pragma: no cover
    optuna = None
    Trial = None
    TrialPruned = None
    _OPTUNA_IMPORT_ERROR = exc
else:
    _OPTUNA_IMPORT_ERROR = None


def _extract_state_dict_from_payload(payload: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint payload must be a mapping, got {type(payload)!r}")
    if "model_state_dict" in payload and isinstance(payload["model_state_dict"], Mapping):
        return payload["model_state_dict"]  # type: ignore[return-value]
    if "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
        return payload["state_dict"]  # type: ignore[return-value]
    if payload and all(torch.is_tensor(v) for v in payload.values()):
        return payload  # type: ignore[return-value]
    raise ValueError("Unsupported checkpoint format (expected state_dict or model_state_dict).")


def _load_state_dict_file(path: str) -> Mapping[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return _extract_state_dict_from_payload(payload)


def _load_compatible_weights(
    model: nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    target_state = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []

    for key, value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            skipped.append(key)
            continue
        if tuple(target_value.shape) != tuple(value.shape):
            skipped.append(key)
            continue
        compatible[key] = value

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "loaded_keys": len(compatible),
        "skipped_keys": len(skipped),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "missing_key_names": list(missing),
        "unexpected_key_names": list(unexpected),
    }


def _is_classifier_head_param(name: str) -> bool:
    return (
        name.startswith("head.")
        or name.startswith("fc.")
        or name.startswith("model.fc.")
    )


def _apply_trial_weight_retention(
    *,
    study,
    trial_weights_dir: str,
    direction: str,
    mode: str,
    top_k: int,
    max_saved: Optional[int],
) -> Dict[str, Any]:
    if not os.path.isdir(trial_weights_dir):
        return {"kept": 0, "deleted": 0, "reason": "missing_dir"}

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"all", "best_only", "top_k"}:
        normalized_mode = "top_k"

    saved_numbers = set()
    trial_paths: Dict[int, Path] = {}
    scanned = 0
    for path in Path(trial_weights_dir).glob("trial_*.pt"):
        scanned += 1
        stem = path.stem
        suffix = stem.replace("trial_", "", 1)
        try:
            trial_number = int(suffix)
        except Exception:
            continue
        saved_numbers.add(trial_number)
        trial_paths[trial_number] = path

    if not saved_numbers:
        return {
            "mode": normalized_mode,
            "kept": 0,
            "deleted": 0,
            "scanned": scanned,
            "top_k": int(top_k),
            "max_saved": (None if max_saved is None else int(max_saved)),
            "reason": "no_saved_trials",
        }

    completed = [trial for trial in study.trials if state_name(trial.state) == "COMPLETE" and trial.value is not None]
    completed.sort(key=lambda trial: float(trial.value), reverse=(direction == "maximize"))

    selected_numbers: List[int] = []
    if normalized_mode == "all":
        for trial in completed:
            trial_number = int(trial.number)
            if trial_number in saved_numbers:
                selected_numbers.append(trial_number)
    else:
        desired = 1 if normalized_mode == "best_only" else max(1, int(top_k))
        for trial in completed:
            trial_number = int(trial.number)
            if trial_number not in saved_numbers:
                continue
            selected_numbers.append(trial_number)
            if len(selected_numbers) >= desired:
                break

    if max_saved is not None and int(max_saved) > 0:
        selected_numbers = selected_numbers[: int(max_saved)]

    keep_numbers = set(selected_numbers)
    deleted = 0
    for trial_number, path in trial_paths.items():
        if trial_number in keep_numbers:
            continue
        try:
            path.unlink()
            deleted += 1
        except Exception:
            pass

    return {
        "mode": normalized_mode,
        "kept": len(keep_numbers),
        "deleted": deleted,
        "scanned": scanned,
        "top_k": int(top_k),
        "max_saved": (None if max_saved is None else int(max_saved)),
    }


def _apply_finetune_mode(model: nn.Module, mode: str) -> Dict[str, int]:
    key = str(mode).strip().lower()
    if key in {"full", "all", "full_finetune"}:
        for _, param in model.named_parameters():
            param.requires_grad = True
    elif key in {"linear_probe", "head_only", "frozen_backbone"}:
        any_trainable = False
        for name, param in model.named_parameters():
            is_head = _is_classifier_head_param(name)
            param.requires_grad = bool(is_head)
            any_trainable = any_trainable or bool(is_head)
        if not any_trainable:
            raise RuntimeError(
                "linear_probe mode found no head parameters. "
                "Expected names prefixed by one of: head., fc., model.fc."
            )
    else:
        raise ValueError(f"Unsupported transfer finetune mode '{mode}'.")

    total = 0
    trainable = 0
    for _, param in model.named_parameters():
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
    return {"trainable_params": trainable, "total_params": total}


def _stage_train_cfg(
    *,
    train_cfg: Mapping[str, Any],
    transfer_cfg: Mapping[str, Any],
    stage: str,
) -> Dict[str, Any]:
    stage_cfg = dict(transfer_cfg.get(stage, {}) or {})
    prefix = f"{stage}_"

    def _pick(name: str, default: Any) -> Any:
        if name in stage_cfg:
            return stage_cfg[name]
        prefixed = f"{prefix}{name}"
        if prefixed in transfer_cfg:
            return transfer_cfg[prefixed]
        return train_cfg.get(name, default)

    patience_raw = _pick("patience", train_cfg.get("patience"))
    grad_clip_raw = _pick("grad_clip", train_cfg.get("grad_clip"))
    log_every_raw = _pick("log_every_epochs", train_cfg.get("log_every_epochs"))

    return {
        "learning_rate": float(_pick("learning_rate", 1e-3)),
        "epochs": int(_pick("epochs", 100)),
        "progress_bar": parse_bool(_pick("progress_bar", False), default=False),
        "patience": int(patience_raw) if patience_raw is not None else None,
        "min_epochs": int(_pick("min_epochs", 1)),
        "monitor_metric": str(_pick("monitor_metric", "accuracy")),
        "min_delta": float(_pick("min_delta", 0.0)),
        "weight_decay": float(_pick("weight_decay", 0.0)),
        "grad_clip": float(grad_clip_raw) if grad_clip_raw is not None else None,
        "optimizer_cfg": _pick("optimizer", None),
        "scheduler_cfg": _pick("scheduler", None),
        "amp": _pick("amp", None),
        "amp_dtype": str(_pick("amp_dtype", "float16")),
        "log_every_epochs": int(log_every_raw) if log_every_raw is not None else None,
    }


def _resolve_transfer_directions(transfer_cfg: Mapping[str, Any]) -> set[str]:
    directions = parse_token_list(transfer_cfg.get("directions") or transfer_cfg.get("direction"))
    if not directions:
        return {"har_to_hid", "hid_to_har"}
    normalised: set[str] = set()
    for item in directions:
        key = item.strip().lower().replace("-", "_")
        if key in {"all", "both"}:
            return {"har_to_hid", "hid_to_har"}
        normalised.add(key)
    return normalised


def _resolve_ntu_pairs(transfer_cfg: Mapping[str, Any]) -> List[Dict[str, str]]:
    directions = _resolve_transfer_directions(transfer_cfg)
    defaults = {
        "har_to_hid": {
            "name": "har_to_hid",
            "source_dataset": "NTU-Fi_HAR",
            "target_dataset": "NTU-Fi-HumanID",
        },
        "hid_to_har": {
            "name": "hid_to_har",
            "source_dataset": "NTU-Fi-HumanID",
            "target_dataset": "NTU-Fi_HAR",
        },
    }
    return [value for key, value in defaults.items() if key in directions]


def _resolve_csi_pairs(transfer_cfg: Mapping[str, Any]) -> List[Dict[str, str]]:
    directions = _resolve_transfer_directions(transfer_cfg)
    defaults = {
        "har_to_hid": {
            "name": "har_to_hid",
            "source_task": "HumanActivityRecognition",
            "target_task": "HumanIdentification",
        },
        "hid_to_har": {
            "name": "hid_to_har",
            "source_task": "HumanIdentification",
            "target_task": "HumanActivityRecognition",
        },
    }
    return [value for key, value in defaults.items() if key in directions]


def _evaluate_transfer_target(
    *,
    model: nn.Module,
    criterion: nn.Module,
    loader,
    device: torch.device,
    use_tqdm: bool,
    desc: str,
) -> Dict[str, float]:
    metrics, _ = run_with_oom_guard(
        lambda: evaluate_classifier(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            use_tqdm=use_tqdm,
            collect_predictions=False,
            collect_probabilities=False,
            desc=desc,
        )
    )
    return {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "loss": float(metrics.get("loss", 0.0)),
        "f1_macro": float(metrics.get("f1_macro", 0.0)),
        "f1_weighted": float(metrics.get("f1_weighted", 0.0)),
    }


def _train_model(
    *,
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    cfg: Mapping[str, Any],
):
    return run_with_oom_guard(
        lambda: train(
            model=model,
            tensor_loader=train_loader,
            num_epochs=int(cfg["epochs"]),
            learning_rate=float(cfg["learning_rate"]),
            criterion=criterion,
            device=device,
            use_tqdm=bool(cfg["progress_bar"]),
            val_loader=val_loader,
            patience=cfg["patience"] if val_loader is not None else None,
            min_epochs=int(cfg["min_epochs"]),
            monitor_metric=str(cfg["monitor_metric"]),
            min_delta=float(cfg["min_delta"]),
            weight_decay=float(cfg["weight_decay"]),
            grad_clip=cfg["grad_clip"],
            optimizer_cfg=cfg["optimizer_cfg"],
            scheduler_cfg=cfg["scheduler_cfg"],
            amp=cfg["amp"],
            amp_dtype=str(cfg["amp_dtype"]),
            log_every_epochs=cfg["log_every_epochs"],
        )
    )


def _run_ntu_transfer_optuna(
    *,
    cfg: Mapping[str, Any],
    config_path: Optional[str],
    experiment_name: str,
    model_names: List[str],
    device: torch.device,
    data_seed: int,
) -> None:
    ntu_cfg = dict(cfg.get("ntu", {}) or {})
    transfer_cfg = dict(cfg.get("transfer", {}) or {})
    train_cfg = dict(cfg.get("training", {}) or {})
    opt_cfg = dict(cfg.get("optuna", {}) or {})
    export_cfg = dict(cfg.get("export", {}) or {})

    root = str(ntu_cfg.get("root", cfg.get("common", {}).get("root", "datasets")))
    task = str(ntu_cfg.get("task", "classification"))
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

    finetune_mode = str(transfer_cfg.get("finetune_mode", "full"))
    base_model_mode = str(transfer_cfg.get("base_model_mode", "source_supervised")).strip().lower()
    source_checkpoint = transfer_cfg.get("source_checkpoint")
    source_stage_cfg = _stage_train_cfg(train_cfg=train_cfg, transfer_cfg=transfer_cfg, stage="source")
    target_stage_cfg = _stage_train_cfg(train_cfg=train_cfg, transfer_cfg=transfer_cfg, stage="target")
    pairs = _resolve_ntu_pairs(transfer_cfg)
    if not pairs:
        raise ValueError("No NTU transfer pairs resolved from transfer.directions.")

    n_trials = int(opt_cfg.get("n_trials", 60))
    timeout_raw = opt_cfg.get("timeout")
    timeout = int(timeout_raw) if timeout_raw is not None else None
    n_jobs = int(opt_cfg.get("n_jobs", 1))
    direction = str(opt_cfg.get("direction", "maximize")).strip().lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("optuna.direction must be either 'maximize' or 'minimize'.")
    objective_metric = str(opt_cfg.get("objective_metric", "val_accuracy")).strip().lower()
    if objective_metric not in {"val_accuracy", "val_loss"}:
        raise ValueError("optuna.objective_metric must be 'val_accuracy' or 'val_loss'.")
    study_prefix = str(opt_cfg.get("study_name", "ntu_transfer_optuna"))
    load_if_exists = parse_bool(opt_cfg.get("load_if_exists"), default=True)
    storage_url = opt_cfg.get("storage")
    gc_after_trial = parse_bool(opt_cfg.get("gc_after_trial"), default=True)
    sampler_seed = int(opt_cfg.get("sampler_seed", data_seed))
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)

    pruner_cfg = opt_cfg.get("pruner", {})
    use_pruner = parse_bool(pruner_cfg.get("enabled"), default=True)
    pruner = None
    if use_pruner:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(pruner_cfg.get("n_startup_trials", 5)),
            n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 0)),
            interval_steps=int(pruner_cfg.get("interval_steps", 1)),
        )

    model_cfg = dict(cfg.get("model", {}) or {})
    save_trial_weights = parse_bool(model_cfg.get("save_trial_weights"), default=False)
    trial_weights_cfg = model_cfg.get("trial_weights", {}) or {}
    trial_weights_dir_name = sanitize_token(str(trial_weights_cfg.get("dir_name", "trial_weights_transfer_optuna"))) or "trial_weights_transfer_optuna"
    trial_weights_mode = str(trial_weights_cfg.get("mode", "top_k"))
    trial_weights_top_k = int(trial_weights_cfg.get("top_k", 5))
    trial_weights_save_every = max(1, int(trial_weights_cfg.get("save_every", 1)))
    trial_weights_max_saved_raw = trial_weights_cfg.get("max_saved")
    trial_weights_max_saved = int(trial_weights_max_saved_raw) if trial_weights_max_saved_raw is not None else None
    trial_weights_cleanup = parse_bool(trial_weights_cfg.get("cleanup_after_study"), default=True)
    export_best_enabled = parse_bool(export_cfg.get("save_best_to_datasets"), default=True)
    export_datasets_root = str(export_cfg.get("datasets_root", root))
    export_weights_dir_name = str(export_cfg.get("weights_dir_name", "model_params_transfer_optuna_best"))
    export_overwrite_strategy = str(export_cfg.get("overwrite_strategy", "unique"))
    export_include_full_config = parse_bool(export_cfg.get("include_full_config"), default=True)
    export_fail_on_error = parse_bool(export_cfg.get("fail_on_error"), default=False)

    criterion = nn.CrossEntropyLoss()

    for pair in pairs:
        source_dataset = pair["source_dataset"]
        target_dataset = pair["target_dataset"]
        pair_name = pair["name"]
        for model_name in model_names:
            model_token = sanitize_token(model_name)
            out_dir, ts, cfg_name = prepare_results_dir(
                experiment_name=experiment_name,
                cfg_path=config_path,
                relative_parts=["ntu", pair_name, model_token],
            )
            completed = False
            trial_weights_dir = os.path.join(out_dir, trial_weights_dir_name)
            if save_trial_weights:
                os.makedirs(trial_weights_dir, exist_ok=True)
            try:
                save_run_meta(
                    out_dir,
                    {
                        "domain": "ntu",
                        "transfer_pair": pair_name,
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "model": model_name,
                        "device": str(device),
                        "base_model_mode": base_model_mode,
                        "finetune_mode": finetune_mode,
                        "optuna": {
                            "n_trials": n_trials,
                            "timeout": timeout,
                            "direction": direction,
                            "objective_metric": objective_metric,
                            "n_jobs": n_jobs,
                        },
                    },
                )
                save_config(out_dir, dict(cfg))
                print_training_start(
                    f"NTU transfer Optuna ({pair_name}/{model_name})",
                    {
                        "output_dir": out_dir,
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "model": model_name,
                        "device": str(device),
                        "base_model_mode": base_model_mode,
                        "finetune_mode": finetune_mode,
                        "n_trials": n_trials,
                        "timeout": timeout,
                    },
                )

                source_bundle = get_experiment_loaders(
                    source_dataset,
                    root,
                    task=task,
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
                target_bundle = get_experiment_loaders(
                    target_dataset,
                    root,
                    task=task,
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
                source_train = source_bundle.train_loader
                source_val = source_bundle.val_loader
                target_train = target_bundle.train_loader
                target_val = target_bundle.val_loader
                target_test = target_bundle.test_loader
                if source_train is None or target_train is None:
                    raise RuntimeError("Missing transfer train loader.")
                eval_loader = target_val if target_val is not None else target_test
                if eval_loader is None:
                    raise RuntimeError("Target val/test loaders both missing.")

                source_spec = infer_spec_from_bundle(source_dataset, source_bundle.info or {}, source_train)
                target_spec = infer_spec_from_bundle(target_dataset, target_bundle.info or {}, target_train)
                source_classes = resolve_ntu_num_classes(source_bundle.info or {}, source_train)
                target_classes = resolve_ntu_num_classes(target_bundle.info or {}, target_train)

                save_training_metadata(
                    out_dir,
                    {
                        "stage": "transfer_optuna_ntu",
                        "transfer_pair": pair_name,
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "model_name": model_name,
                        "source_spec": {
                            "channels": source_spec.channels,
                            "freq_bins": source_spec.freq_bins,
                            "time_steps": source_spec.time_steps,
                            "num_classes": source_classes,
                        },
                        "target_spec": {
                            "channels": target_spec.channels,
                            "freq_bins": target_spec.freq_bins,
                            "time_steps": target_spec.time_steps,
                            "num_classes": target_classes,
                        },
                        "source_stage_cfg": source_stage_cfg,
                        "target_stage_cfg": target_stage_cfg,
                        "optuna": {
                            "n_trials": n_trials,
                            "timeout": timeout,
                            "direction": direction,
                            "objective_metric": objective_metric,
                        },
                    },
                )

                best_weights_path = os.path.join(out_dir, f"{model_token}_best.pt")

                def objective(trial: Trial) -> float:  # type: ignore
                    hparams = suggest_hparams_for_model(trial, model_name, target_spec)
                    source_state = None
                    source_load_stats: Dict[str, Any] = {}
                    try:
                        if base_model_mode in {"source_supervised", "pretrain_source"}:
                            source_model = build_model_from_hparams(source_spec, model_name, source_classes, hparams).to(device)
                            source_summary = _train_model(
                                model=source_model,
                                train_loader=source_train,
                                val_loader=source_val,
                                criterion=criterion,
                                device=device,
                                cfg=source_stage_cfg,
                            )
                            trial.set_user_attr("source_best_epoch", int(source_summary.best_epoch))
                            source_state = {k: v.detach().cpu() for k, v in source_model.state_dict().items()}
                            del source_model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        elif base_model_mode == "source_checkpoint":
                            if not source_checkpoint:
                                raise ValueError("transfer.source_checkpoint is required for base_model_mode=source_checkpoint.")
                            source_state = _load_state_dict_file(str(resolve_project_path(str(source_checkpoint))))
                        elif base_model_mode in {"scratch", "none"}:
                            source_state = None
                        else:
                            raise ValueError(
                                f"Unsupported transfer.base_model_mode '{base_model_mode}'. "
                                "Use source_supervised, source_checkpoint, or scratch."
                            )

                        target_model = build_model_from_hparams(target_spec, model_name, target_classes, hparams).to(device)
                        if source_state is not None:
                            source_load_stats = _load_compatible_weights(target_model, source_state)
                        freeze_stats = _apply_finetune_mode(target_model, finetune_mode)
                        target_summary = _train_model(
                            model=target_model,
                            train_loader=target_train,
                            val_loader=target_val,
                            criterion=criterion,
                            device=device,
                            cfg=target_stage_cfg,
                        )
                        target_metrics = _evaluate_transfer_target(
                            model=target_model,
                            criterion=criterion,
                            loader=eval_loader,
                            device=device,
                            use_tqdm=bool(target_stage_cfg["progress_bar"]),
                            desc=f"transfer_val_{trial.number}",
                        )
                        objective_value = (
                            float(target_metrics["accuracy"])
                            if objective_metric == "val_accuracy"
                            else float(target_metrics["loss"])
                        )

                        trial.set_user_attr("objective_metric", objective_metric)
                        trial.set_user_attr("target_val_accuracy", float(target_metrics["accuracy"]))
                        trial.set_user_attr("target_val_loss", float(target_metrics["loss"]))
                        trial.set_user_attr("target_best_epoch", int(target_summary.best_epoch))
                        trial.set_user_attr("freeze_trainable_params", int(freeze_stats["trainable_params"]))
                        trial.set_user_attr("loaded_source_keys", int(source_load_stats.get("loaded_keys", 0)))

                        if save_trial_weights and (trial.number + 1) % trial_weights_save_every == 0:
                            torch.save(
                                target_model.state_dict(),
                                os.path.join(trial_weights_dir, f"trial_{trial.number:04d}.pt"),
                            )
                        return objective_value

                    except CUDATrialOOMError as exc:
                        trial.set_user_attr("oom", True)
                        raise TrialPruned(str(exc)) from exc
                    except RuntimeError as exc:
                        msg = str(exc).lower()
                        if "out of memory" in msg and ("cuda" in msg or "cudnn" in msg):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            trial.set_user_attr("oom", True)
                            raise TrialPruned("CUDA OOM during transfer trial") from exc
                        raise
                    finally:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                resolved_storage = storage_url or f"sqlite:///{os.path.join(out_dir, 'optuna_study.db')}"
                study_name = f"{study_prefix}_{sanitize_token(pair_name)}_{sanitize_token(model_name)}"
                study = optuna.create_study(
                    study_name=study_name,
                    direction=direction,
                    sampler=sampler,
                    pruner=pruner,
                    storage=resolved_storage,
                    load_if_exists=load_if_exists,
                )
                study.optimize(
                    objective,
                    n_trials=n_trials,
                    timeout=timeout,
                    n_jobs=n_jobs,
                    gc_after_trial=gc_after_trial,
                )

                if save_trial_weights and trial_weights_cleanup:
                    retention_summary = _apply_trial_weight_retention(
                        study=study,
                        trial_weights_dir=trial_weights_dir,
                        direction=direction,
                        mode=trial_weights_mode,
                        top_k=trial_weights_top_k,
                        max_saved=trial_weights_max_saved,
                    )
                    json_dump(os.path.join(trial_weights_dir, "retention_summary.json"), retention_summary)
                    print(
                        "[TRIAL-WEIGHTS] Retention applied: "
                        f"mode={retention_summary.get('mode')} kept={retention_summary.get('kept')} "
                        f"deleted={retention_summary.get('deleted')}"
                    )

                trial_rows: List[Dict[str, Any]] = []
                for trial in study.trials:
                    row: Dict[str, Any] = {
                        "number": trial.number,
                        "state": state_name(trial.state),
                        "value": trial.value,
                        "duration_sec": trial.duration.total_seconds() if trial.duration is not None else None,
                    }
                    for key, value in trial.params.items():
                        row[f"param_{key}"] = value
                    for key, value in trial.user_attrs.items():
                        row[f"attr_{key}"] = value
                    trial_rows.append(row)
                save_results_rows(out_dir, trial_rows, base_name="optuna_trials")

                best_trial = study.best_trial
                best_payload = {
                    "number": best_trial.number,
                    "value": best_trial.value,
                    "params": dict(best_trial.params),
                    "objective_metric": objective_metric,
                    "direction": direction,
                    "study_name": study.study_name,
                    "storage": resolved_storage,
                    "target_val_accuracy": best_trial.user_attrs.get("target_val_accuracy"),
                    "target_val_loss": best_trial.user_attrs.get("target_val_loss"),
                }
                json_dump(os.path.join(out_dir, "best_trial.json"), best_payload)

                # Retrain once with best params to generate clean checkpoint.
                best_hparams = dict(best_trial.params)
                source_state = None
                if base_model_mode in {"source_supervised", "pretrain_source"}:
                    source_model = build_model_from_hparams(source_spec, model_name, source_classes, best_hparams).to(device)
                    _ = _train_model(
                        model=source_model,
                        train_loader=source_train,
                        val_loader=source_val,
                        criterion=criterion,
                        device=device,
                        cfg=source_stage_cfg,
                    )
                    source_state = {k: v.detach().cpu() for k, v in source_model.state_dict().items()}
                    del source_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                elif base_model_mode == "source_checkpoint":
                    if not source_checkpoint:
                        raise ValueError("transfer.source_checkpoint is required for base_model_mode=source_checkpoint.")
                    source_state = _load_state_dict_file(str(resolve_project_path(str(source_checkpoint))))

                best_model = build_model_from_hparams(target_spec, model_name, target_classes, best_hparams).to(device)
                if source_state is not None:
                    _ = _load_compatible_weights(best_model, source_state)
                _ = _apply_finetune_mode(best_model, finetune_mode)
                _ = _train_model(
                    model=best_model,
                    train_loader=target_train,
                    val_loader=target_val,
                    criterion=criterion,
                    device=device,
                    cfg=target_stage_cfg,
                )
                torch.save(best_model.state_dict(), best_weights_path)

                final_rows: List[Dict[str, Any]] = []
                for split_name, loader in [("val", target_val), ("test", target_test)]:
                    if loader is None:
                        continue
                    metrics = _evaluate_transfer_target(
                        model=best_model,
                        criterion=criterion,
                        loader=loader,
                        device=device,
                        use_tqdm=bool(target_stage_cfg["progress_bar"]),
                        desc=f"best_{split_name}",
                    )
                    final_rows.append({"split": split_name, **metrics})
                save_results_rows(out_dir, final_rows, base_name="best_eval")

                if export_best_enabled:
                    export_meta = {
                        "domain": "ntu",
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "model": model_name,
                        "best_trial": best_payload,
                    }
                    try:
                        export_result = export_best_weight_to_datasets(
                            source_checkpoint=best_weights_path,
                            datasets_root=export_datasets_root,
                            domain="ntu",
                            dataset_name=target_dataset,
                            model_name=model_name,
                            run_output_dir=out_dir,
                            run_timestamp=ts,
                            config_name=cfg_name,
                            weights_dir_name=export_weights_dir_name,
                            overwrite_strategy=export_overwrite_strategy,
                            include_full_config=export_include_full_config,
                            config_payload=cfg,
                            metadata_payload=export_meta,
                        )
                        print(f"[EXPORT] {export_result}")
                    except Exception as export_exc:
                        print(f"[WARN] Failed to export best checkpoint: {export_exc}")
                        if export_fail_on_error:
                            raise

                completed = True
                print(f"[RUN] Completed: {out_dir}")
            except BaseException as exc:
                print(f"[ERROR] Transfer Optuna NTU failed for {pair_name}/{model_name}: {exc}")
                traceback.print_exc()
                raise
            finally:
                if not completed:
                    target_dir = move_incomplete_run(out_dir)
                    if target_dir:
                        print(f"[INFO] Moved incomplete run to: {target_dir}")


def _run_csi_transfer_optuna(
    *,
    cfg: Mapping[str, Any],
    config_path: Optional[str],
    experiment_name: str,
    model_names: List[str],
    device: torch.device,
) -> None:
    csi_cfg = dict(cfg.get("csi_bench", {}) or {})
    transfer_cfg = dict(cfg.get("transfer", {}) or {})
    train_cfg = dict(cfg.get("training", {}) or {})
    opt_cfg = dict(cfg.get("optuna", {}) or {})
    export_cfg = dict(cfg.get("export", {}) or {})

    dataset_root = str(csi_cfg.get("dataset_root", cfg.get("common", {}).get("dataset_root", "datasets/CSI-Bench")))
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

    finetune_mode = str(transfer_cfg.get("finetune_mode", "full"))
    base_model_mode = str(transfer_cfg.get("base_model_mode", "source_supervised")).strip().lower()
    source_checkpoint = transfer_cfg.get("source_checkpoint")
    source_stage_cfg = _stage_train_cfg(train_cfg=train_cfg, transfer_cfg=transfer_cfg, stage="source")
    target_stage_cfg = _stage_train_cfg(train_cfg=train_cfg, transfer_cfg=transfer_cfg, stage="target")
    pairs = _resolve_csi_pairs(transfer_cfg)
    if not pairs:
        raise ValueError("No CSI-Bench transfer pairs resolved from transfer.directions.")

    n_trials = int(opt_cfg.get("n_trials", 60))
    timeout_raw = opt_cfg.get("timeout")
    timeout = int(timeout_raw) if timeout_raw is not None else None
    n_jobs = int(opt_cfg.get("n_jobs", 1))
    direction = str(opt_cfg.get("direction", "maximize")).strip().lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("optuna.direction must be either 'maximize' or 'minimize'.")
    objective_metric = str(opt_cfg.get("objective_metric", "val_accuracy")).strip().lower()
    if objective_metric not in {"val_accuracy", "val_loss"}:
        raise ValueError("optuna.objective_metric must be 'val_accuracy' or 'val_loss'.")
    study_prefix = str(opt_cfg.get("study_name", "csi_transfer_optuna"))
    load_if_exists = parse_bool(opt_cfg.get("load_if_exists"), default=True)
    storage_url = opt_cfg.get("storage")
    gc_after_trial = parse_bool(opt_cfg.get("gc_after_trial"), default=True)
    sampler_seed = int(opt_cfg.get("sampler_seed", 42))
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)

    pruner_cfg = opt_cfg.get("pruner", {})
    use_pruner = parse_bool(pruner_cfg.get("enabled"), default=True)
    pruner = None
    if use_pruner:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(pruner_cfg.get("n_startup_trials", 5)),
            n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 0)),
            interval_steps=int(pruner_cfg.get("interval_steps", 1)),
        )

    model_cfg = dict(cfg.get("model", {}) or {})
    save_trial_weights = parse_bool(model_cfg.get("save_trial_weights"), default=False)
    trial_weights_cfg = model_cfg.get("trial_weights", {}) or {}
    trial_weights_dir_name = sanitize_token(str(trial_weights_cfg.get("dir_name", "trial_weights_transfer_optuna"))) or "trial_weights_transfer_optuna"
    trial_weights_mode = str(trial_weights_cfg.get("mode", "top_k"))
    trial_weights_top_k = int(trial_weights_cfg.get("top_k", 5))
    trial_weights_save_every = max(1, int(trial_weights_cfg.get("save_every", 1)))
    trial_weights_max_saved_raw = trial_weights_cfg.get("max_saved")
    trial_weights_max_saved = int(trial_weights_max_saved_raw) if trial_weights_max_saved_raw is not None else None
    trial_weights_cleanup = parse_bool(trial_weights_cfg.get("cleanup_after_study"), default=True)
    export_best_enabled = parse_bool(export_cfg.get("save_best_to_datasets"), default=True)
    export_datasets_root = str(export_cfg.get("datasets_root", "datasets"))
    export_weights_dir_name = str(export_cfg.get("weights_dir_name", "model_params_transfer_optuna_best"))
    export_overwrite_strategy = str(export_cfg.get("overwrite_strategy", "unique"))
    export_include_full_config = parse_bool(export_cfg.get("include_full_config"), default=True)
    export_fail_on_error = parse_bool(export_cfg.get("fail_on_error"), default=False)

    criterion = nn.CrossEntropyLoss()

    for pair in pairs:
        source_task = normalize_csi_bench_task_name(pair["source_task"])
        target_task = normalize_csi_bench_task_name(pair["target_task"])
        pair_name = pair["name"]
        for model_name in model_names:
            model_token = sanitize_token(model_name)
            out_dir, ts, cfg_name = prepare_results_dir(
                experiment_name=experiment_name,
                cfg_path=config_path,
                relative_parts=["csi_bench", pair_name, model_token],
            )
            completed = False
            trial_weights_dir = os.path.join(out_dir, trial_weights_dir_name)
            if save_trial_weights:
                os.makedirs(trial_weights_dir, exist_ok=True)
            try:
                save_run_meta(
                    out_dir,
                    {
                        "domain": "csi_bench",
                        "transfer_pair": pair_name,
                        "source_task": source_task,
                        "target_task": target_task,
                        "model": model_name,
                        "device": str(device),
                        "base_model_mode": base_model_mode,
                        "finetune_mode": finetune_mode,
                        "optuna": {
                            "n_trials": n_trials,
                            "timeout": timeout,
                            "direction": direction,
                            "objective_metric": objective_metric,
                            "n_jobs": n_jobs,
                        },
                    },
                )
                save_config(out_dir, dict(cfg))
                print_training_start(
                    f"CSI transfer Optuna ({pair_name}/{model_name})",
                    {
                        "output_dir": out_dir,
                        "source_task": source_task,
                        "target_task": target_task,
                        "model": model_name,
                        "device": str(device),
                        "base_model_mode": base_model_mode,
                        "finetune_mode": finetune_mode,
                        "n_trials": n_trials,
                        "timeout": timeout,
                    },
                )

                source_bundle = load_csi_bench_supervised(
                    dataset_root=dataset_root,
                    task_name=source_task,
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
                target_bundle = load_csi_bench_supervised(
                    dataset_root=dataset_root,
                    task_name=target_task,
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
                if source_bundle is False or source_bundle is None:
                    raise RuntimeError(f"Failed to load source task {source_task}")
                if target_bundle is False or target_bundle is None:
                    raise RuntimeError(f"Failed to load target task {target_task}")

                source_loaders = source_bundle["loaders"]
                target_loaders = target_bundle["loaders"]
                source_train = source_loaders.get("train")
                source_val = source_loaders.get("val")
                target_train = target_loaders.get("train")
                target_val = target_loaders.get("val")
                target_test = target_loaders.get("test")
                if source_train is None or target_train is None:
                    raise RuntimeError("Missing transfer train loader.")
                eval_loader = target_val if target_val is not None else target_test
                if eval_loader is None:
                    raise RuntimeError("Target val/test loaders both missing.")

                source_shape = infer_input_shape_from_loader(source_train)
                target_shape = infer_input_shape_from_loader(target_train)
                if source_shape is None or target_shape is None:
                    raise RuntimeError("Unable to infer CSI-Bench input shape for transfer.")
                source_spec = DatasetSpec("CSI-Bench", channels=int(source_shape[0]), freq_bins=int(source_shape[1]), time_steps=int(source_shape[2]))
                target_spec = DatasetSpec("CSI-Bench", channels=int(target_shape[0]), freq_bins=int(target_shape[1]), time_steps=int(target_shape[2]))
                source_classes = int(source_bundle["num_classes"])
                target_classes = int(target_bundle["num_classes"])

                save_training_metadata(
                    out_dir,
                    {
                        "stage": "transfer_optuna_csi_bench",
                        "transfer_pair": pair_name,
                        "source_task": source_task,
                        "target_task": target_task,
                        "model_name": model_name,
                        "source_spec": {
                            "channels": source_spec.channels,
                            "freq_bins": source_spec.freq_bins,
                            "time_steps": source_spec.time_steps,
                            "num_classes": source_classes,
                        },
                        "target_spec": {
                            "channels": target_spec.channels,
                            "freq_bins": target_spec.freq_bins,
                            "time_steps": target_spec.time_steps,
                            "num_classes": target_classes,
                        },
                        "source_stage_cfg": source_stage_cfg,
                        "target_stage_cfg": target_stage_cfg,
                        "source_task_metadata": collect_csi_bench_task_metadata(
                            dataset_root=dataset_root,
                            task_name=source_task,
                            splits_dir_name=splits_dir_name,
                            split_names=only_splits,
                        ),
                        "target_task_metadata": collect_csi_bench_task_metadata(
                            dataset_root=dataset_root,
                            task_name=target_task,
                            splits_dir_name=splits_dir_name,
                            split_names=only_splits,
                        ),
                        "optuna": {
                            "n_trials": n_trials,
                            "timeout": timeout,
                            "direction": direction,
                            "objective_metric": objective_metric,
                        },
                    },
                )

                best_weights_path = os.path.join(out_dir, f"{model_token}_best.pt")

                def objective(trial: Trial) -> float:  # type: ignore
                    hparams = suggest_hparams_for_model(trial, model_name, target_spec)
                    source_state = None
                    source_load_stats: Dict[str, Any] = {}
                    try:
                        if base_model_mode in {"source_supervised", "pretrain_source"}:
                            source_model = build_model_from_hparams(source_spec, model_name, source_classes, hparams).to(device)
                            source_summary = _train_model(
                                model=source_model,
                                train_loader=source_train,
                                val_loader=source_val,
                                criterion=criterion,
                                device=device,
                                cfg=source_stage_cfg,
                            )
                            trial.set_user_attr("source_best_epoch", int(source_summary.best_epoch))
                            source_state = {k: v.detach().cpu() for k, v in source_model.state_dict().items()}
                            del source_model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        elif base_model_mode == "source_checkpoint":
                            if not source_checkpoint:
                                raise ValueError("transfer.source_checkpoint is required for base_model_mode=source_checkpoint.")
                            source_state = _load_state_dict_file(str(resolve_project_path(str(source_checkpoint))))
                        elif base_model_mode in {"scratch", "none"}:
                            source_state = None
                        else:
                            raise ValueError(
                                f"Unsupported transfer.base_model_mode '{base_model_mode}'. "
                                "Use source_supervised, source_checkpoint, or scratch."
                            )

                        target_model = build_model_from_hparams(target_spec, model_name, target_classes, hparams).to(device)
                        if source_state is not None:
                            source_load_stats = _load_compatible_weights(target_model, source_state)
                        freeze_stats = _apply_finetune_mode(target_model, finetune_mode)
                        target_summary = _train_model(
                            model=target_model,
                            train_loader=target_train,
                            val_loader=target_val,
                            criterion=criterion,
                            device=device,
                            cfg=target_stage_cfg,
                        )
                        target_metrics = _evaluate_transfer_target(
                            model=target_model,
                            criterion=criterion,
                            loader=eval_loader,
                            device=device,
                            use_tqdm=bool(target_stage_cfg["progress_bar"]),
                            desc=f"transfer_val_{trial.number}",
                        )
                        objective_value = (
                            float(target_metrics["accuracy"])
                            if objective_metric == "val_accuracy"
                            else float(target_metrics["loss"])
                        )
                        trial.set_user_attr("objective_metric", objective_metric)
                        trial.set_user_attr("target_val_accuracy", float(target_metrics["accuracy"]))
                        trial.set_user_attr("target_val_loss", float(target_metrics["loss"]))
                        trial.set_user_attr("target_best_epoch", int(target_summary.best_epoch))
                        trial.set_user_attr("freeze_trainable_params", int(freeze_stats["trainable_params"]))
                        trial.set_user_attr("loaded_source_keys", int(source_load_stats.get("loaded_keys", 0)))
                        if save_trial_weights and (trial.number + 1) % trial_weights_save_every == 0:
                            torch.save(
                                target_model.state_dict(),
                                os.path.join(trial_weights_dir, f"trial_{trial.number:04d}.pt"),
                            )
                        return objective_value

                    except CUDATrialOOMError as exc:
                        trial.set_user_attr("oom", True)
                        raise TrialPruned(str(exc)) from exc
                    except RuntimeError as exc:
                        msg = str(exc).lower()
                        if "out of memory" in msg and ("cuda" in msg or "cudnn" in msg):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            trial.set_user_attr("oom", True)
                            raise TrialPruned("CUDA OOM during transfer trial") from exc
                        raise
                    finally:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                resolved_storage = storage_url or f"sqlite:///{os.path.join(out_dir, 'optuna_study.db')}"
                study_name = f"{study_prefix}_{sanitize_token(pair_name)}_{sanitize_token(model_name)}"
                study = optuna.create_study(
                    study_name=study_name,
                    direction=direction,
                    sampler=sampler,
                    pruner=pruner,
                    storage=resolved_storage,
                    load_if_exists=load_if_exists,
                )
                study.optimize(
                    objective,
                    n_trials=n_trials,
                    timeout=timeout,
                    n_jobs=n_jobs,
                    gc_after_trial=gc_after_trial,
                )

                if save_trial_weights and trial_weights_cleanup:
                    retention_summary = _apply_trial_weight_retention(
                        study=study,
                        trial_weights_dir=trial_weights_dir,
                        direction=direction,
                        mode=trial_weights_mode,
                        top_k=trial_weights_top_k,
                        max_saved=trial_weights_max_saved,
                    )
                    json_dump(os.path.join(trial_weights_dir, "retention_summary.json"), retention_summary)
                    print(
                        "[TRIAL-WEIGHTS] Retention applied: "
                        f"mode={retention_summary.get('mode')} kept={retention_summary.get('kept')} "
                        f"deleted={retention_summary.get('deleted')}"
                    )

                trial_rows: List[Dict[str, Any]] = []
                for trial in study.trials:
                    row: Dict[str, Any] = {
                        "number": trial.number,
                        "state": state_name(trial.state),
                        "value": trial.value,
                        "duration_sec": trial.duration.total_seconds() if trial.duration is not None else None,
                    }
                    for key, value in trial.params.items():
                        row[f"param_{key}"] = value
                    for key, value in trial.user_attrs.items():
                        row[f"attr_{key}"] = value
                    trial_rows.append(row)
                save_results_rows(out_dir, trial_rows, base_name="optuna_trials")

                best_trial = study.best_trial
                best_payload = {
                    "number": best_trial.number,
                    "value": best_trial.value,
                    "params": dict(best_trial.params),
                    "objective_metric": objective_metric,
                    "direction": direction,
                    "study_name": study.study_name,
                    "storage": resolved_storage,
                    "target_val_accuracy": best_trial.user_attrs.get("target_val_accuracy"),
                    "target_val_loss": best_trial.user_attrs.get("target_val_loss"),
                }
                json_dump(os.path.join(out_dir, "best_trial.json"), best_payload)

                best_hparams = dict(best_trial.params)
                source_state = None
                if base_model_mode in {"source_supervised", "pretrain_source"}:
                    source_model = build_model_from_hparams(source_spec, model_name, source_classes, best_hparams).to(device)
                    _ = _train_model(
                        model=source_model,
                        train_loader=source_train,
                        val_loader=source_val,
                        criterion=criterion,
                        device=device,
                        cfg=source_stage_cfg,
                    )
                    source_state = {k: v.detach().cpu() for k, v in source_model.state_dict().items()}
                    del source_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                elif base_model_mode == "source_checkpoint":
                    if not source_checkpoint:
                        raise ValueError("transfer.source_checkpoint is required for base_model_mode=source_checkpoint.")
                    source_state = _load_state_dict_file(str(resolve_project_path(str(source_checkpoint))))

                best_model = build_model_from_hparams(target_spec, model_name, target_classes, best_hparams).to(device)
                if source_state is not None:
                    _ = _load_compatible_weights(best_model, source_state)
                _ = _apply_finetune_mode(best_model, finetune_mode)
                _ = _train_model(
                    model=best_model,
                    train_loader=target_train,
                    val_loader=target_val,
                    criterion=criterion,
                    device=device,
                    cfg=target_stage_cfg,
                )
                torch.save(best_model.state_dict(), best_weights_path)

                final_rows: List[Dict[str, Any]] = []
                for split_name, loader in [("val", target_val), ("test", target_test)]:
                    if loader is None:
                        continue
                    metrics = _evaluate_transfer_target(
                        model=best_model,
                        criterion=criterion,
                        loader=loader,
                        device=device,
                        use_tqdm=bool(target_stage_cfg["progress_bar"]),
                        desc=f"best_{split_name}",
                    )
                    final_rows.append({"split": split_name, **metrics})
                save_results_rows(out_dir, final_rows, base_name="best_eval")

                if export_best_enabled:
                    export_meta = {
                        "domain": "csi_bench",
                        "source_task": source_task,
                        "target_task": target_task,
                        "model": model_name,
                        "best_trial": best_payload,
                    }
                    try:
                        export_result = export_best_weight_to_datasets(
                            source_checkpoint=best_weights_path,
                            datasets_root=export_datasets_root,
                            domain="csi_bench",
                            dataset_name="CSI-Bench",
                            task_name=target_task,
                            model_name=model_name,
                            run_output_dir=out_dir,
                            run_timestamp=ts,
                            config_name=cfg_name,
                            weights_dir_name=export_weights_dir_name,
                            overwrite_strategy=export_overwrite_strategy,
                            include_full_config=export_include_full_config,
                            config_payload=cfg,
                            metadata_payload=export_meta,
                        )
                        print(f"[EXPORT] {export_result}")
                    except Exception as export_exc:
                        print(f"[WARN] Failed to export best checkpoint: {export_exc}")
                        if export_fail_on_error:
                            raise

                completed = True
                print(f"[RUN] Completed: {out_dir}")
            except BaseException as exc:
                print(f"[ERROR] Transfer Optuna CSI-Bench failed for {pair_name}/{model_name}: {exc}")
                traceback.print_exc()
                raise
            finally:
                if not completed:
                    target_dir = move_incomplete_run(out_dir)
                    if target_dir:
                        print(f"[INFO] Moved incomplete run to: {target_dir}")


def main(config_path: Optional[str] = None) -> None:
    if optuna is None:
        raise RuntimeError(f"Optuna is required to run this script: {_OPTUNA_IMPORT_ERROR}")

    cfg = load_config(config_path)
    configure_from_env_and_config(cfg)
    configure_torch_performance(cfg)

    common_cfg = cfg.get("common", {})
    experiment_name = str(common_cfg.get("experiment_name", "sl-transfer-optuna"))
    model_cfg = cfg.get("model", {})
    model_names = resolve_supervised_models(
        as_list(
            model_cfg.get("architectures") or model_cfg.get("models"),
            default=["ResNet18"],
        )
    )
    data_cfg = cfg.get("data", {})
    device = _resolve_device(data_cfg.get("device"))
    data_seed = int(data_cfg.get("seed", 42))

    run_cfg = cfg.get("run", {})
    targets_raw = as_list(run_cfg.get("targets"), default=["ntu", "csi_bench"])
    targets = {token.strip().lower().replace("-", "_") for token in targets_raw}

    if "ntu" in targets:
        _run_ntu_transfer_optuna(
            cfg=cfg,
            config_path=config_path,
            experiment_name=experiment_name,
            model_names=model_names,
            device=device,
            data_seed=data_seed,
        )
    if "csi_bench" in targets:
        _run_csi_transfer_optuna(
            cfg=cfg,
            config_path=config_path,
            experiment_name=experiment_name,
            model_names=model_names,
            device=device,
        )


if __name__ == "__main__":
    cfg_path = os.environ.get("EXP_TRAIN_CONFIG")
    if cfg_path is None and len(sys.argv) > 1:
        cfg_path = resolve_config_path_from_arg(sys.argv[1])
    main(cfg_path)
