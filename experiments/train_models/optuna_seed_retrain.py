"""Retrain Optuna-selected hyperparameters across multiple random seeds.

This runner consumes `optuna_tiebreak_summary.json` (produced by
`scripts/select_optuna_trials.py`) and retrains each selected model config
across N seeds, evaluating both val and test splits per seed.
"""

from __future__ import annotations

import os
import shutil
import statistics
import sys
import traceback
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from csi.data import get_experiment_loaders
from csi.data.utils import filter_none_collate, load_csi_bench_supervised
from csi.models.supervised import train
from csi.models.utils.datasets import DatasetSpec
from csi.models.utils.experiment import load_config, resolve_config_path_from_arg, sanitize_token
from csi.models.utils.gpu_utils import _resolve_device
from csi.models.utils.optimisation import build_model_from_hparams
from csi.models.utils.results import (
    prepare_results_dir,
    save_config,
    save_results_rows,
    save_run_meta,
)
from csi.models.utils.seeding import set_all_seeds
from ..common.shared_utils import (
    DEFAULT_CLEAN_EXPORT_SEED,
    DEFAULT_CLEAN_WEIGHTS_DIR,
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
    infer_spec_from_bundle,
    json_dump,
    resolve_ntu_num_classes,
)
from ..defenses_raw_csi.common_runner import export_csi_checkpoint, export_ntu_checkpoint


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    token = str(value).strip()
    if token == "":
        return token
    low = token.lower()
    if low in {"true", "1", "yes", "y", "on"}:
        return True
    if low in {"false", "0", "no", "n", "off"}:
        return False
    try:
        return int(token)
    except Exception:
        pass
    try:
        return float(token)
    except Exception:
        return token


def _coerce_hparams(model_name: str, params: Mapping[str, Any]) -> Dict[str, Any]:
    int_keys = {
        "hidden_dim",
        "num_layers",
        "base_filters",
        "fc_hidden_dim",
        "pool_size",
        "patch_f",
        "patch_t",
        "emb_dim",
        "depth",
        "num_heads",
    }
    float_keys = {"dropout", "mlp_ratio"}
    bool_keys = {"bidirectional", "pretrained"}

    out: Dict[str, Any] = {}
    for key, raw in params.items():
        value = _coerce_scalar(raw)
        if key in bool_keys:
            out[key] = parse_bool(value, default=False)
            continue
        if key in int_keys:
            try:
                out[key] = int(value)
                continue
            except Exception:
                pass
        if key in float_keys:
            try:
                out[key] = float(value)
                continue
            except Exception:
                pass
        out[key] = value
    return out


def _load_selection(summary_path: str, domain: str) -> List[Dict[str, Any]]:
    payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    studies = payload.get("studies")
    if not isinstance(studies, list):
        raise RuntimeError(f"Invalid summary format (missing studies list): {summary_path}")

    domain_key = str(domain).strip().lower()
    out: List[Dict[str, Any]] = []
    for study in studies:
        if not isinstance(study, Mapping):
            continue
        selected = study.get("selected_trial")
        if not isinstance(selected, Mapping):
            continue
        params = selected.get("params")
        if not isinstance(params, Mapping):
            continue

        dataset = study.get("dataset")
        task_name = study.get("task_name")
        model_name = study.get("model")
        if not model_name:
            continue
        if domain_key == "ntu":
            if not dataset:
                continue
            target_label = str(dataset)
        elif domain_key == "csi_bench":
            if not task_name:
                continue
            target_label = str(task_name)
        else:
            raise ValueError(f"Unsupported domain: {domain}")

        out.append(
            {
                "study_dir": str(study.get("study_dir", "")),
                "dataset": (str(dataset) if dataset is not None else None),
                "task_name": (str(task_name) if task_name is not None else None),
                "model": str(model_name),
                "objective_metric": str(study.get("objective_metric", "val_accuracy")).strip().lower(),
                "direction": str(study.get("direction", "maximize")).strip().lower(),
                "selected_trial_number": selected.get("number"),
                "selected_trial_value": selected.get("value"),
                "params": _coerce_hparams(str(model_name), params),
                "target_label": target_label,
            }
        )

    out.sort(key=lambda row: (str(row.get("target_label", "")), str(row.get("model", ""))))
    return out


def _coerce_seed_list(value: Any) -> List[int]:
    if value is None:
        return [42, 11, 23, 37, 53]
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        return [int(token) for token in tokens]
    if isinstance(value, (list, tuple)):
        return [int(token) for token in value]
    return [int(value)]


def _rank_key(record: Mapping[str, Any], objective_metric: str, direction: str) -> Tuple[Any, ...]:
    val_acc = record.get("val_accuracy")
    val_loss = record.get("val_loss")
    seed = int(record.get("seed", 10**9))
    acc = float(val_acc) if val_acc is not None else float("-inf")
    loss = float(val_loss) if val_loss is not None else float("inf")

    if objective_metric == "val_loss" or direction == "minimize":
        return (loss, -acc, seed)
    return (-acc, loss, seed)


def _mean_std(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def _aggregate_seed_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    val_acc_mean, val_acc_std = _mean_std(row.get("val_accuracy") for row in records)
    val_loss_mean, val_loss_std = _mean_std(row.get("val_loss") for row in records)
    test_acc_mean, test_acc_std = _mean_std(row.get("test_accuracy") for row in records)
    test_loss_mean, test_loss_std = _mean_std(row.get("test_loss") for row in records)
    return {
        "n_seeds": len(records),
        "val_accuracy_mean": val_acc_mean,
        "val_accuracy_std": val_acc_std,
        "val_loss_mean": val_loss_mean,
        "val_loss_std": val_loss_std,
        "test_accuracy_mean": test_acc_mean,
        "test_accuracy_std": test_acc_std,
        "test_loss_mean": test_loss_mean,
        "test_loss_std": test_loss_std,
    }


def _extract_eval_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    val_row = next((row for row in rows if str(row.get("split")) == "val"), None)
    test_row = next((row for row in rows if str(row.get("split")) == "test"), None)
    return {
        "val_accuracy": (float(val_row.get("accuracy")) if val_row and val_row.get("accuracy") is not None else None),  # type: ignore
        "val_loss": (float(val_row.get("loss")) if val_row and val_row.get("loss") is not None else None),  # type: ignore
        "test_accuracy": (float(test_row.get("accuracy")) if test_row and test_row.get("accuracy") is not None else None),  # type: ignore
        "test_loss": (float(test_row.get("loss")) if test_row and test_row.get("loss") is not None else None),  # type: ignore
    }


def _resolve_export_cfg(
    cfg: Mapping[str, Any],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    export_cfg = dict(cfg.get("export", {}) or {})
    enabled = parse_bool(export_cfg.get("enabled"), default=True)
    clean_weights_dir_name = str(export_cfg.get("clean_weights_dir_name", DEFAULT_CLEAN_WEIGHTS_DIR)).strip()
    canonical_seed = int(export_cfg.get("canonical_seed", DEFAULT_CLEAN_EXPORT_SEED))
    latest_alias = parse_bool(export_cfg.get("latest_alias"), default=True)
    if enabled and canonical_seed not in {int(seed) for seed in seeds}:
        raise ValueError(
            f"export.canonical_seed={canonical_seed} is not present in run.seeds={list(seeds)}. "
            "Include the canonical seed or disable export.enabled."
        )
    return {
        "enabled": enabled,
        "clean_weights_dir_name": clean_weights_dir_name,
        "canonical_seed": canonical_seed,
        "latest_alias": latest_alias,
        "datasets_root": export_cfg.get("datasets_root"),
        "dataset_root": export_cfg.get("dataset_root"),
    }


def _copy_seed_weight(
    checkpoint_path: str,
    *,
    model_scope_dir: Path,
    seed: int,
) -> str:
    seed_dir = model_scope_dir / "seed_weights"
    seed_dir.mkdir(parents=True, exist_ok=True)
    target = seed_dir / f"seed_{int(seed):04d}.pt"
    shutil.copy2(checkpoint_path, target)
    return str(target)


def _write_target_seed_summary(
    *,
    model_scope_dir: Path,
    target_info: Mapping[str, Any],
    seed_records: Sequence[Mapping[str, Any]],
    best_record: Mapping[str, Any],
) -> None:
    save_results_rows(str(model_scope_dir), [dict(row) for row in seed_records], base_name="seed_runs")
    payload = {
        "target": dict(target_info),
        "n_seeds": len(seed_records),
        "aggregate": _aggregate_seed_records(seed_records),
        "best_seed": dict(best_record),
        "seed_records": [dict(row) for row in seed_records],
    }
    json_dump(str(model_scope_dir / "seed_runs_summary.json"), payload)


def _filter_targets(targets: Sequence[Mapping[str, Any]], run_cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    include_models = {str(token).strip().lower() for token in as_list(run_cfg.get("include_models"), default=[])}
    include_datasets = {str(token).strip() for token in as_list(run_cfg.get("include_datasets"), default=[])}
    include_tasks = {str(token).strip() for token in as_list(run_cfg.get("include_tasks"), default=[])}

    out: List[Dict[str, Any]] = []
    for row in targets:
        model_name = str(row.get("model", ""))
        dataset = row.get("dataset")
        task_name = row.get("task_name")
        if include_models and model_name.strip().lower() not in include_models:
            continue
        if include_datasets and dataset not in include_datasets:
            continue
        if include_tasks and task_name not in include_tasks:
            continue
        out.append(dict(row))
    return out


def _campaign_scope_tag(run_cfg: Mapping[str, Any]) -> str:
    models = as_list(run_cfg.get("include_models"), default=[])
    datasets = as_list(run_cfg.get("include_datasets"), default=[])
    tasks = as_list(run_cfg.get("include_tasks"), default=[])
    tokens: List[str] = []
    if models:
        tokens.append("models_" + "_".join(sanitize_token(token) for token in models))
    if datasets:
        tokens.append("datasets_" + "_".join(sanitize_token(token) for token in datasets))
    if tasks:
        tokens.append("tasks_" + "_".join(sanitize_token(token) for token in tasks))
    if not tokens:
        return "all_targets"
    return "__".join(tokens)


def _run_ntu_target(
    *,
    cfg: Mapping[str, Any],
    config_path: Optional[str],
    experiment_name: str,
    target: Mapping[str, Any],
    seeds: Sequence[int],
    device: torch.device,
    deterministic: bool,
    require_test_split: bool,
    export_cfg: Mapping[str, Any],
) -> Tuple[Path, List[Dict[str, Any]], Dict[str, Any]]:
    data_cfg = cfg.get("data", {}) or {}
    train_cfg = cfg.get("training", {}) or {}
    root = str(data_cfg.get("root", cfg.get("common", {}).get("root", "datasets")))
    task = str(data_cfg.get("task", "classification"))
    timesteps = int(data_cfg.get("timesteps", 250))
    batch_sizes = data_cfg.get("batch_sizes")
    base_normalisation = str(data_cfg.get("base_normalisation", data_cfg.get("base_normalization", "none")))
    canon_normalisation = str(data_cfg.get("canon_normalisation", data_cfg.get("canonical_normalization", "per_subcarrier")))
    variant = str(data_cfg.get("variant", "raw"))
    smoothing = data_cfg.get("smoothing")
    num_workers_raw = data_cfg.get("num_workers")
    num_workers = int(num_workers_raw) if num_workers_raw is not None else None
    pin_memory_raw = data_cfg.get("pin_memory")
    pin_memory = None if pin_memory_raw is None else parse_bool(pin_memory_raw, default=(device.type == "cuda"))
    persistent_workers_raw = data_cfg.get("persistent_workers")
    persistent_workers = None if persistent_workers_raw is None else parse_bool(persistent_workers_raw, default=True)
    prefetch_factor_raw = data_cfg.get("prefetch_factor")
    prefetch_factor = int(prefetch_factor_raw) if prefetch_factor_raw is not None else None
    augment_train = dict(data_cfg.get("augment", {}) or {})
    augment_eval = dict(data_cfg.get("augment_eval", augment_train) or augment_train)

    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    epochs = int(train_cfg.get("epochs", 80))
    progress_bar = parse_bool(train_cfg.get("progress_bar"), default=False)
    patience_raw = train_cfg.get("patience")
    patience = int(patience_raw) if patience_raw is not None else None
    min_epochs = int(train_cfg.get("min_epochs", 1))
    monitor_metric = str(train_cfg.get("monitor_metric", "accuracy"))
    min_delta = float(train_cfg.get("min_delta", 0.0))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip_raw = train_cfg.get("grad_clip")
    grad_clip = float(grad_clip_raw) if grad_clip_raw is not None else None
    optimizer_cfg = train_cfg.get("optimizer")
    scheduler_cfg = train_cfg.get("scheduler")
    amp = train_cfg.get("amp")
    amp_dtype = str(train_cfg.get("amp_dtype", "float16"))
    log_every_epochs_raw = train_cfg.get("log_every_epochs")
    log_every_epochs = int(log_every_epochs_raw) if log_every_epochs_raw is not None else None

    dataset_name = str(target["dataset"])
    model_name = str(target["model"])
    hparams = dict(target["params"])
    criterion = nn.CrossEntropyLoss()

    seed_records: List[Dict[str, Any]] = []
    model_scope_dir: Optional[Path] = None
    for seed in seeds:
        set_all_seeds(int(seed), deterministic=deterministic)
        model_token = sanitize_token(model_name)
        out_dir, ts, cfg_name = prepare_results_dir(
            experiment_name=experiment_name,
            cfg_path=config_path,
            relative_parts=["ntu", dataset_name, model_token, f"seed_{int(seed):04d}"],
        )
        model_scope_dir = Path(out_dir).parent
        completed = False
        try:
            print_training_start(
                f"Optuna seed retrain NTU ({dataset_name}/{model_name})",
                {
                    "output_dir": out_dir,
                    "seed": int(seed),
                    "selected_trial": target.get("selected_trial_number"),
                    "device": str(device),
                },
            )
            save_run_meta(
                out_dir,
                {
                    "stage": "optuna_seed_retrain",
                    "domain": "ntu",
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": int(seed),
                    "timestamp": ts,
                    "config_name": cfg_name,
                    "root": root,
                    "objective_metric": target.get("objective_metric"),
                    "direction": target.get("direction"),
                    "selected_trial_number": target.get("selected_trial_number"),
                    "selected_trial_value": target.get("selected_trial_value"),
                    "source_study_dir": target.get("study_dir"),
                },
            )
            save_config(out_dir, dict(cfg))

            bundle = get_experiment_loaders(
                dataset_name,
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
                seed=int(seed),
            )
            train_loader = bundle.train_loader
            val_loader = bundle.val_loader
            test_loader = bundle.test_loader
            if train_loader is None:
                raise RuntimeError(f"Train loader unavailable for dataset {dataset_name}")
            if require_test_split and test_loader is None:
                raise RuntimeError(f"Test loader unavailable for dataset {dataset_name}; require_test_split=true.")
            if val_loader is None and test_loader is None:
                raise RuntimeError(f"Both val/test loaders are missing for dataset {dataset_name}")

            bundle_info = bundle.info or {}
            spec = infer_spec_from_bundle(dataset_name, bundle_info, train_loader)
            num_classes = resolve_ntu_num_classes(bundle_info, train_loader)
            model = build_model_from_hparams(spec, model_name, num_classes, hparams).to(device)

            _ = train(
                model=model,
                tensor_loader=train_loader,
                num_epochs=epochs,
                learning_rate=learning_rate,
                criterion=criterion,
                device=device,
                use_tqdm=progress_bar,
                val_loader=val_loader,
                patience=patience if val_loader is not None else None,
                min_epochs=min_epochs,
                monitor_metric=monitor_metric,
                min_delta=min_delta,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
                optimizer_cfg=optimizer_cfg,
                scheduler_cfg=scheduler_cfg,
                amp=amp,
                amp_dtype=amp_dtype,
                log_every_epochs=log_every_epochs,
            )

            eval_rows = evaluate_splits(
                model=model,
                criterion=criterion,
                device=device,
                use_tqdm=progress_bar,
                split_loaders={"val": val_loader, "test": test_loader},
                stage="seed_retrain",
            )
            for row in eval_rows:
                row["domain"] = "ntu"
                row["dataset"] = dataset_name
                row["model"] = model_name
                row["seed"] = int(seed)
                row["selected_trial_number"] = target.get("selected_trial_number")
                row["selected_trial_value"] = target.get("selected_trial_value")
            save_results_rows(out_dir, eval_rows, base_name="summary_rows")

            checkpoint_path = os.path.join(out_dir, f"{model_token}_seed{int(seed):04d}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            copied_seed_ckpt = _copy_seed_weight(checkpoint_path, model_scope_dir=model_scope_dir, seed=int(seed))
            exported_clean_paths: Dict[str, str] = {}
            if export_cfg.get("enabled") and int(seed) == int(export_cfg.get("canonical_seed", DEFAULT_CLEAN_EXPORT_SEED)):
                exported_clean_paths = export_ntu_checkpoint(
                    datasets_root=str(export_cfg.get("datasets_root") or root),
                    export_weights_dir_name=str(export_cfg["clean_weights_dir_name"]),
                    dataset_name=dataset_name,
                    model_name=model_name,
                    source_checkpoint=checkpoint_path,
                    run_timestamp=ts,
                    run_dir=out_dir,
                    config_path=config_path,
                    stage="optuna_seed_retrain_clean_export",
                    latest_alias=bool(export_cfg.get("latest_alias", True)),
                )

            metrics = _extract_eval_metrics(eval_rows)
            seed_record: Dict[str, Any] = {
                "domain": "ntu",
                "dataset": dataset_name,
                "task_name": None,
                "model": model_name,
                "seed": int(seed),
                "objective_metric": str(target.get("objective_metric", "val_accuracy")),
                "direction": str(target.get("direction", "maximize")),
                "selected_trial_number": target.get("selected_trial_number"),
                "selected_trial_value": target.get("selected_trial_value"),
                "val_accuracy": metrics["val_accuracy"],
                "val_loss": metrics["val_loss"],
                "test_accuracy": metrics["test_accuracy"],
                "test_loss": metrics["test_loss"],
                "run_dir": out_dir,
                "weights_path": checkpoint_path,
                "weights_copy_path": copied_seed_ckpt,
                "clean_weights_dir_name": str(export_cfg.get("clean_weights_dir_name", DEFAULT_CLEAN_WEIGHTS_DIR)),
                "canonical_seed": int(export_cfg.get("canonical_seed", DEFAULT_CLEAN_EXPORT_SEED)),
                "clean_export_timestamped_path": exported_clean_paths.get("timestamped_path"),
                "clean_export_latest_path": exported_clean_paths.get("latest_path"),
            }
            seed_records.append(seed_record)

            save_training_metadata(
                out_dir,
                {
                    "stage": "optuna_seed_retrain_ntu",
                    "dataset": dataset_name,
                    "model_name": model_name,
                    "seed": int(seed),
                    "spec": {
                        "channels": spec.channels,
                        "freq_bins": spec.freq_bins,
                        "time_steps": spec.time_steps,
                        "num_classes": num_classes,
                    },
                    "hparams": hparams,
                    "weights_path": checkpoint_path,
                    "weights_copy_path": copied_seed_ckpt,
                    "clean_export_timestamped_path": exported_clean_paths.get("timestamped_path"),
                    "clean_export_latest_path": exported_clean_paths.get("latest_path"),
                },
            )
            completed = True
            print(f"[RUN] Completed: {out_dir}")
        except BaseException as exc:
            print(f"[ERROR] NTU seed retrain failed for {dataset_name}/{model_name}/seed={seed}: {exc}")
            traceback.print_exc()
            raise
        finally:
            if not completed:
                target_dir = move_incomplete_run(out_dir)
                if target_dir:
                    print(f"[INFO] Moved incomplete run to: {target_dir}")

    assert model_scope_dir is not None
    ranked = sorted(
        seed_records,
        key=lambda row: _rank_key(
            row,
            objective_metric=str(target.get("objective_metric", "val_accuracy")),
            direction=str(target.get("direction", "maximize")),
        ),
    )
    best_record = dict(ranked[0])
    best_seed = int(best_record["seed"])
    best_src = Path(str(best_record["weights_copy_path"]))
    best_dir = model_scope_dir / "seed_weights"
    best_dst = best_dir / "best.pt"
    best_seed_dst = best_dir / f"best_seed_{best_seed:04d}.pt"
    shutil.copy2(best_src, best_dst)
    shutil.copy2(best_src, best_seed_dst)
    best_record["best_alias_path"] = str(best_dst)
    best_record["best_seed_alias_path"] = str(best_seed_dst)
    _write_target_seed_summary(
        model_scope_dir=model_scope_dir,
        target_info={
            "domain": "ntu",
            "dataset": dataset_name,
            "task_name": None,
            "model": model_name,
            "objective_metric": target.get("objective_metric"),
            "direction": target.get("direction"),
            "selected_trial_number": target.get("selected_trial_number"),
            "selected_trial_value": target.get("selected_trial_value"),
            "study_dir": target.get("study_dir"),
            "params": hparams,
        },
        seed_records=seed_records,
        best_record=best_record,
    )
    print(f"[INFO] Best seed for {dataset_name}/{model_name}: {best_seed} -> {best_dst}")
    return model_scope_dir, seed_records, best_record


def _run_csi_target(
    *,
    cfg: Mapping[str, Any],
    config_path: Optional[str],
    experiment_name: str,
    target: Mapping[str, Any],
    seeds: Sequence[int],
    device: torch.device,
    deterministic: bool,
    require_test_split: bool,
    export_cfg: Mapping[str, Any],
) -> Tuple[Path, List[Dict[str, Any]], Dict[str, Any]]:
    data_cfg = cfg.get("data", {}) or {}
    train_cfg = cfg.get("training", {}) or {}
    dataset_root = str(data_cfg.get("dataset_root", cfg.get("common", {}).get("dataset_root", "datasets/CSI-Bench")))
    normalisation = str(data_cfg.get("normalisation", "zscore"))
    splits_dir_name = str(data_cfg.get("splits_dir", "splits"))
    target_subcarriers = int(data_cfg.get("target_subcarriers", DEFAULT_CSI_BENCH_SUBCARRIERS))
    target_timesteps = int(data_cfg.get("target_timesteps", DEFAULT_CSI_BENCH_TIMESTEPS))
    only_splits = parse_splits(data_cfg.get("only_splits") or data_cfg.get("splits"), default=["train_id", "val_id", "test_id"])
    batch_size = int(data_cfg.get("batch_size", 256))
    num_workers_raw = data_cfg.get("num_workers")
    num_workers = int(num_workers_raw) if num_workers_raw is not None else None
    pin_memory_raw = data_cfg.get("pin_memory")
    pin_memory = None if pin_memory_raw is None else parse_bool(pin_memory_raw, default=(device.type == "cuda"))
    persistent_workers_raw = data_cfg.get("persistent_workers")
    persistent_workers = None if persistent_workers_raw is None else parse_bool(persistent_workers_raw, default=True)
    prefetch_factor_raw = data_cfg.get("prefetch_factor")
    prefetch_factor = int(prefetch_factor_raw) if prefetch_factor_raw is not None else None

    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    epochs = int(train_cfg.get("epochs", 80))
    progress_bar = parse_bool(train_cfg.get("progress_bar"), default=False)
    patience_raw = train_cfg.get("patience")
    patience = int(patience_raw) if patience_raw is not None else None
    min_epochs = int(train_cfg.get("min_epochs", 1))
    monitor_metric = str(train_cfg.get("monitor_metric", "accuracy"))
    min_delta = float(train_cfg.get("min_delta", 0.0))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip_raw = train_cfg.get("grad_clip")
    grad_clip = float(grad_clip_raw) if grad_clip_raw is not None else None
    optimizer_cfg = train_cfg.get("optimizer")
    scheduler_cfg = train_cfg.get("scheduler")
    amp = train_cfg.get("amp")
    amp_dtype = str(train_cfg.get("amp_dtype", "float16"))
    log_every_epochs_raw = train_cfg.get("log_every_epochs")
    log_every_epochs = int(log_every_epochs_raw) if log_every_epochs_raw is not None else None

    task_name = normalize_csi_bench_task_name(str(target["task_name"]))
    model_name = str(target["model"])
    hparams = dict(target["params"])
    criterion = nn.CrossEntropyLoss()

    seed_records: List[Dict[str, Any]] = []
    model_scope_dir: Optional[Path] = None
    for seed in seeds:
        set_all_seeds(int(seed), deterministic=deterministic)
        model_token = sanitize_token(model_name)
        out_dir, ts, cfg_name = prepare_results_dir(
            experiment_name=experiment_name,
            cfg_path=config_path,
            relative_parts=["csi_bench", task_name, model_token, f"seed_{int(seed):04d}"],
        )
        model_scope_dir = Path(out_dir).parent
        completed = False
        try:
            print_training_start(
                f"Optuna seed retrain CSI-Bench ({task_name}/{model_name})",
                {
                    "output_dir": out_dir,
                    "seed": int(seed),
                    "selected_trial": target.get("selected_trial_number"),
                    "device": str(device),
                },
            )
            save_run_meta(
                out_dir,
                {
                    "stage": "optuna_seed_retrain",
                    "domain": "csi_bench",
                    "dataset": "csi-bench",
                    "task_name": task_name,
                    "model": model_name,
                    "seed": int(seed),
                    "timestamp": ts,
                    "config_name": cfg_name,
                    "dataset_root": dataset_root,
                    "splits": only_splits,
                    "objective_metric": target.get("objective_metric"),
                    "direction": target.get("direction"),
                    "selected_trial_number": target.get("selected_trial_number"),
                    "selected_trial_value": target.get("selected_trial_value"),
                    "source_study_dir": target.get("study_dir"),
                },
            )
            save_config(out_dir, dict(cfg))

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
                raise RuntimeError(f"Failed to load benchmark data for task {task_name}")

            loaders = bundle["loaders"]
            train_loader = loaders.get("train")
            val_loader = loaders.get("val")
            test_loader = loaders.get("test")
            if train_loader is None:
                raise RuntimeError(f"Training loader missing for task {task_name}")
            if require_test_split and test_loader is None:
                raise RuntimeError(f"Test loader unavailable for task {task_name}; require_test_split=true.")
            if val_loader is None and test_loader is None:
                raise RuntimeError(f"Validation and test loaders are both missing for task {task_name}")

            inferred_shape = infer_input_shape_from_loader(train_loader)
            if inferred_shape is None:
                raise RuntimeError("Unable to infer (channels, freq_bins, timesteps) from train loader.")
            channels, freq_bins, time_steps = inferred_shape
            spec = DatasetSpec("CSI-Bench", channels=channels, freq_bins=freq_bins, time_steps=time_steps)
            num_classes = int(bundle["num_classes"])
            model = build_model_from_hparams(spec, model_name, num_classes, hparams).to(device)

            _ = train(
                model=model,
                tensor_loader=train_loader,
                num_epochs=epochs,
                learning_rate=learning_rate,
                criterion=criterion,
                device=device,
                use_tqdm=progress_bar,
                val_loader=val_loader,
                patience=patience if val_loader is not None else None,
                min_epochs=min_epochs,
                monitor_metric=monitor_metric,
                min_delta=min_delta,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
                optimizer_cfg=optimizer_cfg,
                scheduler_cfg=scheduler_cfg,
                amp=amp,
                amp_dtype=amp_dtype,
                log_every_epochs=log_every_epochs,
            )

            eval_rows = evaluate_splits(
                model=model,
                criterion=criterion,
                device=device,
                use_tqdm=progress_bar,
                split_loaders={"val": val_loader, "test": test_loader},
                stage="seed_retrain",
            )
            for row in eval_rows:
                row["domain"] = "csi_bench"
                row["dataset"] = "csi-bench"
                row["task_name"] = task_name
                row["model"] = model_name
                row["seed"] = int(seed)
                row["selected_trial_number"] = target.get("selected_trial_number")
                row["selected_trial_value"] = target.get("selected_trial_value")
            save_results_rows(out_dir, eval_rows, base_name="summary_rows")

            checkpoint_path = os.path.join(out_dir, f"{model_token}_seed{int(seed):04d}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            copied_seed_ckpt = _copy_seed_weight(checkpoint_path, model_scope_dir=model_scope_dir, seed=int(seed))
            exported_clean_paths: Dict[str, str] = {}
            if export_cfg.get("enabled") and int(seed) == int(export_cfg.get("canonical_seed", DEFAULT_CLEAN_EXPORT_SEED)):
                exported_clean_paths = export_csi_checkpoint(
                    dataset_root=str(export_cfg.get("dataset_root") or dataset_root),
                    export_weights_dir_name=str(export_cfg["clean_weights_dir_name"]),
                    task_name=task_name,
                    model_name=model_name,
                    source_checkpoint=checkpoint_path,
                    run_timestamp=ts,
                    run_dir=out_dir,
                    config_path=config_path,
                    stage="optuna_seed_retrain_clean_export",
                    latest_alias=bool(export_cfg.get("latest_alias", True)),
                )

            metrics = _extract_eval_metrics(eval_rows)
            seed_record: Dict[str, Any] = {
                "domain": "csi_bench",
                "dataset": "csi-bench",
                "task_name": task_name,
                "model": model_name,
                "seed": int(seed),
                "objective_metric": str(target.get("objective_metric", "val_accuracy")),
                "direction": str(target.get("direction", "maximize")),
                "selected_trial_number": target.get("selected_trial_number"),
                "selected_trial_value": target.get("selected_trial_value"),
                "val_accuracy": metrics["val_accuracy"],
                "val_loss": metrics["val_loss"],
                "test_accuracy": metrics["test_accuracy"],
                "test_loss": metrics["test_loss"],
                "run_dir": out_dir,
                "weights_path": checkpoint_path,
                "weights_copy_path": copied_seed_ckpt,
                "clean_weights_dir_name": str(export_cfg.get("clean_weights_dir_name", DEFAULT_CLEAN_WEIGHTS_DIR)),
                "canonical_seed": int(export_cfg.get("canonical_seed", DEFAULT_CLEAN_EXPORT_SEED)),
                "clean_export_timestamped_path": exported_clean_paths.get("timestamped_path"),
                "clean_export_latest_path": exported_clean_paths.get("latest_path"),
            }
            seed_records.append(seed_record)

            save_training_metadata(
                out_dir,
                {
                    "stage": "optuna_seed_retrain_csi_bench",
                    "task_name": task_name,
                    "model_name": model_name,
                    "seed": int(seed),
                    "spec": {
                        "channels": spec.channels,
                        "freq_bins": spec.freq_bins,
                        "time_steps": spec.time_steps,
                        "num_classes": num_classes,
                    },
                    "hparams": hparams,
                    "weights_path": checkpoint_path,
                    "weights_copy_path": copied_seed_ckpt,
                    "clean_export_timestamped_path": exported_clean_paths.get("timestamped_path"),
                    "clean_export_latest_path": exported_clean_paths.get("latest_path"),
                },
            )
            completed = True
            print(f"[RUN] Completed: {out_dir}")
        except BaseException as exc:
            print(f"[ERROR] CSI-Bench seed retrain failed for {task_name}/{model_name}/seed={seed}: {exc}")
            traceback.print_exc()
            raise
        finally:
            if not completed:
                target_dir = move_incomplete_run(out_dir)
                if target_dir:
                    print(f"[INFO] Moved incomplete run to: {target_dir}")

    assert model_scope_dir is not None
    ranked = sorted(
        seed_records,
        key=lambda row: _rank_key(
            row,
            objective_metric=str(target.get("objective_metric", "val_accuracy")),
            direction=str(target.get("direction", "maximize")),
        ),
    )
    best_record = dict(ranked[0])
    best_seed = int(best_record["seed"])
    best_src = Path(str(best_record["weights_copy_path"]))
    best_dir = model_scope_dir / "seed_weights"
    best_dst = best_dir / "best.pt"
    best_seed_dst = best_dir / f"best_seed_{best_seed:04d}.pt"
    shutil.copy2(best_src, best_dst)
    shutil.copy2(best_src, best_seed_dst)
    best_record["best_alias_path"] = str(best_dst)
    best_record["best_seed_alias_path"] = str(best_seed_dst)
    _write_target_seed_summary(
        model_scope_dir=model_scope_dir,
        target_info={
            "domain": "csi_bench",
            "dataset": "csi-bench",
            "task_name": task_name,
            "model": model_name,
            "objective_metric": target.get("objective_metric"),
            "direction": target.get("direction"),
            "selected_trial_number": target.get("selected_trial_number"),
            "selected_trial_value": target.get("selected_trial_value"),
            "study_dir": target.get("study_dir"),
            "params": hparams,
        },
        seed_records=seed_records,
        best_record=best_record,
    )
    print(f"[INFO] Best seed for {task_name}/{model_name}: {best_seed} -> {best_dst}")
    return model_scope_dir, seed_records, best_record


def main(config_path: Optional[str] = None) -> None:
    cfg = load_config(config_path)
    configure_torch_performance(cfg)

    common_cfg = cfg.get("common", {}) or {}
    run_cfg = cfg.get("run", {}) or {}
    selection_cfg = cfg.get("selection", {}) or {}
    experiment_name = str(common_cfg.get("experiment_name", "optuna-seed-retrain"))

    summary_json = selection_cfg.get("summary_json")
    if not summary_json:
        raise ValueError("selection.summary_json is required.")

    domain = str(run_cfg.get("domain", "")).strip().lower()
    if domain not in {"ntu", "csi_bench"}:
        raise ValueError("run.domain must be one of: ntu, csi_bench")

    seeds = _coerce_seed_list(run_cfg.get("seeds"))
    if not seeds:
        raise ValueError("run.seeds must contain at least one seed.")
    deterministic = parse_bool(run_cfg.get("deterministic", os.environ.get("DETERMINISTIC")), default=False)
    require_test_split = parse_bool(run_cfg.get("require_test_split"), default=True)
    continue_on_error = parse_bool(run_cfg.get("continue_on_error"), default=False)
    export_cfg = _resolve_export_cfg(cfg, seeds)

    data_cfg = cfg.get("data", {}) or {}
    device = _resolve_device(data_cfg.get("device"))

    targets = _load_selection(str(summary_json), domain)
    targets = _filter_targets(targets, run_cfg)
    if not targets:
        raise RuntimeError(f"No selectable targets found in summary for domain={domain}: {summary_json}")

    print_training_start(
        "Optuna seed retrain campaign",
        {
            "summary_json": str(summary_json),
            "domain": domain,
            "num_targets": len(targets),
            "seeds": [int(seed) for seed in seeds],
            "deterministic": deterministic,
            "require_test_split": require_test_split,
            "clean_weights_dir_name": export_cfg.get("clean_weights_dir_name"),
            "canonical_seed": int(export_cfg.get("canonical_seed", DEFAULT_CLEAN_EXPORT_SEED)),
            "device": str(device),
        },
    )

    campaign_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for target in targets:
        try:
            if domain == "ntu":
                _, seed_records, best_record = _run_ntu_target(
                    cfg=cfg,
                    config_path=config_path,
                    experiment_name=experiment_name,
                    target=target,
                    seeds=seeds,
                    device=device,
                    deterministic=deterministic,
                    require_test_split=require_test_split,
                    export_cfg=export_cfg,
                )
            else:
                _, seed_records, best_record = _run_csi_target(
                    cfg=cfg,
                    config_path=config_path,
                    experiment_name=experiment_name,
                    target=target,
                    seeds=seeds,
                    device=device,
                    deterministic=deterministic,
                    require_test_split=require_test_split,
                    export_cfg=export_cfg,
                )
            campaign_rows.extend(seed_records)
            campaign_rows.append(
                {
                    "domain": best_record.get("domain"),
                    "dataset": best_record.get("dataset"),
                    "task_name": best_record.get("task_name"),
                    "model": best_record.get("model"),
                    "seed": best_record.get("seed"),
                    "role": "best_seed",
                    "val_accuracy": best_record.get("val_accuracy"),
                    "val_loss": best_record.get("val_loss"),
                    "test_accuracy": best_record.get("test_accuracy"),
                    "test_loss": best_record.get("test_loss"),
                    "weights_copy_path": best_record.get("weights_copy_path"),
                    "best_alias_path": best_record.get("best_alias_path"),
                }
            )
        except BaseException as exc:
            failure = {
                "domain": domain,
                "dataset": target.get("dataset"),
                "task_name": target.get("task_name"),
                "model": target.get("model"),
                "error": str(exc),
            }
            failures.append(failure)
            if continue_on_error:
                print(f"[WARN] Continuing after failure: {failure}")
                continue
            raise

    if campaign_rows:
        scope_tag = _campaign_scope_tag(run_cfg)
        out_dir, _, _ = prepare_results_dir(
            experiment_name=experiment_name,
            cfg_path=config_path,
            relative_parts=[domain, "campaign_summary", scope_tag],
        )
        save_results_rows(out_dir, campaign_rows, base_name="seed_campaign_rows")
        json_dump(
            os.path.join(out_dir, "seed_campaign_summary.json"),
            {
                "stage": "optuna_seed_retrain_campaign",
                "domain": domain,
                "summary_json": str(summary_json),
                "scope_tag": scope_tag,
                "num_targets": len(targets),
                "seeds": [int(seed) for seed in seeds],
                "clean_weights_dir_name": export_cfg.get("clean_weights_dir_name"),
                "canonical_seed": int(export_cfg.get("canonical_seed", DEFAULT_CLEAN_EXPORT_SEED)),
                "num_failures": len(failures),
                "failures": failures,
            },
        )
        print(f"[INFO] Campaign summary written: {out_dir}")


if __name__ == "__main__":
    cfg_path = os.environ.get("EXP_TRAIN_CONFIG")
    if cfg_path is None and len(sys.argv) > 1:
        cfg_path = resolve_config_path_from_arg(sys.argv[1])
    main(cfg_path)
