"""Train quick clean baselines for NTU and CSI-Bench (HAR/HID)."""

from __future__ import annotations

import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from csi.data import get_experiment_loaders
from csi.data.utils import filter_none_collate, load_csi_bench_supervised
from csi.models.model_factory import build_model
from csi.models.supervised import train
from csi.models.utils.experiment import load_config, resolve_config_path_from_arg, sanitize_token
from csi.models.utils.gpu_utils import _resolve_device
from csi.models.utils.results import (
    prepare_results_dir,
    save_config,
    save_results_rows,
    save_run_meta,
)
from csi.models.utils.seeding import configure_from_env_and_config
from ..common.shared_utils import (
    collect_csi_bench_task_metadata,
    history_rows_supervised,
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
    write_json_file,
)


def _export_ntu_checkpoint(
    *,
    datasets_root: str,
    weights_dir_name: str,
    dataset_name: str,
    model_name: str,
    source_checkpoint: str,
    source_out_dir: str,
    config_path: Optional[str],
    run_timestamp: Optional[str],
) -> str:
    root = resolve_project_path(datasets_root)
    export_dir = root / dataset_name / weights_dir_name
    export_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_name}_{model_name}"
    ts_token = sanitize_token(run_timestamp or "") or sanitize_token(Path(source_out_dir).name) or "run"
    export_path = export_dir / f"{stem}__{ts_token}.pt"
    latest_path = export_dir / f"{stem}.pt"

    shutil.copy2(source_checkpoint, export_path)
    shutil.copy2(source_checkpoint, latest_path)
    write_json_file(
        Path(f"{export_path}.meta.json"),
        {
            "stage": "sl_trial_training_export",
            "domain": "ntu",
            "dataset": dataset_name,
            "model": model_name,
            "variant": "timestamped",
            "run_timestamp": run_timestamp,
            "latest_path": str(latest_path),
            "source_checkpoint": str(Path(source_checkpoint).resolve()),
            "source_run_dir": str(Path(source_out_dir).resolve()),
            "config_path": str(Path(config_path).resolve()) if config_path else None,
        },
    )
    write_json_file(
        Path(f"{latest_path}.meta.json"),
        {
            "stage": "sl_trial_training_export",
            "domain": "ntu",
            "dataset": dataset_name,
            "model": model_name,
            "variant": "latest_alias",
            "run_timestamp": run_timestamp,
            "timestamped_path": str(export_path),
            "source_checkpoint": str(Path(source_checkpoint).resolve()),
            "source_run_dir": str(Path(source_out_dir).resolve()),
            "config_path": str(Path(config_path).resolve()) if config_path else None,
        },
    )
    return str(export_path)


def _export_csi_bench_checkpoint(
    *,
    datasets_root: str,
    weights_dir_name: str,
    task_name: str,
    model_name: str,
    source_checkpoint: str,
    source_out_dir: str,
    config_path: Optional[str],
    run_timestamp: Optional[str],
) -> str:
    root = resolve_project_path(datasets_root)
    csi_root = root if root.name == "CSI-Bench" else (root / "CSI-Bench")
    export_dir = csi_root / weights_dir_name / task_name
    export_dir.mkdir(parents=True, exist_ok=True)
    stem = f"CSI-Bench_{model_name}"
    ts_token = sanitize_token(run_timestamp or "") or sanitize_token(Path(source_out_dir).name) or "run"
    export_path = export_dir / f"{stem}__{ts_token}.pt"
    latest_path = export_dir / f"{stem}.pt"

    shutil.copy2(source_checkpoint, export_path)
    shutil.copy2(source_checkpoint, latest_path)
    write_json_file(
        Path(f"{export_path}.meta.json"),
        {
            "stage": "sl_trial_training_export",
            "domain": "csi_bench",
            "dataset": "CSI-Bench",
            "task_name": task_name,
            "model": model_name,
            "variant": "timestamped",
            "run_timestamp": run_timestamp,
            "latest_path": str(latest_path),
            "source_checkpoint": str(Path(source_checkpoint).resolve()),
            "source_run_dir": str(Path(source_out_dir).resolve()),
            "config_path": str(Path(config_path).resolve()) if config_path else None,
        },
    )
    write_json_file(
        Path(f"{latest_path}.meta.json"),
        {
            "stage": "sl_trial_training_export",
            "domain": "csi_bench",
            "dataset": "CSI-Bench",
            "task_name": task_name,
            "model": model_name,
            "variant": "latest_alias",
            "run_timestamp": run_timestamp,
            "timestamped_path": str(export_path),
            "source_checkpoint": str(Path(source_checkpoint).resolve()),
            "source_run_dir": str(Path(source_out_dir).resolve()),
            "config_path": str(Path(config_path).resolve()) if config_path else None,
        },
    )
    return str(export_path)


def _run_ntu(
    *,
    cfg: Mapping[str, Any],
    config_path: Optional[str],
    experiment_name: str,
    model_names: Sequence[str],
    save_weights: bool,
    dataset_weights_dir_name: str,
    train_cfg: Mapping[str, Any],
    device: torch.device,
    data_seed: int,
) -> None:
    ntu_cfg = dict(cfg.get("ntu", {}) or {})
    datasets = as_list(ntu_cfg.get("datasets") or ntu_cfg.get("dataset"), default=["NTU-Fi_HAR", "NTU-Fi-HumanID"])
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

    lr = float(train_cfg.get("learning_rate", 1e-3))
    epochs_override = train_cfg.get("epochs")
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

    criterion = nn.CrossEntropyLoss()

    for dataset_name in datasets:
        for model_name in model_names:
            model_token = sanitize_token(model_name)
            out_dir, ts, cfg_name = prepare_results_dir(
                experiment_name=experiment_name,
                cfg_path=config_path,
                relative_parts=["ntu", dataset_name, model_token],
            )
            completed = False
            try:
                save_run_meta(
                    out_dir,
                    {
                        "domain": "ntu",
                        "dataset": dataset_name,
                        "model": model_name,
                        "timestamp": ts,
                        "config_name": cfg_name,
                        "root": root,
                        "task": task,
                        "timesteps": timesteps,
                        "batch_sizes": batch_sizes,
                        "num_workers": num_workers,
                        "pin_memory": pin_memory,
                        "persistent_workers": persistent_workers,
                        "prefetch_factor": prefetch_factor,
                        "device": str(device),
                    },
                )
                save_config(out_dir, dict(cfg))
                print_training_start(
                    f"Clean NTU training ({dataset_name}/{model_name})",
                    {
                        "output_dir": out_dir,
                        "dataset": dataset_name,
                        "root": root,
                        "model": model_name,
                        "device": str(device),
                        "timesteps": timesteps,
                        "batch_sizes": batch_sizes,
                        "num_workers": num_workers,
                        "pin_memory": pin_memory,
                        "persistent_workers": persistent_workers,
                        "prefetch_factor": prefetch_factor,
                        "learning_rate": lr,
                        "epochs": epochs_override or "auto",
                    },
                )

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
                    seed=ntu_seed,
                )
                train_loader = bundle.train_loader
                val_loader = bundle.val_loader
                test_loader = bundle.test_loader
                if train_loader is None:
                    raise RuntimeError(f"Train loader missing for dataset {dataset_name}")
                if val_loader is None and test_loader is None:
                    raise RuntimeError(f"Validation and test loaders are both missing for dataset {dataset_name}")

                bundle_info = bundle.info or {}
                shape = infer_ntu_shape(bundle_info, train_loader)
                num_classes = resolve_ntu_num_classes(bundle_info, train_loader)
                arch_metadata: Dict[str, Any] = {"num_classes": num_classes}
                if shape is not None:
                    channels, freq_bins, time_steps = shape
                    arch_metadata.update(
                        {
                            "channels": channels,
                            "freq_bins": freq_bins,
                            "time_steps": time_steps,
                            "input_shape": [channels, freq_bins, time_steps],
                        }
                    )

                model, suggested_epochs = build_model(
                    dataset_name,
                    model_name,
                    num_classes=num_classes,
                    arch_metadata=arch_metadata,
                )
                train_epochs = int(epochs_override) if epochs_override is not None else int(suggested_epochs)
                model = model.to(device)

                summary = train(
                    model=model,
                    tensor_loader=train_loader,
                    num_epochs=train_epochs,
                    learning_rate=lr,
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

                save_results_rows(out_dir, history_rows_supervised(summary.history), base_name="train_history")

                eval_rows = evaluate_splits(
                    model=model,
                    criterion=criterion,
                    device=device,
                    use_tqdm=progress_bar,
                    split_loaders={
                        "val": val_loader,
                        "test": test_loader,
                    },
                )
                save_results_rows(out_dir, eval_rows, base_name="summary_rows")

                weights_path = None
                exported_weights_path = None
                if save_weights:
                    weights_path = save_model_weights(out_dir=out_dir, filename_stem=model_name, model=model)
                    exported_weights_path = _export_ntu_checkpoint(
                        datasets_root=root,
                        weights_dir_name=dataset_weights_dir_name,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        source_checkpoint=weights_path,
                        source_out_dir=out_dir,
                        config_path=config_path,
                        run_timestamp=ts,
                    )

                save_training_metadata(
                    out_dir,
                    {
                        "stage": "clean_train_ntu",
                        "dataset": dataset_name,
                        "model_name": model_name,
                        "root": root,
                        "task": task,
                        "timesteps": timesteps,
                        "batch_sizes": batch_sizes,
                        "num_workers": num_workers,
                        "pin_memory": pin_memory,
                        "persistent_workers": persistent_workers,
                        "prefetch_factor": prefetch_factor,
                        "device": str(device),
                        "num_classes": num_classes,
                        "learning_rate": lr,
                        "epochs": train_epochs,
                        "weights_path": weights_path,
                        "dataset_export_weights_path": exported_weights_path,
                        "dataset_export_weights_dir_name": dataset_weights_dir_name,
                    },
                )
                completed = True
                print(f"[RUN] Completed: {out_dir}")
                if exported_weights_path:
                    print(f"[INFO] Exported dataset checkpoint: {exported_weights_path}")

            except BaseException as exc:
                print(f"[ERROR] NTU run failed for {dataset_name}/{model_name}: {exc}")
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
    model_names: Sequence[str],
    save_weights: bool,
    dataset_weights_dir_name: str,
    train_cfg: Mapping[str, Any],
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
    csi_seed = int(csi_cfg.get("seed", data_seed))

    lr = float(train_cfg.get("learning_rate", 1e-3))
    epochs_override = train_cfg.get("epochs")
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

    criterion = nn.CrossEntropyLoss()

    for task_name in task_names:
        for model_name in model_names:
            model_token = sanitize_token(model_name)
            out_dir, ts, cfg_name = prepare_results_dir(
                experiment_name=experiment_name,
                cfg_path=config_path,
                relative_parts=["csi_bench", task_name, model_token],
            )
            completed = False
            try:
                save_run_meta(
                    out_dir,
                    {
                        "domain": "csi_bench",
                        "dataset": "CSI-Bench",
                        "task_name": task_name,
                        "model": model_name,
                        "timestamp": ts,
                        "config_name": cfg_name,
                        "dataset_root": dataset_root,
                        "splits": only_splits,
                        "device": str(device),
                    },
                )
                save_config(out_dir, dict(cfg))
                print_training_start(
                    f"Clean CSI-Bench training ({task_name}/{model_name})",
                    {
                        "output_dir": out_dir,
                        "dataset_root": dataset_root,
                        "task": task_name,
                        "model": model_name,
                        "device": str(device),
                        "batch_size": batch_size,
                        "num_workers": num_workers,
                        "pin_memory": pin_memory,
                        "persistent_workers": persistent_workers,
                        "prefetch_factor": prefetch_factor,
                        "normalisation": normalisation,
                        "target_subcarriers": target_subcarriers,
                        "target_timesteps": target_timesteps,
                        "learning_rate": lr,
                        "epochs": epochs_override or "auto",
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
                if val_loader is None and test_loader is None:
                    raise RuntimeError(f"Validation and test loaders are both missing for task {task_name}")

                num_classes = int(bundle["num_classes"])
                shape = infer_input_shape_from_loader(train_loader)
                arch_metadata: Dict[str, Any] = {"num_classes": num_classes}
                if shape is not None:
                    channels, freq_bins, time_steps = shape
                    arch_metadata.update(
                        {
                            "channels": channels,
                            "freq_bins": freq_bins,
                            "time_steps": time_steps,
                            "input_shape": [channels, freq_bins, time_steps],
                        }
                    )

                model, suggested_epochs = build_model(
                    "csi-bench",
                    model_name,
                    num_classes=num_classes,
                    arch_metadata=arch_metadata,
                )
                train_epochs = int(epochs_override) if epochs_override is not None else int(suggested_epochs)
                model = model.to(device)

                summary = train(
                    model=model,
                    tensor_loader=train_loader,
                    num_epochs=train_epochs,
                    learning_rate=lr,
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

                save_results_rows(out_dir, history_rows_supervised(summary.history), base_name="train_history")
                eval_rows = evaluate_splits(
                    model=model,
                    criterion=criterion,
                    device=device,
                    use_tqdm=progress_bar,
                    split_loaders={
                        "val": val_loader,
                        "test": test_loader,
                    },
                )
                save_results_rows(out_dir, eval_rows, base_name="summary_rows")

                weights_path = None
                exported_weights_path = None
                if save_weights:
                    weights_path = save_model_weights(out_dir=out_dir, filename_stem=model_name, model=model)
                    exported_weights_path = _export_csi_bench_checkpoint(
                        datasets_root=dataset_root,
                        weights_dir_name=dataset_weights_dir_name,
                        task_name=task_name,
                        model_name=model_name,
                        source_checkpoint=weights_path,
                        source_out_dir=out_dir,
                        config_path=config_path,
                        run_timestamp=ts,
                    )

                save_training_metadata(
                    out_dir,
                    {
                        "stage": "clean_train_csi_bench",
                        "dataset": "CSI-Bench",
                        "task_name": task_name,
                        "model_name": model_name,
                        "dataset_root": dataset_root,
                        "splits": only_splits,
                        "batch_size": batch_size,
                        "num_workers": num_workers,
                        "pin_memory": pin_memory,
                        "persistent_workers": persistent_workers,
                        "prefetch_factor": prefetch_factor,
                        "normalisation": normalisation,
                        "target_subcarriers": target_subcarriers,
                        "target_timesteps": target_timesteps,
                        "device": str(device),
                        "num_classes": num_classes,
                        "learning_rate": lr,
                        "epochs": train_epochs,
                        "weights_path": weights_path,
                        "dataset_export_weights_path": exported_weights_path,
                        "dataset_export_weights_dir_name": dataset_weights_dir_name,
                        "task_metadata": collect_csi_bench_task_metadata(
                            dataset_root=dataset_root,
                            task_name=task_name,
                            splits_dir_name=splits_dir_name,
                            split_names=only_splits,
                        ),
                    },
                )
                completed = True
                print(f"[RUN] Completed: {out_dir}")
                if exported_weights_path:
                    print(f"[INFO] Exported dataset checkpoint: {exported_weights_path}")

            except BaseException as exc:
                print(f"[ERROR] CSI-Bench run failed for {task_name}/{model_name}: {exc}")
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
    experiment_name = str(common_cfg.get("experiment_name", "clean-baselines"))
    model_cfg = cfg.get("model", {})
    model_names = resolve_supervised_models(
        as_list(
            model_cfg.get("architectures") or model_cfg.get("models"),
            default=["ResNet18"],
        )
    )
    save_weights = parse_bool(model_cfg.get("save_weights"), default=True)
    dataset_weights_dir_name = str(
        model_cfg.get("dataset_weights_dir_name")
        or model_cfg.get("weights_dir_name")
        or "model_params_sl_trials"
    )

    data_cfg = cfg.get("data", {})
    device = _resolve_device(data_cfg.get("device"))
    data_seed = int(data_cfg.get("seed", 42))

    run_cfg = cfg.get("run", {})
    targets_raw = as_list(run_cfg.get("targets"), default=["ntu", "csi_bench"])
    targets = {token.strip().lower().replace("-", "_") for token in targets_raw}

    train_cfg = cfg.get("training", {})

    if "ntu" in targets:
        _run_ntu(
            cfg=cfg,
            config_path=config_path,
            experiment_name=experiment_name,
            model_names=model_names,
            save_weights=save_weights,
            dataset_weights_dir_name=dataset_weights_dir_name,
            train_cfg=train_cfg,
            device=device,
            data_seed=data_seed,
        )
    if "csi_bench" in targets:
        _run_csi_bench(
            cfg=cfg,
            config_path=config_path,
            experiment_name=experiment_name,
            model_names=model_names,
            save_weights=save_weights,
            dataset_weights_dir_name=dataset_weights_dir_name,
            train_cfg=train_cfg,
            device=device,
            data_seed=data_seed,
        )


if __name__ == "__main__":
    cfg_path = os.environ.get("EXP_TRAIN_CONFIG")
    if cfg_path is None and len(sys.argv) > 1:
        cfg_path = resolve_config_path_from_arg(sys.argv[1])
    main(cfg_path)
