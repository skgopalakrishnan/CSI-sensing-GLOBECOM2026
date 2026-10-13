"""Optuna hyperparameter tuning for supervised NTU/UT CSI models."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from csi.data import get_experiment_loaders
from csi.models.supervised import evaluate_classifier, train
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
    export_best_weight_to_datasets,
    move_incomplete_run,
    parse_bool,
    print_training_start,
    save_training_metadata,
)
from ..common.train_models_utils import (
    as_list,
    configure_torch_performance,
    infer_spec_from_bundle,
    json_dump,
    resolve_ntu_num_classes,
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


def main(config_path: Optional[str] = None) -> None:
    if optuna is None:
        raise RuntimeError(f"Optuna is required to run this script: {_OPTUNA_IMPORT_ERROR}")

    cfg = load_config(config_path)
    configure_from_env_and_config(cfg)
    configure_torch_performance(cfg)

    common_cfg = cfg.get("common", {})
    datasets = as_list(common_cfg.get("datasets") or cfg.get("datasets"), default=["NTU-Fi_HAR"])
    root = str(common_cfg.get("root", "datasets"))
    experiment_name = str(common_cfg.get("experiment_name", "ntu-optuna"))

    data_cfg = cfg.get("data", {})
    batch_sizes = data_cfg.get("batch_sizes")
    task = str(data_cfg.get("task", "classification"))
    timesteps = int(data_cfg.get("timesteps", 250))
    base_normalisation = str(data_cfg.get("base_normalisation", data_cfg.get("base_normalization", "none")))
    canon_normalisation = str(data_cfg.get("canon_normalisation", data_cfg.get("canonical_normalization", "per_subcarrier")))
    variant = str(data_cfg.get("variant", "raw"))
    smoothing = data_cfg.get("smoothing")
    num_workers_raw = data_cfg.get("num_workers")
    num_workers = int(num_workers_raw) if num_workers_raw is not None else None
    device = _resolve_device(data_cfg.get("device"))
    pin_memory_raw = data_cfg.get("pin_memory")
    pin_memory = None if pin_memory_raw is None else parse_bool(pin_memory_raw, default=(device.type == "cuda"))
    persistent_workers_raw = data_cfg.get("persistent_workers")
    persistent_workers = None if persistent_workers_raw is None else parse_bool(persistent_workers_raw, default=True)
    prefetch_factor_raw = data_cfg.get("prefetch_factor")
    prefetch_factor = int(prefetch_factor_raw) if prefetch_factor_raw is not None else None
    augment_train: Dict[str, float] = dict(data_cfg.get("augment", {}) or {})
    augment_eval: Dict[str, float] = dict(data_cfg.get("augment_eval", augment_train) or augment_train)
    data_seed = int(data_cfg.get("seed", 42))

    model_cfg = cfg.get("model", {})
    model_names = as_list(model_cfg.get("architectures") or model_cfg.get("models"), default=["ResNet18"])
    save_trial_weights = parse_bool(model_cfg.get("save_trial_weights"), default=False)
    trial_weights_cfg = model_cfg.get("trial_weights", {}) or {}
    trial_weights_dir_name = sanitize_token(str(trial_weights_cfg.get("dir_name", "trial_weights_single_optuna"))) or "trial_weights_single_optuna"
    trial_weights_mode = str(trial_weights_cfg.get("mode", "top_k"))
    trial_weights_top_k = int(trial_weights_cfg.get("top_k", 5))
    trial_weights_save_every = max(1, int(trial_weights_cfg.get("save_every", 1)))
    trial_weights_max_saved_raw = trial_weights_cfg.get("max_saved")
    trial_weights_max_saved = int(trial_weights_max_saved_raw) if trial_weights_max_saved_raw is not None else None
    trial_weights_cleanup = parse_bool(trial_weights_cfg.get("cleanup_after_study"), default=True)

    train_cfg = cfg.get("training", {})
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    epochs = int(train_cfg.get("epochs", 40))
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

    opt_cfg = cfg.get("optuna", {})
    n_trials = int(opt_cfg.get("n_trials", 50))
    timeout_raw = opt_cfg.get("timeout")
    timeout = int(timeout_raw) if timeout_raw is not None else None
    n_jobs = int(opt_cfg.get("n_jobs", 1))
    direction = str(opt_cfg.get("direction", "maximize")).strip().lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("optuna.direction must be either 'maximize' or 'minimize'.")
    objective_metric = str(opt_cfg.get("objective_metric", "val_accuracy")).strip().lower()
    if objective_metric not in {"val_accuracy", "val_loss"}:
        raise ValueError("optuna.objective_metric must be 'val_accuracy' or 'val_loss'.")
    study_prefix = str(opt_cfg.get("study_name", "ntu_optuna"))
    load_if_exists = parse_bool(opt_cfg.get("load_if_exists"), default=True)
    storage_url = opt_cfg.get("storage")
    gc_after_trial = parse_bool(opt_cfg.get("gc_after_trial"), default=True)

    export_cfg = cfg.get("export", {})
    export_best_enabled = parse_bool(export_cfg.get("save_best_to_datasets"), default=True)
    export_datasets_root = str(export_cfg.get("datasets_root", root))
    export_weights_dir_name = str(export_cfg.get("weights_dir_name", "model_params_optuna_best"))
    export_overwrite_strategy = str(export_cfg.get("overwrite_strategy", "unique"))
    export_include_full_config = parse_bool(export_cfg.get("include_full_config"), default=True)
    export_fail_on_error = parse_bool(export_cfg.get("fail_on_error"), default=False)

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

    criterion = nn.CrossEntropyLoss()

    for dataset_name in datasets:
        for model_name in model_names:
            model_token = sanitize_token(model_name)
            out_dir, ts, cfg_name = prepare_results_dir(
                experiment_name=experiment_name,
                cfg_path=config_path,
                relative_parts=[dataset_name, model_token],
            )
            trial_weights_dir = os.path.join(out_dir, trial_weights_dir_name)
            if save_trial_weights:
                os.makedirs(trial_weights_dir, exist_ok=True)
            completed = False

            run_meta = {
                "dataset": dataset_name,
                "model": model_name,
                "timestamp": ts,
                "config_name": cfg_name,
                "root": root,
                "device": str(device),
                "optuna": {
                    "n_trials": n_trials,
                    "timeout": timeout,
                    "direction": direction,
                    "objective_metric": objective_metric,
                    "n_jobs": n_jobs,
                },
            }
            save_run_meta(out_dir, run_meta)
            save_config(out_dir, cfg)

            print_training_start(
                f"NTU Optuna tuning ({dataset_name}/{model_name})",
                {
                    "output_dir": out_dir,
                    "dataset": dataset_name,
                    "model": model_name,
                    "root": root,
                    "device": str(device),
                    "timesteps": timesteps,
                    "batch_sizes": batch_sizes,
                    "num_workers": num_workers,
                    "pin_memory": pin_memory,
                    "persistent_workers": persistent_workers,
                    "prefetch_factor": prefetch_factor,
                    "n_trials": n_trials,
                    "timeout": timeout,
                    "objective_metric": objective_metric,
                    "direction": direction,
                },
            )

            try:
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
                    seed=data_seed,
                )

                train_loader = bundle.train_loader
                val_loader = bundle.val_loader
                test_loader = bundle.test_loader
                if train_loader is None:
                    raise RuntimeError(f"Train loader unavailable for dataset {dataset_name}")
                if val_loader is None and test_loader is None:
                    raise RuntimeError(f"Both val/test loaders are missing for dataset {dataset_name}")

                bundle_info = bundle.info or {}
                spec = infer_spec_from_bundle(dataset_name, bundle_info, train_loader)
                num_classes = resolve_ntu_num_classes(bundle_info, train_loader)
                eval_loader = val_loader if val_loader is not None else test_loader
                assert eval_loader is not None

                save_training_metadata(
                    out_dir,
                    {
                        "stage": "ntu_optuna",
                        "dataset": dataset_name,
                        "model": model_name,
                        "root": root,
                        "device": str(device),
                        "task": task,
                        "timesteps": timesteps,
                        "batch_sizes": batch_sizes,
                        "num_workers": num_workers,
                        "pin_memory": pin_memory,
                        "persistent_workers": persistent_workers,
                        "prefetch_factor": prefetch_factor,
                        "spec": {
                            "channels": spec.channels,
                            "freq_bins": spec.freq_bins,
                            "time_steps": spec.time_steps,
                            "num_classes": num_classes,
                        },
                        "training": {
                            "learning_rate": learning_rate,
                            "epochs": epochs,
                            "patience": patience,
                            "min_epochs": min_epochs,
                            "monitor_metric": monitor_metric,
                            "min_delta": min_delta,
                            "weight_decay": weight_decay,
                            "grad_clip": grad_clip,
                        },
                        "optuna": {
                            "n_trials": n_trials,
                            "timeout": timeout,
                            "direction": direction,
                            "objective_metric": objective_metric,
                        },
                    },
                )

                best_weights_path = os.path.join(out_dir, f"{model_token}_best.pt")

                def objective(trial: "Trial") -> float: # type: ignore
                    hparams = suggest_hparams_for_model(trial, model_name, spec)

                    try:
                        model = build_model_from_hparams(spec, model_name, num_classes, hparams).to(device)
                        summary = run_with_oom_guard(
                            lambda: train(
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
                        )

                        val_metrics, _ = run_with_oom_guard(
                            lambda: evaluate_classifier(
                                model=model,
                                loader=eval_loader,
                                criterion=criterion,
                                device=device,
                                use_tqdm=progress_bar,
                                collect_predictions=False,
                                collect_probabilities=False,
                                desc=f"val_trial_{trial.number}",
                            )
                        )
                        test_metrics = None
                        if test_loader is not None:
                            test_metrics, _ = run_with_oom_guard(
                                lambda: evaluate_classifier(
                                    model=model,
                                    loader=test_loader,
                                    criterion=criterion,
                                    device=device,
                                    use_tqdm=progress_bar,
                                    collect_predictions=False,
                                    collect_probabilities=False,
                                    desc=f"test_trial_{trial.number}",
                                )
                            )

                        objective_value = float(
                            val_metrics.get("accuracy", 0.0)
                            if objective_metric == "val_accuracy"
                            else val_metrics.get("loss", float("inf"))
                        )

                        trial.set_user_attr("objective_metric", objective_metric)
                        trial.set_user_attr("val_accuracy", float(val_metrics.get("accuracy", 0.0)))
                        trial.set_user_attr("val_loss", float(val_metrics.get("loss", 0.0)))
                        trial.set_user_attr("best_epoch", summary.best_epoch)
                        if test_metrics is not None:
                            trial.set_user_attr("test_accuracy", float(test_metrics.get("accuracy", 0.0)))
                            trial.set_user_attr("test_loss", float(test_metrics.get("loss", 0.0)))

                        if save_trial_weights:
                            if (trial.number + 1) % trial_weights_save_every == 0:
                                torch.save(
                                    model.state_dict(),
                                    os.path.join(trial_weights_dir, f"trial_{trial.number:04d}.pt"),
                                )

                        return objective_value

                    except CUDATrialOOMError as exc:
                        trial.set_user_attr("oom", True)
                        raise TrialPruned(str(exc)) from exc # type: ignore
                    except RuntimeError as exc:
                        msg = str(exc).lower()
                        if "out of memory" in msg and ("cuda" in msg or "cudnn" in msg):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            trial.set_user_attr("oom", True)
                            raise TrialPruned("CUDA OOM during trial") from exc # type: ignore
                        raise
                    finally:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                resolved_storage = storage_url
                if not resolved_storage:
                    resolved_storage = f"sqlite:///{os.path.join(out_dir, 'optuna_study.db')}"
                study_name = f"{study_prefix}_{sanitize_token(dataset_name)}_{sanitize_token(model_name)}"

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
                    "val_accuracy": best_trial.user_attrs.get("val_accuracy"),
                    "val_loss": best_trial.user_attrs.get("val_loss"),
                    "test_accuracy": best_trial.user_attrs.get("test_accuracy"),
                    "test_loss": best_trial.user_attrs.get("test_loss"),
                }
                json_dump(os.path.join(out_dir, "best_trial.json"), best_payload)

                # Retrain once with the best hparams to produce a clean deployable checkpoint.
                best_model = build_model_from_hparams(spec, model_name, num_classes, dict(best_trial.params)).to(device)
                _ = train(
                    model=best_model,
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
                torch.save(best_model.state_dict(), best_weights_path)

                if export_best_enabled:
                    export_meta = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "best_trial": best_payload,
                        "objective_metric": objective_metric,
                        "direction": direction,
                        "study_name": study.study_name,
                        "storage": resolved_storage,
                    }
                    try:
                        export_result = export_best_weight_to_datasets(
                            source_checkpoint=best_weights_path,
                            datasets_root=export_datasets_root,
                            domain="ntu",
                            dataset_name=dataset_name,
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
                        if export_result.get("exported"):
                            print(
                                "[EXPORT] Saved best checkpoint to datasets: "
                                f"{export_result.get('target_checkpoint')}"
                            )
                        else:
                            print(
                                "[EXPORT] Skipped best-checkpoint export "
                                f"({export_result.get('reason')}): {export_result.get('target_checkpoint')}"
                            )
                    except Exception as export_exc:
                        print(f"[WARN] Failed to export best checkpoint to datasets: {export_exc}")
                        if export_fail_on_error:
                            raise

                final_splits: List[Tuple[str, Any]] = [("val", eval_loader)]
                if test_loader is not None:
                    final_splits.append(("test", test_loader))
                final_rows: List[Dict[str, Any]] = []
                for split_name, loader in final_splits:
                    metrics, _ = evaluate_classifier(
                        model=best_model,
                        loader=loader,
                        criterion=criterion,
                        device=device,
                        use_tqdm=progress_bar,
                        collect_predictions=False,
                        collect_probabilities=False,
                        desc=f"best_{split_name}",
                    )
                    final_rows.append(
                        {
                            "split": split_name,
                            "accuracy": float(metrics.get("accuracy", 0.0)),
                            "loss": float(metrics.get("loss", 0.0)),
                            "f1_macro": float(metrics.get("f1_macro", 0.0)),
                            "f1_weighted": float(metrics.get("f1_weighted", 0.0)),
                        }
                    )
                save_results_rows(out_dir, final_rows, base_name="best_eval")

                completed = True
                print(f"[RUN] Completed: {out_dir}")

            except BaseException as exc:
                print(f"[ERROR] Run failed for {dataset_name}/{model_name}: {exc}")
                traceback.print_exc()
                raise
            finally:
                if not completed:
                    target_dir = move_incomplete_run(out_dir)
                    if target_dir:
                        print(f"[INFO] Moved incomplete run to: {target_dir}")


if __name__ == "__main__":
    cfg_path = os.environ.get("EXP_TRAIN_CONFIG")
    if cfg_path is None and len(sys.argv) > 1:
        cfg_path = resolve_config_path_from_arg(sys.argv[1])
    main(cfg_path)
