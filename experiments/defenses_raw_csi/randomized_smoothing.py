"""Randomized smoothing training + certification runner for NTU and CSI-Bench."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from csi.data import get_experiment_loaders
from csi.data.utils import filter_none_collate, load_csi_bench_supervised
from csi.defenses.adv_training.randomized_smoothing import RandomizedSmoothingTrainer
from csi.defenses.certified_robustness.randomized_smoothing import (
    RandomizedSmoother,
    certify_dataset,
    certified_accuracy_at_radius,
    evaluate_smoothed_accuracy,
)
from csi.models.model_factory import build_model
from csi.models.utils.datasets import DatasetSpec
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
    DEFAULT_CLEAN_WEIGHTS_DIR,
    normalize_csi_bench_task_name,
    parse_bool,
    parse_splits,
    parse_token_list,
    print_training_start,
)
from ..common.edge_profile import augment_profile_row, build_model_profile, resolve_profile_cfg
from ..common.optuna_clean_runtime import build_model_for_checkpoint, ensure_optuna_clean_checkpoint
from ..common.train_models_utils import (
    DEFAULT_CSI_BENCH_SUBCARRIERS,
    DEFAULT_CSI_BENCH_TIMESTEPS,
    as_list,
    configure_torch_performance,
    infer_input_shape_from_loader,
    infer_ntu_shape,
    resolve_project_path,
    resolve_ntu_num_classes,
    resolve_supervised_models,
)
from .common_runner import (
    as_weight_dir_list,
    build_checkpoint_payload,
    call_with_supported_kwargs,
    clean_metrics,
    default_csi_tasks,
    default_ntu_datasets,
    export_csi_checkpoint,
    export_ntu_checkpoint,
    loader_key_for_csi_split,
    nested_checkpoint_override,
    normalise_csi_split_name,
    normalise_selection_mode,
    resolve_checkpoint,
    resolve_checkpoint_epochs,
    resolve_csi_bench_checkpoint_dir,
    load_checkpoint_into_model,
)
from .rs_sigma import resolve_sigma_for_domain


def _parse_optional_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        out = int(value)
    except Exception:
        return None
    return out if out > 0 else None


def _resolve_sigma_for_domain(rs_cfg: Mapping[str, Any], *, domain: str) -> float:
    return resolve_sigma_for_domain(rs_cfg, domain=domain)


def _resolve_rs_eval_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    eval_cfg = dict(cfg.get("eval", {}) or {})
    smoothed_cfg = dict(eval_cfg.get("smoothed", {}) or {})
    cert_cfg = dict(eval_cfg.get("certification", {}) or {})

    radii_raw = cert_cfg.get("radii", [0.0, 0.02, 0.05, 0.1])
    if isinstance(radii_raw, str):
        radii = [float(token.strip()) for token in radii_raw.split(",") if token.strip()]
    elif isinstance(radii_raw, (list, tuple)):
        radii = [float(x) for x in radii_raw]
    else:
        radii = [0.0, 0.02, 0.05, 0.1]
    radii = sorted(set(max(0.0, float(r)) for r in radii))

    return {
        "max_batches": _parse_optional_positive_int(eval_cfg.get("max_batches")),
        "smoothed": {
            "n": int(smoothed_cfg.get("n", 1024)),
            "alpha": float(smoothed_cfg.get("alpha", 0.001)),
            "batch_size": int(smoothed_cfg.get("batch_size", 256)),
            "skip": int(smoothed_cfg.get("skip", 1)),
            "max_examples": _parse_optional_positive_int(smoothed_cfg.get("max_examples")),
            "use_tqdm": parse_bool(smoothed_cfg.get("use_tqdm"), default=False),
        },
        "cert": {
            "n0": int(cert_cfg.get("n0", 100)),
            "n": int(cert_cfg.get("n", 10000)),
            "alpha": float(cert_cfg.get("alpha", 0.001)),
            "batch_size": int(cert_cfg.get("batch_size", 256)),
            "skip": int(cert_cfg.get("skip", 1)),
            "max_examples": _parse_optional_positive_int(cert_cfg.get("max_examples")),
            "use_tqdm": parse_bool(cert_cfg.get("use_tqdm"), default=False),
            "radii": radii,
        },
        "quick": {
            "enabled": parse_bool(eval_cfg.get("eval_during_training"), default=False),
            "n": int(eval_cfg.get("quick_n", 128)),
            "alpha": float(eval_cfg.get("quick_alpha", 0.001)),
            "batch_size": int(eval_cfg.get("quick_batch_size", 128)),
            "max_examples": _parse_optional_positive_int(eval_cfg.get("quick_max_examples", 128)),
        },
    }


def _apply_env_overrides(cfg: Dict[str, Any]) -> None:
    run_cfg = cfg.setdefault("run", {})
    model_cfg = cfg.setdefault("model", {})
    rs_cfg = cfg.setdefault("certified_robustness", {}).setdefault("randomized_smoothing", {})
    ntu_cfg = cfg.setdefault("ntu", {})
    csi_cfg = cfg.setdefault("csi_bench", {})
    weights_cfg = cfg.setdefault("weights", {})

    models_env = os.environ.get("RS_MODELS")
    if models_env:
        model_cfg["architectures"] = parse_token_list(models_env)

    targets_env = os.environ.get("RS_TARGETS")
    if targets_env:
        run_cfg["targets"] = parse_token_list(targets_env)

    task_family_env = os.environ.get("RS_TASK_FAMILY")
    if task_family_env:
        run_cfg["task_family"] = str(task_family_env).strip().lower()

    sigma_env = os.environ.get("RS_SIGMA")
    if sigma_env:
        try:
            rs_cfg["sigma"] = float(sigma_env)
        except Exception:
            pass

    snr_env = os.environ.get("RS_SNR_DB")
    if snr_env:
        try:
            rs_cfg["snr_db"] = float(snr_env)
        except Exception:
            pass

    ntu_datasets_env = os.environ.get("RS_NTU_DATASETS")
    if ntu_datasets_env:
        ntu_cfg["datasets"] = parse_token_list(ntu_datasets_env)

    csi_tasks_env = os.environ.get("RS_CSI_TASKS")
    if csi_tasks_env:
        csi_cfg["task_names"] = parse_token_list(csi_tasks_env)

    source_weights_env = os.environ.get("RS_SOURCE_WEIGHTS_DIR")
    if source_weights_env:
        weights_cfg["source_weights_dir_name"] = str(source_weights_env).strip()

    rs_weights_env = os.environ.get("RS_WEIGHTS_DIR")
    if rs_weights_env:
        weights_cfg["rs_weights_dir_name"] = str(rs_weights_env).strip()


def _collect_cert_metrics(
    *,
    model,
    loader,
    device: torch.device,
    num_classes: int,
    sigma: float,
    noise_type: str,
    amplitude_clamp_min: Optional[float],
    eval_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    smoother = RandomizedSmoother(
        model,
        num_classes=num_classes,
        sigma=float(sigma),
        device=device,
        noise_type=noise_type,
        amplitude_clamp_min=amplitude_clamp_min,
    )

    smoothed = dict(eval_cfg["smoothed"])
    smoothed_eval = evaluate_smoothed_accuracy(
        smoother,
        loader,
        n=int(smoothed["n"]),
        alpha=float(smoothed["alpha"]),
        batch_size=int(smoothed["batch_size"]),
        skip=int(smoothed["skip"]),
        max_examples=smoothed["max_examples"],
        use_tqdm=bool(smoothed["use_tqdm"]),
    )

    cert = dict(eval_cfg["cert"])
    cert_report = certify_dataset(
        smoother,
        loader,
        n0=int(cert["n0"]),
        n=int(cert["n"]),
        alpha=float(cert["alpha"]),
        batch_size=int(cert["batch_size"]),
        skip=int(cert["skip"]),
        max_examples=cert["max_examples"],
        use_tqdm=bool(cert["use_tqdm"]),
    )
    cert_rows = cert_report.get("rows", [])
    cert_summary = cert_report.get("summary", {})

    radii = cert.get("radii", [])
    cert_acc_by_radius: Dict[str, float] = {}
    for radius in radii:
        key = f"{float(radius):g}".replace(".", "p")
        cert_acc_by_radius[f"cert_acc_r_{key}"] = certified_accuracy_at_radius(cert_rows, radius=float(radius))

    return {
        "smoothed": smoothed_eval,
        "cert_summary": cert_summary,
        "cert_rows": cert_rows,
        "cert_acc_by_radius": cert_acc_by_radius,
    }


def _run_ntu(
    *,
    cfg: Mapping[str, Any],
    cfg_path: Optional[str],
    out_dir: str,
    run_ts: str,
    model_names: Sequence[str],
    device: torch.device,
    task_family: str,
    applied_seed: Optional[int],
    train_rows: List[Dict[str, Any]],
    final_rows: List[Dict[str, Any]],
    profile_cfg: Mapping[str, Any],
) -> None:
    ntu_cfg = dict(cfg.get("ntu", {}) or {})
    weights_cfg = dict(cfg.get("weights", {}) or {})
    weights_ntu = dict(weights_cfg.get("ntu", {}) or {})
    train_cfg = dict(cfg.get("training", {}) or {})
    rs_cfg = dict(cfg.get("certified_robustness", {}).get("randomized_smoothing", {}) or {})
    eval_cfg = _resolve_rs_eval_cfg(cfg)

    sigma = _resolve_sigma_for_domain(rs_cfg, domain="ntu")
    noise_type = str(rs_cfg.get("noise_type", "gaussian")).strip().lower()
    amplitude_clamp_min = rs_cfg.get("amplitude_clamp_min", 0.0)
    amplitude_clamp_min = None if amplitude_clamp_min is None else float(amplitude_clamp_min)

    datasets = as_list(ntu_cfg.get("datasets") or ntu_cfg.get("dataset"), default=default_ntu_datasets(task_family))
    root = str(ntu_cfg.get("root", cfg.get("common", {}).get("root", "datasets")))
    task = str(ntu_cfg.get("task", "classification"))
    timesteps = int(ntu_cfg.get("timesteps", 250))
    batch_sizes = ntu_cfg.get("batch_sizes")
    base_normalisation = str(ntu_cfg.get("base_normalisation", "none"))
    canon_normalisation = str(ntu_cfg.get("canon_normalisation", "per_subcarrier"))
    variant = str(ntu_cfg.get("variant", "raw"))
    smoothing = ntu_cfg.get("smoothing")
    num_workers = _parse_optional_positive_int(ntu_cfg.get("num_workers"))
    pin_memory_raw = ntu_cfg.get("pin_memory")
    pin_memory = None if pin_memory_raw is None else parse_bool(pin_memory_raw, default=(device.type == "cuda"))
    persistent_workers_raw = ntu_cfg.get("persistent_workers")
    persistent_workers = None if persistent_workers_raw is None else parse_bool(persistent_workers_raw, default=True)
    prefetch_factor = _parse_optional_positive_int(ntu_cfg.get("prefetch_factor"))
    augment_train = dict(ntu_cfg.get("augment", {}) or {})
    augment_eval = dict(ntu_cfg.get("augment_eval", augment_train) or augment_train)
    data_seed = int(ntu_cfg.get("seed", cfg.get("data", {}).get("seed", 42)))
    eval_split = str(ntu_cfg.get("eval_split", "test")).strip().lower()
    if eval_split not in {"test", "val"}:
        raise ValueError(f"Unsupported ntu.eval_split={eval_split!r}. Use 'test' or 'val'.")

    source_root = str(weights_ntu.get("datasets_root", root))
    source_dir_names = as_weight_dir_list(
        weights_ntu.get("source_weights_dir_names", weights_ntu.get("weights_dir_names")),
        default=str(
            weights_ntu.get("source_weights_dir_name")
            or weights_cfg.get("source_weights_dir_name")
            or weights_cfg.get("weights_dir_name")
            or DEFAULT_CLEAN_WEIGHTS_DIR
        ),
    )
    rs_weights_dir_name = str(
        weights_ntu.get("rs_weights_dir_name")
        or weights_cfg.get("rs_weights_dir_name")
        or f"model_params_rs_{task_family}"
    )
    checkpoint_glob = weights_ntu.get("checkpoint_glob")
    checkpoint_paths = weights_ntu.get("checkpoint_paths")
    selection = normalise_selection_mode(weights_cfg.get("selection"))
    strict_load = parse_bool(weights_cfg.get("strict_load"), default=True)

    epochs = int(train_cfg.get("epochs", 20))
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    progress_bar = parse_bool(train_cfg.get("progress_bar"), default=False)
    use_amp = parse_bool(train_cfg.get("amp"), default=False)
    save_epochs = resolve_checkpoint_epochs(train_cfg, epochs)

    for dataset_name in datasets:
        for model_name in model_names:
            for source_dir_name in source_dir_names:
                print_training_start(
                    f"RS NTU ({dataset_name}/{model_name})",
                    {
                        "sigma": sigma,
                        "noise_type": noise_type,
                        "attack_free": True,
                        "data_seed": data_seed,
                        "source_weights_dir": source_dir_name,
                        "rs_weights_dir": rs_weights_dir_name,
                        "eval_split": eval_split,
                        "device": str(device),
                    },
                )

                bundle = call_with_supported_kwargs(
                    get_experiment_loaders,
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
                eval_loader = bundle.val_loader if eval_split == "val" else bundle.test_loader
                if train_loader is None:
                    raise RuntimeError(f"NTU train loader missing for {dataset_name}.")
                if eval_loader is None:
                    raise RuntimeError(f"NTU eval loader '{eval_split}' missing for {dataset_name}.")

                bundle_info = bundle.info or {}
                shape = infer_ntu_shape(bundle_info, train_loader)
                num_classes = resolve_ntu_num_classes(bundle_info, train_loader)
                arch_metadata: Dict[str, Any] = {"num_classes": num_classes}
                if shape is not None:
                    c, f, t = shape
                    arch_metadata.update({"channels": c, "freq_bins": f, "time_steps": t, "input_shape": [c, f, t]})

                source_base_dir = resolve_project_path(source_root) / dataset_name / source_dir_name
                explicit_checkpoint = nested_checkpoint_override(checkpoint_paths, dataset_name, model_name)
                explicit_checkpoint = ensure_optuna_clean_checkpoint(
                    domain="ntu",
                    model_name=model_name,
                    weights_dir_name=source_dir_name,
                    weights_cfg=weights_cfg,
                    canonical_base_dir=source_base_dir,
                    alias_stems=[f"{dataset_name}_{model_name}"],
                    dataset_name=dataset_name,
                ) or explicit_checkpoint
                if shape is None:
                    inferred = infer_input_shape_from_loader(train_loader)
                    if inferred is None:
                        raise RuntimeError(f"Unable to infer input shape for {dataset_name}/{model_name}.")
                    c, f, t = inferred
                spec = DatasetSpec(dataset_name, channels=c, freq_bins=f, time_steps=t)
                model, _ = build_model_for_checkpoint(
                    domain="ntu",
                    model_name=model_name,
                    weights_dir_name=source_dir_name,
                    weights_cfg=weights_cfg,
                    spec=spec,
                    dataset_key_for_model=dataset_name,
                    num_classes=num_classes,
                    arch_metadata=arch_metadata,
                    dataset_name=dataset_name,
                )
                model = model.to(device)
                checkpoint_path = resolve_checkpoint(
                    base_dir=source_base_dir,
                    model_name=model_name,
                    selection=selection,
                    alias_stems=[f"{dataset_name}_{model_name}"],
                    checkpoint_glob=str(checkpoint_glob) if checkpoint_glob else None,
                    explicit_checkpoint=explicit_checkpoint,
                )
                load_info = load_checkpoint_into_model(
                    model=model,
                    checkpoint_path=str(checkpoint_path),
                    strict=strict_load,
                    device=device,
                    eval_mode=False,
                )

                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                trainer_cfg = copy.deepcopy(dict(cfg))
                trainer_rs_cfg = trainer_cfg.setdefault("certified_robustness", {}).setdefault("randomized_smoothing", {})
                trainer_rs_cfg["sigma"] = float(sigma)
                trainer_rs_cfg["noise_type"] = noise_type
                trainer_rs_cfg["amplitude_clamp_min"] = amplitude_clamp_min
                trainer = RandomizedSmoothingTrainer(model, optimizer, trainer_cfg, amp=use_amp)

                local_ckpt_dir = Path(out_dir) / "checkpoints" / "ntu" / dataset_name / sanitize_token(model_name)
                local_ckpt_dir.mkdir(parents=True, exist_ok=True)
                final_local_ckpt = local_ckpt_dir / f"{sanitize_token(model_name)}_final.pt"

                for epoch in range(1, epochs + 1):
                    stats = trainer.train_one_epoch(train_loader, device=device, use_tqdm=progress_bar)
                    clean = clean_metrics(model=model, loader=eval_loader, device=device)

                    if epoch in save_epochs:
                        ckpt_path = local_ckpt_dir / f"{sanitize_token(model_name)}_epoch{epoch:03d}.pt"
                        torch.save(model.state_dict(), str(ckpt_path))
                        if epoch == epochs:
                            final_local_ckpt = ckpt_path

                    train_rows.append(
                        {
                            "domain": "ntu",
                            "dataset": dataset_name,
                            "model": model_name,
                            "source_weights_dir_name": source_dir_name,
                            "rs_weights_dir_name": rs_weights_dir_name,
                            "epoch": int(epoch),
                            "sigma": float(sigma),
                            "noise_type": noise_type,
                            "train_loss": float(stats.get("loss", 0.0)),
                            "train_loss_noisy": float(stats.get("loss_noisy", 0.0)),
                            "train_loss_clean": float(stats.get("loss_clean", 0.0)),
                            **clean,
                            "seed": int(applied_seed) if applied_seed is not None else None,
                        }
                    )

                if not final_local_ckpt.exists():
                    torch.save(model.state_dict(), str(final_local_ckpt))

                export_paths = export_ntu_checkpoint(
                    datasets_root=source_root,
                    export_weights_dir_name=rs_weights_dir_name,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    source_checkpoint=str(final_local_ckpt),
                    run_timestamp=run_ts,
                    run_dir=out_dir,
                    config_path=cfg_path,
                    stage="randomized_smoothing_export",
                    latest_alias=True,
                    checkpoint_payload=build_checkpoint_payload(model),
                )

                clean_final = clean_metrics(model=model, loader=eval_loader, device=device)
                model_profile = build_model_profile(model=model, loader=eval_loader, device=device, profile_cfg=profile_cfg)
                cert_metrics = _collect_cert_metrics(
                    model=model,
                    loader=eval_loader,
                    device=device,
                    num_classes=int(num_classes),
                    sigma=float(sigma),
                    noise_type=noise_type,
                    amplitude_clamp_min=amplitude_clamp_min,
                    eval_cfg=eval_cfg,
                )

                cert_rows = []
                for row in cert_metrics["cert_rows"]:
                    cert_rows.append(
                        {
                            "domain": "ntu",
                            "dataset": dataset_name,
                            "model": model_name,
                            "source_weights_dir_name": source_dir_name,
                            "rs_weights_dir_name": rs_weights_dir_name,
                            **row,
                        }
                    )
                save_results_rows(
                    out_dir,
                    cert_rows,
                    base_name=f"cert_rows_ntu_{sanitize_token(dataset_name)}_{sanitize_token(model_name)}_{sanitize_token(source_dir_name)}",
                )

                row: Dict[str, Any] = {
                    "domain": "ntu",
                    "dataset": dataset_name,
                    "model": model_name,
                    "source_weights_dir_name": source_dir_name,
                    "rs_weights_dir_name": rs_weights_dir_name,
                    "source_checkpoint_path": str(checkpoint_path),
                    "weights_path": export_paths["latest_path"],
                    "sigma": float(sigma),
                    "noise_type": noise_type,
                    "amplitude_clamp_min": amplitude_clamp_min,
                    **model_profile,
                    **clean_final,
                    "smoothed_n_eval": int(cert_metrics["smoothed"].get("n_eval", 0)),
                    "smoothed_acc": float(cert_metrics["smoothed"].get("accuracy", 0.0)),
                    "smoothed_abstain_rate": float(cert_metrics["smoothed"].get("abstain_rate", 0.0)),
                    "cert_n": int(cert_metrics["cert_summary"].get("n_certified", 0)),
                    "cert_acc": float(cert_metrics["cert_summary"].get("accuracy", 0.0)),
                    "cert_abstain_rate": float(cert_metrics["cert_summary"].get("abstain_rate", 0.0)),
                    "cert_mean_radius": float(cert_metrics["cert_summary"].get("mean_radius", 0.0)),
                    "checkpoint_missing_keys": len(load_info.get("missing_keys", [])),
                    "checkpoint_unexpected_keys": len(load_info.get("unexpected_keys", [])),
                    "seed": int(applied_seed) if applied_seed is not None else None,
                }
                row.update(cert_metrics["cert_acc_by_radius"])
                augment_profile_row(row, profile_cfg)
                final_rows.append(row)


def _run_csi_bench(
    *,
    cfg: Mapping[str, Any],
    cfg_path: Optional[str],
    out_dir: str,
    run_ts: str,
    model_names: Sequence[str],
    device: torch.device,
    task_family: str,
    applied_seed: Optional[int],
    train_rows: List[Dict[str, Any]],
    final_rows: List[Dict[str, Any]],
    profile_cfg: Mapping[str, Any],
) -> None:
    csi_cfg = dict(cfg.get("csi_bench", {}) or {})
    weights_cfg = dict(cfg.get("weights", {}) or {})
    weights_csi = dict(weights_cfg.get("csi_bench", {}) or {})
    train_cfg = dict(cfg.get("training", {}) or {})
    rs_cfg = dict(cfg.get("certified_robustness", {}).get("randomized_smoothing", {}) or {})
    eval_cfg = _resolve_rs_eval_cfg(cfg)

    sigma = _resolve_sigma_for_domain(rs_cfg, domain="csi_bench")
    noise_type = str(rs_cfg.get("noise_type", "gaussian")).strip().lower()
    amplitude_clamp_min = rs_cfg.get("amplitude_clamp_min", 0.0)
    amplitude_clamp_min = None if amplitude_clamp_min is None else float(amplitude_clamp_min)

    task_names = parse_token_list(
        csi_cfg.get("task_names")
        or csi_cfg.get("tasks")
        or csi_cfg.get("task_name")
        or ",".join(default_csi_tasks(task_family))
    )
    task_names = [normalize_csi_bench_task_name(name) for name in task_names]
    if not task_names:
        task_names = default_csi_tasks(task_family)

    dataset_root = str(csi_cfg.get("dataset_root", cfg.get("common", {}).get("dataset_root", "datasets/CSI-Bench")))
    batch_size = int(csi_cfg.get("batch_size", 256))
    num_workers = _parse_optional_positive_int(csi_cfg.get("num_workers"))
    pin_memory_raw = csi_cfg.get("pin_memory")
    pin_memory = None if pin_memory_raw is None else parse_bool(pin_memory_raw, default=(device.type == "cuda"))
    persistent_workers_raw = csi_cfg.get("persistent_workers")
    persistent_workers = None if persistent_workers_raw is None else parse_bool(persistent_workers_raw, default=True)
    prefetch_factor = _parse_optional_positive_int(csi_cfg.get("prefetch_factor"))
    normalisation = str(csi_cfg.get("normalisation", "zscore"))
    splits_dir_name = str(csi_cfg.get("splits_dir", "splits"))
    target_subcarriers = int(csi_cfg.get("target_subcarriers", DEFAULT_CSI_BENCH_SUBCARRIERS))
    target_timesteps = int(csi_cfg.get("target_timesteps", DEFAULT_CSI_BENCH_TIMESTEPS))
    only_splits = parse_splits(csi_cfg.get("only_splits") or csi_cfg.get("splits"), default=["train_id", "val_id", "test_id"])
    eval_split_name = str(csi_cfg.get("eval_split_name", "test_id"))
    canonical_eval_split = normalise_csi_split_name(eval_split_name)
    loader_key_eval = loader_key_for_csi_split(eval_split_name)
    if canonical_eval_split not in {normalise_csi_split_name(split) for split in only_splits}:
        only_splits.append(canonical_eval_split)

    source_root = str(weights_csi.get("dataset_root", dataset_root))
    source_dir_names = as_weight_dir_list(
        weights_csi.get("source_weights_dir_names", weights_csi.get("weights_dir_names")),
        default=str(
            weights_csi.get("source_weights_dir_name")
            or weights_cfg.get("source_weights_dir_name")
            or weights_cfg.get("weights_dir_name")
            or DEFAULT_CLEAN_WEIGHTS_DIR
        ),
    )
    rs_weights_dir_name = str(
        weights_csi.get("rs_weights_dir_name")
        or weights_cfg.get("rs_weights_dir_name")
        or f"model_params_rs_{task_family}"
    )
    checkpoint_glob = weights_csi.get("checkpoint_glob")
    checkpoint_paths = weights_csi.get("checkpoint_paths")
    selection = normalise_selection_mode(weights_cfg.get("selection"))
    strict_load = parse_bool(weights_cfg.get("strict_load"), default=True)

    epochs = int(train_cfg.get("epochs", 20))
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    progress_bar = parse_bool(train_cfg.get("progress_bar"), default=False)
    use_amp = parse_bool(train_cfg.get("amp"), default=False)
    save_epochs = resolve_checkpoint_epochs(train_cfg, epochs)

    for task_name in task_names:
        for model_name in model_names:
            for source_dir_name in source_dir_names:
                print_training_start(
                    f"RS CSI-Bench ({task_name}/{model_name})",
                    {
                        "sigma": sigma,
                        "noise_type": noise_type,
                        "source_weights_dir": source_dir_name,
                        "rs_weights_dir": rs_weights_dir_name,
                        "eval_split_name": eval_split_name,
                        "device": str(device),
                    },
                )

                bundle = call_with_supported_kwargs(
                    load_csi_bench_supervised,
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
                if not bundle:
                    raise RuntimeError(f"Failed to load CSI-Bench task {task_name}")

                loaders = bundle["loaders"]
                train_loader = loaders.get("train")
                eval_loader = loaders.get(loader_key_eval)
                if train_loader is None:
                    raise RuntimeError(f"Training loader missing for task {task_name}")
                if eval_loader is None:
                    raise RuntimeError(
                        f"Requested eval split '{eval_split_name}' unavailable for {task_name}. "
                        f"Available={sorted(loaders.keys())}"
                    )

                num_classes = int(bundle["num_classes"])
                shape = infer_input_shape_from_loader(train_loader)
                arch_metadata: Dict[str, Any] = {"num_classes": num_classes}
                if shape is not None:
                    c, f, t = shape
                    arch_metadata.update({"channels": c, "freq_bins": f, "time_steps": t, "input_shape": [c, f, t]})

                source_base_dir = resolve_csi_bench_checkpoint_dir(
                    weights_dataset_root=source_root,
                    weights_dir_name=source_dir_name,
                    task_name=task_name,
                    model_name=model_name,
                )
                explicit_checkpoint = nested_checkpoint_override(checkpoint_paths, task_name, model_name)
                explicit_checkpoint = ensure_optuna_clean_checkpoint(
                    domain="csi_bench",
                    model_name=model_name,
                    weights_dir_name=source_dir_name,
                    weights_cfg=weights_cfg,
                    canonical_base_dir=source_base_dir,
                    alias_stems=[f"CSI-Bench_{model_name}"],
                    task_name=task_name,
                ) or explicit_checkpoint
                if shape is None:
                    inferred = infer_input_shape_from_loader(train_loader)
                    if inferred is None:
                        raise RuntimeError(f"Unable to infer input shape for {task_name}/{model_name}.")
                    c, f, t = inferred
                spec = DatasetSpec("CSI-Bench", channels=c, freq_bins=f, time_steps=t)
                model, _ = build_model_for_checkpoint(
                    domain="csi_bench",
                    model_name=model_name,
                    weights_dir_name=source_dir_name,
                    weights_cfg=weights_cfg,
                    spec=spec,
                    dataset_key_for_model="csi-bench",
                    num_classes=num_classes,
                    arch_metadata=arch_metadata,
                    task_name=task_name,
                )
                model = model.to(device)
                checkpoint_path = resolve_checkpoint(
                    base_dir=source_base_dir,
                    model_name=model_name,
                    selection=selection,
                    alias_stems=[f"CSI-Bench_{model_name}"],
                    checkpoint_glob=str(checkpoint_glob) if checkpoint_glob else None,
                    explicit_checkpoint=explicit_checkpoint,
                )
                load_info = load_checkpoint_into_model(
                    model=model,
                    checkpoint_path=str(checkpoint_path),
                    strict=strict_load,
                    device=device,
                    eval_mode=False,
                )

                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                trainer_cfg = copy.deepcopy(dict(cfg))
                trainer_rs_cfg = trainer_cfg.setdefault("certified_robustness", {}).setdefault("randomized_smoothing", {})
                trainer_rs_cfg["sigma"] = float(sigma)
                trainer_rs_cfg["noise_type"] = noise_type
                trainer_rs_cfg["amplitude_clamp_min"] = amplitude_clamp_min
                trainer = RandomizedSmoothingTrainer(model, optimizer, trainer_cfg, amp=use_amp)

                local_ckpt_dir = Path(out_dir) / "checkpoints" / "csi_bench" / normalize_csi_bench_task_name(task_name) / sanitize_token(model_name)
                local_ckpt_dir.mkdir(parents=True, exist_ok=True)
                final_local_ckpt = local_ckpt_dir / f"{sanitize_token(model_name)}_final.pt"

                for epoch in range(1, epochs + 1):
                    stats = trainer.train_one_epoch(train_loader, device=device, use_tqdm=progress_bar)
                    clean = clean_metrics(model=model, loader=eval_loader, device=device)

                    if epoch in save_epochs:
                        ckpt_path = local_ckpt_dir / f"{sanitize_token(model_name)}_epoch{epoch:03d}.pt"
                        torch.save(model.state_dict(), str(ckpt_path))
                        if epoch == epochs:
                            final_local_ckpt = ckpt_path

                    train_rows.append(
                        {
                            "domain": "csi_bench",
                            "dataset": task_name,
                            "task_name": task_name,
                            "model": model_name,
                            "source_weights_dir_name": source_dir_name,
                            "rs_weights_dir_name": rs_weights_dir_name,
                            "epoch": int(epoch),
                            "sigma": float(sigma),
                            "noise_type": noise_type,
                            "train_loss": float(stats.get("loss", 0.0)),
                            "train_loss_noisy": float(stats.get("loss_noisy", 0.0)),
                            "train_loss_clean": float(stats.get("loss_clean", 0.0)),
                            **clean,
                            "seed": int(applied_seed) if applied_seed is not None else None,
                        }
                    )

                if not final_local_ckpt.exists():
                    torch.save(model.state_dict(), str(final_local_ckpt))

                export_paths = export_csi_checkpoint(
                    dataset_root=source_root,
                    export_weights_dir_name=rs_weights_dir_name,
                    task_name=task_name,
                    model_name=model_name,
                    source_checkpoint=str(final_local_ckpt),
                    run_timestamp=run_ts,
                    run_dir=out_dir,
                    config_path=cfg_path,
                    stage="randomized_smoothing_export",
                    latest_alias=True,
                    checkpoint_payload=build_checkpoint_payload(model),
                )

                clean_final = clean_metrics(model=model, loader=eval_loader, device=device)
                model_profile = build_model_profile(model=model, loader=eval_loader, device=device, profile_cfg=profile_cfg)
                cert_metrics = _collect_cert_metrics(
                    model=model,
                    loader=eval_loader,
                    device=device,
                    num_classes=int(num_classes),
                    sigma=float(sigma),
                    noise_type=noise_type,
                    amplitude_clamp_min=amplitude_clamp_min,
                    eval_cfg=eval_cfg,
                )

                cert_rows = []
                for row in cert_metrics["cert_rows"]:
                    cert_rows.append(
                        {
                            "domain": "csi_bench",
                            "dataset": task_name,
                            "task_name": task_name,
                            "model": model_name,
                            "source_weights_dir_name": source_dir_name,
                            "rs_weights_dir_name": rs_weights_dir_name,
                            **row,
                        }
                    )
                save_results_rows(
                    out_dir,
                    cert_rows,
                    base_name=f"cert_rows_csi_bench_{sanitize_token(task_name)}_{sanitize_token(model_name)}_{sanitize_token(source_dir_name)}",
                )

                row: Dict[str, Any] = {
                    "domain": "csi_bench",
                    "dataset": task_name,
                    "task_name": task_name,
                    "model": model_name,
                    "source_weights_dir_name": source_dir_name,
                    "rs_weights_dir_name": rs_weights_dir_name,
                    "source_checkpoint_path": str(checkpoint_path),
                    "weights_path": export_paths["latest_path"],
                    "sigma": float(sigma),
                    "noise_type": noise_type,
                    "amplitude_clamp_min": amplitude_clamp_min,
                    **model_profile,
                    **clean_final,
                    "smoothed_n_eval": int(cert_metrics["smoothed"].get("n_eval", 0)),
                    "smoothed_acc": float(cert_metrics["smoothed"].get("accuracy", 0.0)),
                    "smoothed_abstain_rate": float(cert_metrics["smoothed"].get("abstain_rate", 0.0)),
                    "cert_n": int(cert_metrics["cert_summary"].get("n_certified", 0)),
                    "cert_acc": float(cert_metrics["cert_summary"].get("accuracy", 0.0)),
                    "cert_abstain_rate": float(cert_metrics["cert_summary"].get("abstain_rate", 0.0)),
                    "cert_mean_radius": float(cert_metrics["cert_summary"].get("mean_radius", 0.0)),
                    "checkpoint_missing_keys": len(load_info.get("missing_keys", [])),
                    "checkpoint_unexpected_keys": len(load_info.get("unexpected_keys", [])),
                    "seed": int(applied_seed) if applied_seed is not None else None,
                }
                row.update(cert_metrics["cert_acc_by_radius"])
                augment_profile_row(row, profile_cfg)
                final_rows.append(row)


def main(
    config_name: Optional[str] = None,
    *,
    default_config_relpath: Optional[str] = "experiments/config/campaign/defenses/randomized_smoothing/rs_har.yaml",
) -> None:
    cfg_arg = config_name or os.environ.get("EXP_RS_CONFIG")
    cfg_path = resolve_config_path_from_arg(cfg_arg)
    if cfg_path is None and default_config_relpath:
        cfg_path = resolve_config_path_from_arg(default_config_relpath)
    cfg = load_config(cfg_path)
    _apply_env_overrides(cfg)

    applied_seed = configure_from_env_and_config(cfg)
    configure_torch_performance(cfg)

    common_cfg = cfg.get("common", {})
    experiment_name = str(common_cfg.get("experiment_name", "randomized_smoothing"))

    data_cfg = cfg.get("data", {})
    device = _resolve_device(data_cfg.get("device"))
    profile_cfg = resolve_profile_cfg(cfg)

    model_cfg = cfg.get("model", {})
    model_names = resolve_supervised_models(
        as_list(model_cfg.get("architectures") or model_cfg.get("models"), default=["ResNet18", "LeNet", "BiLSTM"])
    )

    run_cfg = cfg.get("run", {})
    targets = {
        token.strip().lower().replace("-", "_")
        for token in as_list(run_cfg.get("targets"), default=["ntu", "csi_bench"])
    }
    task_family = str(run_cfg.get("task_family", "har")).strip().lower()
    if task_family not in {"har", "hid"}:
        raise ValueError("run.task_family must be one of: har, hid")

    results_cfg = dict(cfg.get("results", {}) or {})
    out_dir, ts, cfg_name = prepare_results_dir(
        experiment_name=experiment_name,
        cfg_path=cfg_path,
        mode=f"rs_{task_family}",
        job_dir=str(results_cfg.get("job_dir")) if results_cfg.get("job_dir") else None,
        job_name=str(results_cfg.get("job_name")) if results_cfg.get("job_name") else None,
        job_timestamp=str(results_cfg.get("job_timestamp")) if results_cfg.get("job_timestamp") else None,
        relative_parts=[] if parse_bool(results_cfg.get("simple_paths"), default=True) else [task_family],
    )
    save_config(out_dir, dict(cfg))

    print_training_start(
        "Randomized smoothing run",
        {
            "config_path": cfg_path,
            "output_dir": out_dir,
            "task_family": task_family,
            "device": str(device),
            "seed": applied_seed,
            "targets": sorted(targets),
            "models": list(model_names),
        },
    )

    train_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []

    if "ntu" in targets:
        _run_ntu(
            cfg=cfg,
            cfg_path=cfg_path,
            out_dir=out_dir,
            run_ts=ts,
            model_names=model_names,
            device=device,
            task_family=task_family,
            applied_seed=applied_seed,
            train_rows=train_rows,
            final_rows=final_rows,
            profile_cfg=profile_cfg,
        )
    if "csi_bench" in targets:
        _run_csi_bench(
            cfg=cfg,
            cfg_path=cfg_path,
            out_dir=out_dir,
            run_ts=ts,
            model_names=model_names,
            device=device,
            task_family=task_family,
            applied_seed=applied_seed,
            train_rows=train_rows,
            final_rows=final_rows,
            profile_cfg=profile_cfg,
        )

    save_results_rows(out_dir, train_rows, base_name="train_log")
    save_results_rows(out_dir, final_rows, base_name="summary_rows")
    save_run_meta(
        out_dir,
        {
            "timestamp": ts,
            "config_name": cfg_name,
            "models": list(model_names),
            "targets": sorted(targets),
            "task_family": task_family,
            "seed": int(applied_seed) if applied_seed is not None else None,
            "num_train_rows": int(len(train_rows)),
            "num_summary_rows": int(len(final_rows)),
        },
    )
    print(f"[INFO] Saved randomized smoothing results: {out_dir}")


if __name__ == "__main__":
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_name=arg)
