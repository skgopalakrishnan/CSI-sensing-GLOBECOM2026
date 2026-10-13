"""Adversarial training runner for APGD/APGDT on NTU and CSI-Bench."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from csi.attacks.evaluation import evaluate_pgdl2_over_budgets
from csi.attacks.utils.attack_utils import _compute_budgets
from csi.data import get_experiment_loaders
from csi.data.utils import filter_none_collate, load_csi_bench_supervised
from csi.defenses.adv_training.pgd_at import PGDATrainer
from csi.models.model_factory import build_model
from csi.models.utils.datasets import DatasetSpec
from csi.models.utils.experiment import load_config, resolve_config_path_from_arg, sanitize_token
from csi.models.utils.gpu_utils import _resolve_device
from csi.models.utils.results import (
    ASR_CI_FIELD_ORDER,
    WB_ATTACK_FIELD_ORDER,
    compute_asr_ci_summary,
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
from ..common.optuna_clean_runtime import build_model_for_checkpoint, ensure_optuna_clean_checkpoint
from ..common.edge_profile import augment_profile_row, profile_kwargs_from_cfg, resolve_profile_cfg
from ..common.train_models_utils import (
    DEFAULT_CSI_BENCH_SUBCARRIERS,
    DEFAULT_CSI_BENCH_TIMESTEPS,
    as_list,
    configure_torch_performance,
    infer_input_shape_from_loader,
    infer_ntu_shape,
    resolve_ntu_num_classes,
    resolve_project_path,
    resolve_supervised_models,
)
from ..attacks_raw_csi.whitebox_attack_apgd import (
    _apply_attack_kwargs_to_row,
    _format_attack_plan,
    _format_budget_diag,
)
from .common_runner import (
    as_weight_dir_list,
    build_checkpoint_payload,
    call_with_supported_kwargs,
    clean_metrics as clean_metrics_common,
    default_csi_tasks,
    default_ntu_datasets,
    export_csi_checkpoint,
    export_ntu_checkpoint,
    extract_state_dict,
    load_checkpoint_into_model,
    loader_key_for_csi_split,
    nested_checkpoint_override,
    normalise_csi_split_name,
    normalise_selection_mode,
    resolve_checkpoint,
    resolve_checkpoint_epochs as resolve_checkpoint_epochs_common,
    resolve_csi_bench_checkpoint_dir,
    safe_token,
    write_json,
)


_SUPPORTED_ATTACK_METHODS = {"apgd", "apgdt"}
_SUPPORTED_ATTACK_TYPES = {"untargeted", "targeted"}
_ATTACK_CFG_RESERVED_KEYS = {
    "method",
    "methods",
    "type",
    "task_family",
    "steps",
    "alpha",
    "alpha_fraction",
    "random_start",
    "n_restarts",
    "snr_db",
    "snr_dB",
    "target_snr_db",
    "eps_2",
    "epsilon",
    "eps",
}


def _call_with_supported_kwargs(func, *args, **kwargs):
    return call_with_supported_kwargs(func, *args, **kwargs)


def _normalise_selection_mode(value: Any) -> str:
    return normalise_selection_mode(value)


def _as_weight_dir_list(value: Any, *, default: str) -> List[str]:
    return as_weight_dir_list(value, default=default)


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    return extract_state_dict(payload)


def _load_checkpoint_into_model(
    *,
    model: nn.Module,
    checkpoint_path: str,
    strict: bool,
    device: torch.device,
) -> Dict[str, Any]:
    return load_checkpoint_into_model(
        model=model,
        checkpoint_path=checkpoint_path,
        strict=strict,
        device=device,
        eval_mode=False,
    )


def _nested_checkpoint_override(mapping: Any, key_1: str, key_2: str) -> Optional[str]:
    return nested_checkpoint_override(mapping, key_1, key_2)


def _resolve_checkpoint(
    *,
    base_dir: Path,
    model_name: str,
    selection: str,
    alias_stems: Sequence[str],
    checkpoint_glob: Optional[str],
    explicit_checkpoint: Optional[str],
) -> Path:
    return resolve_checkpoint(
        base_dir=base_dir,
        model_name=model_name,
        selection=selection,
        alias_stems=alias_stems,
        checkpoint_glob=checkpoint_glob,
        explicit_checkpoint=explicit_checkpoint,
    )


def _resolve_csi_bench_checkpoint_dir(
    *,
    weights_dataset_root: str,
    weights_dir_name: str,
    task_name: str,
    model_name: Optional[str] = None,
) -> Path:
    return resolve_csi_bench_checkpoint_dir(
        weights_dataset_root=weights_dataset_root,
        weights_dir_name=weights_dir_name,
        task_name=task_name,
        model_name=model_name,
    )


def _normalise_csi_split_name(split_name: str) -> str:
    return normalise_csi_split_name(split_name)


def _loader_key_for_csi_split(split_name: str) -> str:
    return loader_key_for_csi_split(split_name)


def _default_ntu_datasets(task_family: str) -> List[str]:
    return default_ntu_datasets(task_family)


def _default_csi_tasks(task_family: str) -> List[str]:
    return default_csi_tasks(task_family)


def _clean_metrics(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    return clean_metrics_common(model=model, loader=loader, device=device)


def _attack_kwargs_from_cfg(
    attack_cfg: Mapping[str, Any],
    *,
    attack_method: str,
    attack_type: str,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        key: value for key, value in dict(attack_cfg).items() if key not in _ATTACK_CFG_RESERVED_KEYS
    }
    kwargs.setdefault("enforce_har_hid_policy", True)
    if attack_method == "apgdt" and attack_type == "targeted":
        kwargs.setdefault("target_class_sweep", True)
    if attack_method == "apgd" and attack_type == "targeted":
        kwargs.setdefault("loss", "dlr")
    return kwargs


def _resolve_checkpoint_epochs(training_cfg: Mapping[str, Any], epochs: int) -> List[int]:
    return resolve_checkpoint_epochs_common(training_cfg, epochs)


def _safe_token(value: Any, fallback: str = "na") -> str:
    return safe_token(value, fallback=fallback)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)


def _export_ntu_adv_checkpoint(
    *,
    datasets_root: str,
    adv_weights_dir_name: str,
    dataset_name: str,
    model_name: str,
    source_checkpoint: str,
    run_timestamp: str,
    run_dir: str,
    config_path: Optional[str],
    latest_alias: bool = True,
    checkpoint_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    return export_ntu_checkpoint(
        datasets_root=datasets_root,
        export_weights_dir_name=adv_weights_dir_name,
        dataset_name=dataset_name,
        model_name=model_name,
        source_checkpoint=source_checkpoint,
        run_timestamp=run_timestamp,
        run_dir=run_dir,
        config_path=config_path,
        stage="adv_training_export",
        latest_alias=latest_alias,
        checkpoint_payload=checkpoint_payload,
    )


def _export_csi_adv_checkpoint(
    *,
    dataset_root: str,
    adv_weights_dir_name: str,
    task_name: str,
    model_name: str,
    source_checkpoint: str,
    run_timestamp: str,
    run_dir: str,
    config_path: Optional[str],
    latest_alias: bool = True,
    checkpoint_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    return export_csi_checkpoint(
        dataset_root=dataset_root,
        export_weights_dir_name=adv_weights_dir_name,
        task_name=task_name,
        model_name=model_name,
        source_checkpoint=source_checkpoint,
        run_timestamp=run_timestamp,
        run_dir=run_dir,
        config_path=config_path,
        stage="adv_training_export",
        latest_alias=latest_alias,
        checkpoint_payload=checkpoint_payload,
    )


def _apply_env_overrides(cfg: Dict[str, Any]) -> None:
    attack_cfg = cfg.setdefault("attack", {})
    model_cfg = cfg.setdefault("model", {})
    run_cfg = cfg.setdefault("run", {})
    ntu_cfg = cfg.setdefault("ntu", {})
    csi_cfg = cfg.setdefault("csi_bench", {})
    weights_cfg = cfg.setdefault("weights", {})

    method_env = os.environ.get("ADVTRAIN_ATTACK_METHOD")
    if method_env:
        attack_cfg["method"] = str(method_env).strip().lower()

    task_family_env = os.environ.get("ADVTRAIN_TASK_FAMILY")
    if task_family_env:
        attack_cfg["task_family"] = str(task_family_env).strip().lower()

    attack_type_env = os.environ.get("ADVTRAIN_ATTACK_TYPE")
    if attack_type_env:
        attack_cfg["type"] = str(attack_type_env).strip().lower()

    targets_env = os.environ.get("ADVTRAIN_TARGETS")
    if targets_env:
        run_cfg["targets"] = parse_token_list(targets_env)

    models_env = os.environ.get("ADVTRAIN_MODELS")
    if models_env:
        model_cfg["architectures"] = parse_token_list(models_env)

    ntu_datasets_env = os.environ.get("ADVTRAIN_NTU_DATASETS")
    if ntu_datasets_env:
        ntu_cfg["datasets"] = parse_token_list(ntu_datasets_env)

    csi_tasks_env = os.environ.get("ADVTRAIN_CSI_TASKS")
    if csi_tasks_env:
        csi_cfg["task_names"] = parse_token_list(csi_tasks_env)

    source_weights_env = os.environ.get("ADVTRAIN_SOURCE_WEIGHTS_DIR")
    if source_weights_env:
        weights_cfg["source_weights_dir_name"] = str(source_weights_env).strip()

    adv_weights_env = os.environ.get("ADVTRAIN_ADV_WEIGHTS_DIR")
    if adv_weights_env:
        weights_cfg["adv_weights_dir_name"] = str(adv_weights_env).strip()

    snr_env = os.environ.get("ATTACK_SNR_DB")
    eps2_env = os.environ.get("ATTACK_EPS2")
    if eps2_env is not None:
        try:
            attack_cfg["epsilon"] = float(eps2_env)
            attack_cfg.pop("snr_db", None)
            attack_cfg.pop("snr_dB", None)
        except Exception:
            pass
    elif snr_env is not None:
        try:
            attack_cfg["snr_dB"] = [float(snr_env)]
            attack_cfg.pop("epsilon", None)
        except Exception:
            pass

    steps_env = os.environ.get("ATTACK_STEPS")
    if steps_env is not None:
        try:
            attack_cfg["steps"] = int(steps_env)
        except Exception:
            pass

    alpha_env = os.environ.get("ATTACK_ALPHA")
    if alpha_env is not None:
        try:
            attack_cfg["alpha"] = float(alpha_env)
        except Exception:
            pass

    alpha_fraction_env = os.environ.get("ATTACK_ALPHA_FRACTION")
    if alpha_fraction_env is not None:
        try:
            attack_cfg["alpha_fraction"] = float(alpha_fraction_env)
        except Exception:
            pass

    n_restarts_env = os.environ.get("ATTACK_N_RESTARTS")
    if n_restarts_env is not None:
        try:
            attack_cfg["n_restarts"] = int(n_restarts_env)
        except Exception:
            pass

    random_start_env = os.environ.get("ATTACK_RANDOM_START")
    if random_start_env is not None:
        attack_cfg["random_start"] = parse_bool(random_start_env, default=True)


def _robust_eval_for_budget(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
    dataset_name: str,
    attack_type: str,
    budgets: List[Tuple[str, float]],
    steps: int,
    alpha: Optional[float],
    alpha_fraction: Optional[float],
    random_start: bool,
    n_restarts: int,
    dataloader_workers: Optional[int],
    max_batches: Optional[int],
    attack_method: str,
    attack_kwargs: Dict[str, Any],
    profile_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not budgets:
        return []
    results, _ = evaluate_pgdl2_over_budgets(
        model,
        loader,
        device,
        dataset_name=dataset_name,
        attack_type=attack_type,
        budgets=budgets,
        steps=steps,
        alpha_fraction=alpha_fraction,
        alpha=alpha,
        random_start=random_start,
        dataloader_workers=dataloader_workers,
        n_restarts=n_restarts,
        max_batches=max_batches,
        attack_method=attack_method,
        attack_kwargs=attack_kwargs,
        profile_kwargs=profile_kwargs,
    )
    return results


def _run_ntu_adv_training(
    *,
    cfg: Mapping[str, Any],
    cfg_path: Optional[str],
    out_dir: str,
    run_ts: str,
    model_names: Sequence[str],
    device: torch.device,
    attack_method: str,
    attack_type: str,
    task_family: str,
    budgets: List[Tuple[str, float]],
    attack_kwargs: Dict[str, Any],
    steps: int,
    alpha: Optional[float],
    alpha_fraction: Optional[float],
    random_start: bool,
    n_restarts: int,
    train_cfg: Mapping[str, Any],
    eval_cfg: Mapping[str, Any],
    weights_cfg: Mapping[str, Any],
    selection: str,
    strict_load: bool,
    applied_seed: Optional[int],
    train_rows: List[Dict[str, Any]],
    final_rows: List[Dict[str, Any]],
    profile_cfg: Mapping[str, Any],
) -> None:
    ntu_cfg = dict(cfg.get("ntu", {}) or {})
    weights_ntu = dict(weights_cfg.get("ntu", {}) or {})

    datasets = as_list(
        ntu_cfg.get("datasets") or ntu_cfg.get("dataset"),
        default=_default_ntu_datasets(task_family),
    )
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
    data_seed = int(ntu_cfg.get("seed", cfg.get("data", {}).get("seed", 42)))
    eval_split = str(ntu_cfg.get("eval_split", "test")).strip().lower()
    if eval_split not in {"test", "val"}:
        raise ValueError(f"Unsupported ntu.eval_split={eval_split!r}. Use 'test' or 'val'.")

    source_root = str(weights_ntu.get("datasets_root", root))
    source_dir_names = _as_weight_dir_list(
        weights_ntu.get("source_weights_dir_names", weights_ntu.get("weights_dir_names")),
        default=str(
            weights_ntu.get("source_weights_dir_name")
            or weights_cfg.get("source_weights_dir_name")
            or weights_cfg.get("weights_dir_name")
            or DEFAULT_CLEAN_WEIGHTS_DIR
        ),
    )
    adv_weights_dir_name = str(
        weights_ntu.get("adv_weights_dir_name")
        or weights_cfg.get("adv_weights_dir_name")
        or f"model_params_adv_{attack_method}"
    )
    checkpoint_glob = weights_ntu.get("checkpoint_glob")
    checkpoint_paths = weights_ntu.get("checkpoint_paths")

    epochs = int(train_cfg.get("epochs", 20))
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    progress_bar = parse_bool(train_cfg.get("progress_bar"), default=False)
    use_amp = parse_bool(train_cfg.get("amp"), default=False)
    eval_during_training = parse_bool(train_cfg.get("eval_during_training"), default=False)
    save_epochs = _resolve_checkpoint_epochs(train_cfg, epochs)

    max_batches_raw = eval_cfg.get("max_batches")
    try:
        max_batches = int(max_batches_raw) if max_batches_raw is not None else None
        if max_batches is not None and max_batches <= 0:
            max_batches = None
    except Exception:
        max_batches = None
    max_batches_train_eval_raw = eval_cfg.get("max_batches_during_training", 1)
    try:
        max_batches_train_eval = int(max_batches_train_eval_raw) if max_batches_train_eval_raw is not None else 1
        if max_batches_train_eval <= 0:
            max_batches_train_eval = 1
    except Exception:
        max_batches_train_eval = 1

    eval_n_restarts_raw = eval_cfg.get("n_restarts")
    try:
        eval_n_restarts = int(eval_n_restarts_raw) if eval_n_restarts_raw is not None else int(n_restarts)
    except Exception:
        eval_n_restarts = int(n_restarts)
    eval_n_restarts = max(1, eval_n_restarts)

    for dataset_name in datasets:
        for model_name in model_names:
            for source_dir_name in source_dir_names:
                print_training_start(
                    f"AdvTrain NTU ({dataset_name}/{model_name})",
                    {
                        "attack_method": attack_method,
                        "attack_type": attack_type,
                        "data_seed": data_seed,
                        "attack_plan": _format_attack_plan(
                            steps=steps,
                            alpha=alpha,
                            alpha_fraction=alpha_fraction,
                            random_start=random_start,
                            n_restarts=n_restarts,
                            attack_kwargs=attack_kwargs,
                        ),
                        "source_weights_dir": source_dir_name,
                        "adv_weights_dir": adv_weights_dir_name,
                        "eval_split": eval_split,
                        "device": str(device),
                    },
                )

                bundle = _call_with_supported_kwargs(
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
                val_loader = bundle.val_loader
                test_loader = bundle.test_loader
                eval_loader = val_loader if eval_split == "val" else test_loader
                if train_loader is None:
                    raise RuntimeError(f"NTU train loader missing for {dataset_name}.")
                if eval_loader is None:
                    raise RuntimeError(f"Requested NTU eval split '{eval_split}' unavailable for {dataset_name}.")

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

                source_base_dir = resolve_project_path(source_root) / dataset_name / source_dir_name
                explicit_checkpoint = _nested_checkpoint_override(checkpoint_paths, dataset_name, model_name)
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
                    channels, freq_bins, time_steps = inferred
                spec = DatasetSpec(dataset_name, channels=channels, freq_bins=freq_bins, time_steps=time_steps)
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
                checkpoint_path = _resolve_checkpoint(
                    base_dir=source_base_dir,
                    model_name=model_name,
                    selection=selection,
                    alias_stems=[f"{dataset_name}_{model_name}"],
                    checkpoint_glob=str(checkpoint_glob) if checkpoint_glob else None,
                    explicit_checkpoint=explicit_checkpoint,
                )
                load_info = _load_checkpoint_into_model(
                    model=model,
                    checkpoint_path=str(checkpoint_path),
                    strict=strict_load,
                    device=device,
                )

                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                trainer_cfg = copy.deepcopy(dict(cfg))
                trainer_attack_cfg = trainer_cfg.setdefault("attack", {})
                trainer_attack_cfg["method"] = attack_method
                trainer_attack_cfg["steps"] = int(steps)
                if alpha is not None:
                    trainer_attack_cfg["alpha"] = float(alpha)
                if alpha_fraction is not None:
                    trainer_attack_cfg["alpha_fraction"] = float(alpha_fraction)
                trainer_attack_cfg["random_start"] = bool(random_start)
                trainer_attack_cfg["n_restarts"] = int(n_restarts)
                if attack_method == "apgdt":
                    trainer_attack_cfg.setdefault("loss", "dlr")
                else:
                    trainer_attack_cfg.setdefault("loss", "ce")
                trainer = PGDATrainer(model, optimizer, trainer_cfg, amp=use_amp)

                local_ckpt_dir = Path(out_dir) / "checkpoints" / "ntu" / dataset_name / sanitize_token(model_name)
                local_ckpt_dir.mkdir(parents=True, exist_ok=True)
                final_local_ckpt = local_ckpt_dir / f"{sanitize_token(model_name)}_final.pt"

                for epoch in range(1, epochs + 1):
                    stats = trainer.train_one_epoch(train_loader, device, use_tqdm=progress_bar)
                    clean = _clean_metrics(model=model, loader=eval_loader, device=device)

                    robust_acc_quick = None
                    if eval_during_training and budgets:
                        eval_attack_kwargs = dict(attack_kwargs)
                        eval_attack_kwargs.setdefault("n_classes", int(num_classes))
                        quick_results = _robust_eval_for_budget(
                            model=model,
                            loader=eval_loader,
                            device=device,
                            dataset_name=dataset_name,
                            attack_type=attack_type,
                            budgets=[budgets[0]],
                            steps=steps,
                            alpha=alpha,
                            alpha_fraction=alpha_fraction,
                            random_start=random_start,
                            n_restarts=eval_n_restarts,
                            dataloader_workers=num_workers,
                            max_batches=max_batches_train_eval,
                            attack_method=attack_method,
                            attack_kwargs=eval_attack_kwargs,
                            profile_kwargs=profile_kwargs_from_cfg(profile_cfg, enabled=False),
                        )
                        if quick_results:
                            robust_acc_quick = 1.0 - float(quick_results[0].get("asr", 0.0))

                    local_ckpt_path = None
                    if epoch in save_epochs:
                        local_ckpt_path = local_ckpt_dir / f"{sanitize_token(model_name)}_epoch{epoch:03d}.pt"
                        torch.save(model.state_dict(), str(local_ckpt_path))
                        if epoch == epochs:
                            final_local_ckpt = local_ckpt_path

                    print(
                        f"[AT][EPOCH] domain=ntu dataset={dataset_name} model={model_name} "
                        f"source_weights={source_dir_name} epoch={epoch}/{epochs} "
                        f"loss={float(stats.get('loss', 0.0)):.5f} clean_acc={clean['clean_acc']:.4f}"
                        + (f" robust_acc={float(robust_acc_quick):.4f}" if robust_acc_quick is not None else "")
                    )

                    train_rows.append(
                        {
                            "domain": "ntu",
                            "dataset": dataset_name,
                            "model": model_name,
                            "source_weights_dir_name": source_dir_name,
                            "adv_weights_dir_name": adv_weights_dir_name,
                            "epoch": int(epoch),
                            "attack_method": attack_method,
                            "attack_type": attack_type,
                            "train_loss": float(stats.get("loss", 0.0)),
                            "train_loss_adv": float(stats.get("loss_adv", 0.0)),
                            "clean_acc": clean["clean_acc"],
                            "clean_loss": clean["clean_loss"],
                            "clean_f1_macro": clean["clean_f1_macro"],
                            "clean_f1_weighted": clean["clean_f1_weighted"],
                            "robust_acc_quick": float(robust_acc_quick) if robust_acc_quick is not None else None,
                            "local_checkpoint_path": str(local_ckpt_path) if local_ckpt_path else None,
                            "source_checkpoint_path": str(checkpoint_path),
                            "seed": int(applied_seed) if applied_seed is not None else None,
                        }
                    )
                    _apply_attack_kwargs_to_row(train_rows[-1], attack_kwargs)

                if not final_local_ckpt.exists():
                    torch.save(model.state_dict(), str(final_local_ckpt))

                export_paths = _export_ntu_adv_checkpoint(
                    datasets_root=source_root,
                    adv_weights_dir_name=adv_weights_dir_name,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    source_checkpoint=str(final_local_ckpt),
                    run_timestamp=run_ts,
                    run_dir=out_dir,
                    config_path=cfg_path,
                    latest_alias=True,
                    checkpoint_payload=build_checkpoint_payload(model),
                )

                clean_final = _clean_metrics(model=model, loader=eval_loader, device=device)
                eval_attack_kwargs = dict(attack_kwargs)
                eval_attack_kwargs.setdefault("n_classes", int(num_classes))
                res_list = _robust_eval_for_budget(
                    model=model,
                    loader=eval_loader,
                    device=device,
                    dataset_name=dataset_name,
                    attack_type=attack_type,
                    budgets=budgets,
                    steps=steps,
                    alpha=alpha,
                    alpha_fraction=alpha_fraction,
                    random_start=random_start,
                    n_restarts=eval_n_restarts,
                    dataloader_workers=num_workers,
                    max_batches=max_batches,
                    attack_method=attack_method,
                    attack_kwargs=eval_attack_kwargs,
                    profile_kwargs=profile_kwargs_from_cfg(profile_cfg),
                )
                if not res_list:
                    print(
                        f"[AT][SKIP] domain=ntu dataset={dataset_name} model={model_name} "
                        "reason=no_results_from_evaluation"
                    )
                for res in res_list:
                    diag = res.get("diag", {})
                    if not isinstance(diag, dict):
                        diag = {}
                    row: Dict[str, Any] = {
                        "domain": "ntu",
                        "dataset": dataset_name,
                        "model": model_name,
                        "source_weights_dir_name": source_dir_name,
                        "adv_weights_dir_name": adv_weights_dir_name,
                        "source_checkpoint_path": str(checkpoint_path),
                        "weights_path": export_paths["latest_path"],
                        "attack_method": attack_method,
                        "attack_mode": attack_type,
                        "targeted": bool(attack_type == "targeted"),
                        "budget_label": str(res.get("budget_label")),
                        "eps_2": float(res.get("eps_2", 0.0)),
                        "steps": int(steps),
                        "alpha": float(alpha) if alpha is not None else None,
                        "alpha_fraction": float(alpha_fraction) if alpha_fraction is not None else None,
                        "n_restarts": int(eval_n_restarts),
                        "random_start": bool(random_start),
                        "asr": float(res.get("asr", 0.0)),
                        "device": str(device),
                        "seed": int(applied_seed) if applied_seed is not None else None,
                        **clean_final,
                        **diag,
                        "checkpoint_missing_keys": len(load_info.get("missing_keys", [])),
                        "checkpoint_unexpected_keys": len(load_info.get("unexpected_keys", [])),
                    }
                    _apply_attack_kwargs_to_row(row, eval_attack_kwargs)
                    augment_profile_row(row, profile_cfg)
                    for metric_key in (
                        "clean_f1_macro",
                        "clean_f1_weighted",
                        "clean_f1_micro",
                        "adv_f1_macro",
                        "adv_f1_weighted",
                        "adv_f1_micro",
                    ):
                        if metric_key in res:
                            row[metric_key] = res[metric_key]
                    final_rows.append(row)
                    print(
                        f"[AT][DONE] domain=ntu dataset={dataset_name} model={model_name} "
                        f"budget={str(res.get('budget_label'))} asr={float(res.get('asr', 0.0)):.4f} "
                        f"n_clean_correct={int(diag.get('n_clean_correct', 0))} "
                        f"{_format_budget_diag(diag)}"
                    )


def _run_csi_bench_adv_training(
    *,
    cfg: Mapping[str, Any],
    cfg_path: Optional[str],
    out_dir: str,
    run_ts: str,
    model_names: Sequence[str],
    device: torch.device,
    attack_method: str,
    attack_type: str,
    task_family: str,
    budgets: List[Tuple[str, float]],
    attack_kwargs: Dict[str, Any],
    steps: int,
    alpha: Optional[float],
    alpha_fraction: Optional[float],
    random_start: bool,
    n_restarts: int,
    train_cfg: Mapping[str, Any],
    eval_cfg: Mapping[str, Any],
    weights_cfg: Mapping[str, Any],
    selection: str,
    strict_load: bool,
    applied_seed: Optional[int],
    train_rows: List[Dict[str, Any]],
    final_rows: List[Dict[str, Any]],
    profile_cfg: Mapping[str, Any],
) -> None:
    csi_cfg = dict(cfg.get("csi_bench", {}) or {})
    weights_csi = dict(weights_cfg.get("csi_bench", {}) or {})

    task_names = parse_token_list(
        csi_cfg.get("task_names")
        or csi_cfg.get("tasks")
        or csi_cfg.get("task_name")
        or ",".join(_default_csi_tasks(task_family))
    )
    task_names = [normalize_csi_bench_task_name(name) for name in task_names]
    if not task_names:
        task_names = _default_csi_tasks(task_family)

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
    only_splits = parse_splits(
        csi_cfg.get("only_splits") or csi_cfg.get("splits"),
        default=["train_id", "val_id", "test_id"],
    )
    eval_split_name = str(csi_cfg.get("eval_split_name", "test_id"))
    canonical_eval_split = _normalise_csi_split_name(eval_split_name)
    loader_key_eval = _loader_key_for_csi_split(eval_split_name)
    if canonical_eval_split not in {_normalise_csi_split_name(split) for split in only_splits}:
        only_splits.append(canonical_eval_split)

    source_root = str(weights_csi.get("dataset_root", dataset_root))
    source_dir_names = _as_weight_dir_list(
        weights_csi.get("source_weights_dir_names", weights_csi.get("weights_dir_names")),
        default=str(
            weights_csi.get("source_weights_dir_name")
            or weights_cfg.get("source_weights_dir_name")
            or weights_cfg.get("weights_dir_name")
            or DEFAULT_CLEAN_WEIGHTS_DIR
        ),
    )
    adv_weights_dir_name = str(
        weights_csi.get("adv_weights_dir_name")
        or weights_cfg.get("adv_weights_dir_name")
        or f"model_params_adv_{attack_method}"
    )
    checkpoint_glob = weights_csi.get("checkpoint_glob")
    checkpoint_paths = weights_csi.get("checkpoint_paths")

    epochs = int(train_cfg.get("epochs", 20))
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    progress_bar = parse_bool(train_cfg.get("progress_bar"), default=False)
    use_amp = parse_bool(train_cfg.get("amp"), default=False)
    eval_during_training = parse_bool(train_cfg.get("eval_during_training"), default=False)
    save_epochs = _resolve_checkpoint_epochs(train_cfg, epochs)

    max_batches_raw = eval_cfg.get("max_batches")
    try:
        max_batches = int(max_batches_raw) if max_batches_raw is not None else None
        if max_batches is not None and max_batches <= 0:
            max_batches = None
    except Exception:
        max_batches = None
    max_batches_train_eval_raw = eval_cfg.get("max_batches_during_training", 1)
    try:
        max_batches_train_eval = int(max_batches_train_eval_raw) if max_batches_train_eval_raw is not None else 1
        if max_batches_train_eval <= 0:
            max_batches_train_eval = 1
    except Exception:
        max_batches_train_eval = 1

    eval_n_restarts_raw = eval_cfg.get("n_restarts")
    try:
        eval_n_restarts = int(eval_n_restarts_raw) if eval_n_restarts_raw is not None else int(n_restarts)
    except Exception:
        eval_n_restarts = int(n_restarts)
    eval_n_restarts = max(1, eval_n_restarts)

    for task_name in task_names:
        for model_name in model_names:
            for source_dir_name in source_dir_names:
                print_training_start(
                    f"AdvTrain CSI-Bench ({task_name}/{model_name})",
                    {
                        "attack_method": attack_method,
                        "attack_type": attack_type,
                        "attack_plan": _format_attack_plan(
                            steps=steps,
                            alpha=alpha,
                            alpha_fraction=alpha_fraction,
                            random_start=random_start,
                            n_restarts=n_restarts,
                            attack_kwargs=attack_kwargs,
                        ),
                        "source_weights_dir": source_dir_name,
                        "adv_weights_dir": adv_weights_dir_name,
                        "eval_split_name": eval_split_name,
                        "device": str(device),
                    },
                )

                bundle = _call_with_supported_kwargs(
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
                if bundle is False or bundle is None:
                    raise RuntimeError(f"Failed to load CSI-Bench task {task_name}")

                loaders = bundle["loaders"]
                train_loader = loaders.get("train")
                eval_loader = loaders.get(loader_key_eval)
                if train_loader is None:
                    raise RuntimeError(f"Training loader missing for task {task_name}")
                if eval_loader is None:
                    raise RuntimeError(
                        f"Requested CSI-Bench eval split '{eval_split_name}' unavailable for {task_name}. "
                        f"Available={sorted(loaders.keys())}"
                    )

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

                source_base_dir = _resolve_csi_bench_checkpoint_dir(
                    weights_dataset_root=source_root,
                    weights_dir_name=source_dir_name,
                    task_name=task_name,
                    model_name=model_name,
                )
                explicit_checkpoint = _nested_checkpoint_override(checkpoint_paths, task_name, model_name)
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
                    channels, freq_bins, time_steps = inferred
                spec = DatasetSpec("CSI-Bench", channels=channels, freq_bins=freq_bins, time_steps=time_steps)
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
                checkpoint_path = _resolve_checkpoint(
                    base_dir=source_base_dir,
                    model_name=model_name,
                    selection=selection,
                    alias_stems=[f"CSI-Bench_{model_name}"],
                    checkpoint_glob=str(checkpoint_glob) if checkpoint_glob else None,
                    explicit_checkpoint=explicit_checkpoint,
                )
                load_info = _load_checkpoint_into_model(
                    model=model,
                    checkpoint_path=str(checkpoint_path),
                    strict=strict_load,
                    device=device,
                )

                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                trainer_cfg = copy.deepcopy(dict(cfg))
                trainer_attack_cfg = trainer_cfg.setdefault("attack", {})
                trainer_attack_cfg["method"] = attack_method
                trainer_attack_cfg["steps"] = int(steps)
                if alpha is not None:
                    trainer_attack_cfg["alpha"] = float(alpha)
                if alpha_fraction is not None:
                    trainer_attack_cfg["alpha_fraction"] = float(alpha_fraction)
                trainer_attack_cfg["random_start"] = bool(random_start)
                trainer_attack_cfg["n_restarts"] = int(n_restarts)
                if attack_method == "apgdt":
                    trainer_attack_cfg.setdefault("loss", "dlr")
                else:
                    trainer_attack_cfg.setdefault("loss", "ce")
                trainer = PGDATrainer(model, optimizer, trainer_cfg, amp=use_amp)

                local_ckpt_dir = Path(out_dir) / "checkpoints" / "csi_bench" / normalize_csi_bench_task_name(task_name) / sanitize_token(model_name)
                local_ckpt_dir.mkdir(parents=True, exist_ok=True)
                final_local_ckpt = local_ckpt_dir / f"{sanitize_token(model_name)}_final.pt"

                for epoch in range(1, epochs + 1):
                    stats = trainer.train_one_epoch(train_loader, device, use_tqdm=progress_bar)
                    clean = _clean_metrics(model=model, loader=eval_loader, device=device)

                    robust_acc_quick = None
                    if eval_during_training and budgets:
                        eval_attack_kwargs = dict(attack_kwargs)
                        eval_attack_kwargs.setdefault("n_classes", int(num_classes))
                        quick_results = _robust_eval_for_budget(
                            model=model,
                            loader=eval_loader,
                            device=device,
                            dataset_name=task_name,
                            attack_type=attack_type,
                            budgets=[budgets[0]],
                            steps=steps,
                            alpha=alpha,
                            alpha_fraction=alpha_fraction,
                            random_start=random_start,
                            n_restarts=eval_n_restarts,
                            dataloader_workers=num_workers,
                            max_batches=max_batches_train_eval,
                            attack_method=attack_method,
                            attack_kwargs=eval_attack_kwargs,
                            profile_kwargs=profile_kwargs_from_cfg(profile_cfg, enabled=False),
                        )
                        if quick_results:
                            robust_acc_quick = 1.0 - float(quick_results[0].get("asr", 0.0))

                    local_ckpt_path = None
                    if epoch in save_epochs:
                        local_ckpt_path = local_ckpt_dir / f"{sanitize_token(model_name)}_epoch{epoch:03d}.pt"
                        torch.save(model.state_dict(), str(local_ckpt_path))
                        if epoch == epochs:
                            final_local_ckpt = local_ckpt_path

                    print(
                        f"[AT][EPOCH] domain=csi_bench task={task_name} model={model_name} "
                        f"source_weights={source_dir_name} epoch={epoch}/{epochs} "
                        f"loss={float(stats.get('loss', 0.0)):.5f} clean_acc={clean['clean_acc']:.4f}"
                        + (f" robust_acc={float(robust_acc_quick):.4f}" if robust_acc_quick is not None else "")
                    )

                    train_rows.append(
                        {
                            "domain": "csi_bench",
                            "dataset": task_name,
                            "task_name": task_name,
                            "model": model_name,
                            "source_weights_dir_name": source_dir_name,
                            "adv_weights_dir_name": adv_weights_dir_name,
                            "epoch": int(epoch),
                            "attack_method": attack_method,
                            "attack_type": attack_type,
                            "train_loss": float(stats.get("loss", 0.0)),
                            "train_loss_adv": float(stats.get("loss_adv", 0.0)),
                            "clean_acc": clean["clean_acc"],
                            "clean_loss": clean["clean_loss"],
                            "clean_f1_macro": clean["clean_f1_macro"],
                            "clean_f1_weighted": clean["clean_f1_weighted"],
                            "robust_acc_quick": float(robust_acc_quick) if robust_acc_quick is not None else None,
                            "local_checkpoint_path": str(local_ckpt_path) if local_ckpt_path else None,
                            "source_checkpoint_path": str(checkpoint_path),
                            "seed": int(applied_seed) if applied_seed is not None else None,
                        }
                    )
                    _apply_attack_kwargs_to_row(train_rows[-1], attack_kwargs)

                if not final_local_ckpt.exists():
                    torch.save(model.state_dict(), str(final_local_ckpt))

                export_paths = _export_csi_adv_checkpoint(
                    dataset_root=source_root,
                    adv_weights_dir_name=adv_weights_dir_name,
                    task_name=task_name,
                    model_name=model_name,
                    source_checkpoint=str(final_local_ckpt),
                    run_timestamp=run_ts,
                    run_dir=out_dir,
                    config_path=cfg_path,
                    latest_alias=True,
                    checkpoint_payload=build_checkpoint_payload(model),
                )

                clean_final = _clean_metrics(model=model, loader=eval_loader, device=device)
                eval_attack_kwargs = dict(attack_kwargs)
                eval_attack_kwargs.setdefault("n_classes", int(num_classes))
                res_list = _robust_eval_for_budget(
                    model=model,
                    loader=eval_loader,
                    device=device,
                    dataset_name=task_name,
                    attack_type=attack_type,
                    budgets=budgets,
                    steps=steps,
                    alpha=alpha,
                    alpha_fraction=alpha_fraction,
                    random_start=random_start,
                    n_restarts=eval_n_restarts,
                    dataloader_workers=num_workers,
                    max_batches=max_batches,
                    attack_method=attack_method,
                    attack_kwargs=eval_attack_kwargs,
                    profile_kwargs=profile_kwargs_from_cfg(profile_cfg),
                )
                if not res_list:
                    print(
                        f"[AT][SKIP] domain=csi_bench task={task_name} model={model_name} "
                        "reason=no_results_from_evaluation"
                    )
                for res in res_list:
                    diag = res.get("diag", {})
                    if not isinstance(diag, dict):
                        diag = {}
                    row: Dict[str, Any] = {
                        "domain": "csi_bench",
                        "dataset": task_name,
                        "task_name": task_name,
                        "model": model_name,
                        "source_weights_dir_name": source_dir_name,
                        "adv_weights_dir_name": adv_weights_dir_name,
                        "source_checkpoint_path": str(checkpoint_path),
                        "weights_path": export_paths["latest_path"],
                        "attack_method": attack_method,
                        "attack_mode": attack_type,
                        "targeted": bool(attack_type == "targeted"),
                        "budget_label": str(res.get("budget_label")),
                        "eps_2": float(res.get("eps_2", 0.0)),
                        "steps": int(steps),
                        "alpha": float(alpha) if alpha is not None else None,
                        "alpha_fraction": float(alpha_fraction) if alpha_fraction is not None else None,
                        "n_restarts": int(eval_n_restarts),
                        "random_start": bool(random_start),
                        "asr": float(res.get("asr", 0.0)),
                        "device": str(device),
                        "seed": int(applied_seed) if applied_seed is not None else None,
                        **clean_final,
                        **diag,
                        "checkpoint_missing_keys": len(load_info.get("missing_keys", [])),
                        "checkpoint_unexpected_keys": len(load_info.get("unexpected_keys", [])),
                    }
                    _apply_attack_kwargs_to_row(row, eval_attack_kwargs)
                    augment_profile_row(row, profile_cfg)
                    for metric_key in (
                        "clean_f1_macro",
                        "clean_f1_weighted",
                        "clean_f1_micro",
                        "adv_f1_macro",
                        "adv_f1_weighted",
                        "adv_f1_micro",
                    ):
                        if metric_key in res:
                            row[metric_key] = res[metric_key]
                    final_rows.append(row)
                    print(
                        f"[AT][DONE] domain=csi_bench task={task_name} model={model_name} "
                        f"budget={str(res.get('budget_label'))} asr={float(res.get('asr', 0.0)):.4f} "
                        f"n_clean_correct={int(diag.get('n_clean_correct', 0))} "
                        f"{_format_budget_diag(diag)}"
                    )


def main(
    config_name: Optional[str] = None,
    *,
    attack_method_override: Optional[str] = None,
    default_config_relpath: Optional[str] = "experiments/config/adv_trained/adv_train_apgd_har.yaml",
) -> None:
    cfg_arg = config_name or os.environ.get("EXP_ADVTRAIN_CONFIG")
    cfg_path = resolve_config_path_from_arg(cfg_arg)
    if cfg_path is None and default_config_relpath:
        cfg_path = resolve_config_path_from_arg(default_config_relpath)
    cfg = load_config(cfg_path)
    _apply_env_overrides(cfg)

    applied_seed = configure_from_env_and_config(cfg)
    configure_torch_performance(cfg)

    common_cfg = cfg.get("common", {})
    experiment_name = str(common_cfg.get("experiment_name", "adv_train_apgd"))

    data_cfg = cfg.get("data", {})
    device = _resolve_device(data_cfg.get("device"))
    profile_cfg = resolve_profile_cfg(cfg)

    model_cfg = cfg.get("model", {})
    model_names = resolve_supervised_models(
        as_list(model_cfg.get("architectures") or model_cfg.get("models"), default=["ResNet18", "LeNet", "BiLSTM"])
    )

    run_cfg = cfg.get("run", {})
    targets_raw = as_list(run_cfg.get("targets"), default=["ntu", "csi_bench"])
    targets = {token.strip().lower().replace("-", "_") for token in targets_raw}

    attack_cfg = dict(cfg.get("attack", {}) or {})
    attack_method = str(
        attack_method_override
        or attack_cfg.get("method")
        or (attack_cfg.get("methods") or ["apgd"])[0]
    ).strip().lower()
    if attack_method not in _SUPPORTED_ATTACK_METHODS:
        raise ValueError(f"Unsupported attack.method={attack_method!r}. Use one of {sorted(_SUPPORTED_ATTACK_METHODS)}.")

    task_family = str(attack_cfg.get("task_family", "har")).strip().lower()
    if task_family not in {"har", "hid"}:
        raise ValueError("attack.task_family must be 'har' or 'hid'.")

    default_attack_type = "targeted" if task_family == "hid" else "untargeted"
    attack_type = str(attack_cfg.get("type", default_attack_type)).strip().lower()
    if attack_type not in _SUPPORTED_ATTACK_TYPES:
        raise ValueError(f"Unsupported attack.type={attack_type!r}. Use one of {sorted(_SUPPORTED_ATTACK_TYPES)}.")

    if attack_type == "targeted":
        print("[WARN] PGD-AT inner training uses untargeted labels. attack.type is used for robustness evaluation.")

    steps = int(attack_cfg.get("steps", 100))
    alpha = attack_cfg.get("alpha")
    alpha = float(alpha) if alpha is not None else None
    alpha_fraction = attack_cfg.get("alpha_fraction")
    alpha_fraction = float(alpha_fraction) if alpha_fraction is not None else None
    random_start = bool(attack_cfg.get("random_start", True))
    n_restarts = int(attack_cfg.get("n_restarts", 1))

    budgets = _compute_budgets(attack_cfg)
    attack_kwargs = _attack_kwargs_from_cfg(
        attack_cfg,
        attack_method=attack_method,
        attack_type=attack_type,
    )

    weights_cfg = dict(cfg.get("weights", {}) or {})
    selection = _normalise_selection_mode(weights_cfg.get("selection"))
    strict_load = parse_bool(weights_cfg.get("strict_load"), default=True)

    results_cfg = dict(cfg.get("results", {}) or {})
    results_job_dir = results_cfg.get("job_dir")
    results_job_name = results_cfg.get("job_name")
    results_job_timestamp = results_cfg.get("job_timestamp")
    simple_paths = parse_bool(results_cfg.get("simple_paths"), default=True)
    relative_parts: List[str]
    if simple_paths:
        relative_parts = []
    else:
        relative_parts = [attack_method, attack_type, task_family]
        if applied_seed is not None:
            relative_parts.append(f"seed_{int(applied_seed)}")
    out_dir, ts, cfg_name = prepare_results_dir(
        experiment_name=experiment_name,
        cfg_path=cfg_path,
        mode=f"{attack_method}_{task_family}",
        job_dir=str(results_job_dir) if results_job_dir else None,
        job_name=str(results_job_name) if results_job_name else None,
        job_timestamp=str(results_job_timestamp) if results_job_timestamp else None,
        relative_parts=relative_parts,
    )
    save_config(out_dir, dict(cfg))

    train_cfg = dict(cfg.get("training", {}) or {})
    eval_cfg = dict(cfg.get("eval", {}) or {})

    print_training_start(
        "Adversarial training run",
        {
            "config_path": cfg_path,
            "output_dir": out_dir,
            "attack_method": attack_method,
            "attack_type": attack_type,
            "task_family": task_family,
            "budgets": [label for (label, _) in budgets],
            "steps": steps,
            "alpha": alpha,
            "alpha_fraction": alpha_fraction,
            "random_start": random_start,
            "n_restarts": n_restarts,
            "attack_kwargs": dict(attack_kwargs),
            "device": str(device),
            "seed": applied_seed,
            "targets": sorted(targets),
            "models": list(model_names),
        },
    )

    train_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []
    if "ntu" in targets:
        _run_ntu_adv_training(
            cfg=cfg,
            cfg_path=cfg_path,
            out_dir=out_dir,
            run_ts=ts,
            model_names=model_names,
            device=device,
            attack_method=attack_method,
            attack_type=attack_type,
            task_family=task_family,
            budgets=budgets,
            attack_kwargs=attack_kwargs,
            steps=steps,
            alpha=alpha,
            alpha_fraction=alpha_fraction,
            random_start=random_start,
            n_restarts=n_restarts,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            weights_cfg=weights_cfg,
            selection=selection,
            strict_load=strict_load,
            applied_seed=applied_seed,
            train_rows=train_rows,
            final_rows=final_rows,
            profile_cfg=profile_cfg,
        )
    if "csi_bench" in targets:
        _run_csi_bench_adv_training(
            cfg=cfg,
            cfg_path=cfg_path,
            out_dir=out_dir,
            run_ts=ts,
            model_names=model_names,
            device=device,
            attack_method=attack_method,
            attack_type=attack_type,
            task_family=task_family,
            budgets=budgets,
            attack_kwargs=attack_kwargs,
            steps=steps,
            alpha=alpha,
            alpha_fraction=alpha_fraction,
            random_start=random_start,
            n_restarts=n_restarts,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            weights_cfg=weights_cfg,
            selection=selection,
            strict_load=strict_load,
            applied_seed=applied_seed,
            train_rows=train_rows,
            final_rows=final_rows,
            profile_cfg=profile_cfg,
        )

    save_results_rows(out_dir, train_rows, base_name="train_log")
    save_results_rows(out_dir, final_rows, base_name="summary_rows", field_order=WB_ATTACK_FIELD_ORDER)
    ci_rows = compute_asr_ci_summary(final_rows, config_name=cfg_name)
    if ci_rows:
        save_results_rows(out_dir, ci_rows, base_name="asr_ci95", field_order=ASR_CI_FIELD_ORDER)

    save_run_meta(
        out_dir,
        {
            "timestamp": ts,
            "config_name": cfg_name,
            "attack": {
                "method": attack_method,
                "type": attack_type,
                "task_family": task_family,
                "steps": steps,
                "alpha": alpha,
                "alpha_fraction": alpha_fraction,
                "random_start": random_start,
                "n_restarts": n_restarts,
                "kwargs": attack_kwargs,
                "budgets": [label for (label, _) in budgets],
            },
            "weights": {
                "selection": selection,
                "strict_load": bool(strict_load),
            },
            "models": list(model_names),
            "targets": sorted(targets),
            "seed": int(applied_seed) if applied_seed is not None else None,
            "num_train_rows": int(len(train_rows)),
            "num_summary_rows": int(len(final_rows)),
        },
    )
    print(f"[INFO] Saved adversarial training results: {out_dir}")


if __name__ == "__main__":
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_name=arg)
