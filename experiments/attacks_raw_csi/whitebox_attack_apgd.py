"""White-box robustness benchmarking for APGD/APGDT on NTU and CSI-Bench."""

from __future__ import annotations

import os
import inspect
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from csi.attacks.evaluation import evaluate_pgdl2_over_budgets
from csi.attacks.utils.attack_utils import _compute_budgets
from csi.data import get_experiment_loaders
from csi.data.utils import filter_none_collate, load_csi_bench_supervised
from csi.models.model_factory import build_model
from csi.models.supervised import evaluate_classifier
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


_SUPPORTED_ATTACK_METHODS = {"apgd", "apgdt"}
_SUPPORTED_ATTACK_TYPES = {"untargeted", "targeted"}
_ATTACK_CFG_RESERVED_KEYS = {
    "method",
    "methods",
    "type",
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
    """Call `func` while dropping kwargs unsupported by its current signature."""
    try:
        sig = inspect.signature(func)
    except Exception:
        return func(*args, **kwargs)

    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    has_var_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
    if has_var_kw:
        # Signature provides no useful keyword filtering (common for shim wrappers).
        return func(*args, **kwargs)
    supported = {name for name, p in params.items() if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    if has_var_args:
        return func(*args, **filtered)
    return func(*args, **filtered)


def _normalise_selection_mode(value: Any) -> str:
    key = str(value or "auto").strip().lower()
    if key not in {"auto", "alias", "latest"}:
        raise ValueError("weights.selection must be one of: auto, alias, latest")
    return key


def _as_weight_dir_list(value: Any, *, default: str) -> List[str]:
    if value is None:
        return [default]
    if isinstance(value, str):
        out = [token.strip() for token in value.split(",") if token.strip()]
        return out if out else [default]
    if isinstance(value, (list, tuple)):
        out = [str(token).strip() for token in value if str(token).strip()]
        return out if out else [default]
    return [default]


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], Mapping):
            return payload["model_state_dict"]  # type: ignore[return-value]
        if "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
            return payload["state_dict"]  # type: ignore[return-value]
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload  # type: ignore[return-value]
    raise ValueError(f"Unsupported checkpoint format: {type(payload)!r}")


def _load_checkpoint_into_model(
    *,
    model: nn.Module,
    checkpoint_path: str,
    strict: bool,
    device: torch.device,
) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = _extract_state_dict(payload)
    result = model.load_state_dict(state, strict=bool(strict))
    model.to(device)
    model.eval()
    return {
        "checkpoint_path": checkpoint_path,
        "strict": bool(strict),
        "num_state_keys": int(len(state)),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def _nested_checkpoint_override(mapping: Any, key_1: str, key_2: str) -> Optional[str]:
    if not isinstance(mapping, Mapping):
        return None
    level_1 = mapping.get(key_1)
    if not isinstance(level_1, Mapping):
        return None
    value = level_1.get(key_2)
    return str(value) if value else None


def _resolve_checkpoint(
    *,
    base_dir: Path,
    model_name: str,
    selection: str,
    alias_stems: Sequence[str],
    checkpoint_glob: Optional[str],
    explicit_checkpoint: Optional[str],
) -> Path:
    if explicit_checkpoint:
        explicit_path = resolve_project_path(str(explicit_checkpoint))
        if not explicit_path.exists():
            raise FileNotFoundError(f"Explicit checkpoint does not exist: {explicit_path}")
        return explicit_path

    if not base_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {base_dir}")

    selection_key = _normalise_selection_mode(selection)
    if selection_key in {"auto", "alias"}:
        for stem in alias_stems:
            alias_path = base_dir / f"{stem}.pt"
            if alias_path.exists():
                return alias_path
        if selection_key == "alias":
            raise FileNotFoundError(
                f"No alias checkpoint found in {base_dir}. Tried stems: {list(alias_stems)}"
            )

    pattern = checkpoint_glob or "*.pt"
    candidates = [path for path in base_dir.glob(pattern) if path.is_file() and path.suffix == ".pt"]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matched pattern '{pattern}' in {base_dir}")

    model_token = sanitize_token(model_name).lower()
    token_hits = [path for path in candidates if model_token in sanitize_token(path.stem).lower()]
    pool = token_hits if token_hits else candidates
    return max(pool, key=lambda path: path.stat().st_mtime)


def _resolve_csi_bench_checkpoint_dir(
    *,
    weights_dataset_root: str,
    weights_dir_name: str,
    task_name: str,
) -> Path:
    root = resolve_project_path(weights_dataset_root)
    candidate_dirs = [
        root / weights_dir_name / task_name,
        (root / "CSI-Bench") / weights_dir_name / task_name,
    ]
    if root.name == "CSI-Bench":
        candidate_dirs.append((root / "CSI-Bench") / weights_dir_name / task_name)
    else:
        candidate_dirs.append(root / "CSI-Bench" / "CSI-Bench" / weights_dir_name / task_name)

    seen: set[Path] = set()
    ordered: List[Path] = []
    for candidate in candidate_dirs:
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    for candidate in ordered:
        if candidate.exists():
            return candidate
    return ordered[0]


def _normalise_csi_split_name(split_name: str) -> str:
    key = str(split_name).strip().lower().replace("-", "_")
    if key in {"train", "train_id"}:
        return "train_id"
    if key in {"val", "validation", "val_id"}:
        return "val_id"
    if key in {"test", "test_id"}:
        return "test_id"
    return key


def _loader_key_for_csi_split(split_name: str) -> str:
    split_key = _normalise_csi_split_name(split_name)
    if split_key == "train_id":
        return "train"
    if split_key == "val_id":
        return "val"
    if split_key == "test_id":
        return "test"
    if split_key.startswith("test_"):
        return split_key
    return f"test_{split_key}"


def _default_ntu_datasets(task_family: str) -> List[str]:
    if task_family == "hid":
        return ["NTU-Fi-HumanID"]
    return ["NTU-Fi_HAR"]


def _default_csi_tasks(task_family: str) -> List[str]:
    if task_family == "hid":
        return ["HumanIdentification"]
    return ["HumanActivityRecognition"]


def _clean_metrics(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    metrics, _ = evaluate_classifier(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        use_tqdm=False,
        collect_predictions=False,
        collect_probabilities=False,
        desc="clean_eval",
    )
    return {
        "clean_acc": float(metrics.get("accuracy", 0.0)),
        "clean_loss": float(metrics.get("loss", 0.0)),
        "clean_f1_macro": float(metrics.get("f1_macro", 0.0)),
        "clean_f1_weighted": float(metrics.get("f1_weighted", 0.0)),
    }


def _attack_kwargs_from_cfg(
    attack_cfg: Mapping[str, Any],
    *,
    attack_method: str,
    attack_type: str,
    default_free_targeted: bool,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        key: value
        for key, value in dict(attack_cfg).items()
        if key not in _ATTACK_CFG_RESERVED_KEYS
    }
    kwargs.setdefault("enforce_har_hid_policy", True)
    if attack_method == "apgdt":
        kwargs.setdefault("target_class_sweep", bool(default_free_targeted))
    if attack_method == "apgd" and attack_type == "targeted":
        kwargs.setdefault("loss", "dlr")
    return kwargs


def _format_log_value(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _format_attack_plan(
    *,
    steps: int,
    alpha: Optional[float],
    alpha_fraction: Optional[float],
    random_start: bool,
    n_restarts: int,
    attack_kwargs: Mapping[str, Any],
) -> str:
    parts = [
        f"steps={int(steps)}",
        f"alpha={_format_log_value(alpha, digits=6)}",
        f"alpha_fraction={_format_log_value(alpha_fraction, digits=6)}",
        f"random_start={_format_log_value(bool(random_start))}",
        f"n_restarts={int(n_restarts)}",
    ]
    for key in ("loss", "target_class_sweep", "n_target_classes", "constraint_mode", "use_mmd_loss", "mmd_weight"):
        if key in attack_kwargs:
            parts.append(f"{key}={_format_log_value(attack_kwargs.get(key), digits=6)}")
    return " ".join(parts)


def _apply_attack_kwargs_to_row(row: Dict[str, Any], attack_kwargs: Mapping[str, Any]) -> None:
    for key in ("loss", "target_class_sweep", "n_target_classes", "constraint_mode", "use_mmd_loss", "mmd_weight"):
        if key in attack_kwargs:
            row[key] = attack_kwargs[key]


def _format_budget_diag(diag: Mapping[str, Any]) -> str:
    fields = [
        ("attack_defined_success_all", diag.get("attack_defined_success_rate_all_mean", diag.get("attack_success_rate_mean"))),
        ("l2_ratio", diag.get("l2_ratio_mean")),
        ("utilization", diag.get("utilization_mean")),
        ("post_l2", diag.get("post_l2_mean")),
        ("budget_l2", diag.get("budget_l2_mean")),
    ]
    parts = [f"{name}={_format_log_value(value)}" for name, value in fields]
    if diag.get("budget_diag_error"):
        parts.append(f"budget_diag_error={diag['budget_diag_error']}")
    return " ".join(parts)


def _run_ntu_whitebox(
    *,
    cfg: Mapping[str, Any],
    model_names: Sequence[str],
    device: torch.device,
    attack_method: str,
    attack_type: str,
    budgets: List[Tuple[str, float]],
    steps: int,
    alpha: Optional[float],
    alpha_fraction: Optional[float],
    random_start: bool,
    n_restarts: int,
    attack_kwargs: Dict[str, Any],
    results_rows: List[Dict[str, Any]],
    strict_load: bool,
    selection: str,
    weights_cfg: Mapping[str, Any],
    applied_seed: Optional[int],
    max_batches: Optional[int],
    profile_cfg: Mapping[str, Any],
) -> None:
    ntu_cfg = dict(cfg.get("ntu", {}) or {})
    weights_ntu = dict(weights_cfg.get("ntu", {}) or {})

    default_family = "hid" if attack_type == "targeted" else "har"
    datasets = as_list(ntu_cfg.get("datasets") or ntu_cfg.get("dataset"), default=_default_ntu_datasets(default_family))
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

    weights_root = str(weights_ntu.get("datasets_root", root))
    weights_dir_names = _as_weight_dir_list(
        weights_ntu.get("weights_dir_names", weights_ntu.get("weights_dir_name")),
        default=str(weights_cfg.get("weights_dir_name") or DEFAULT_CLEAN_WEIGHTS_DIR),
    )
    checkpoint_glob = weights_ntu.get("checkpoint_glob")
    checkpoint_paths = weights_ntu.get("checkpoint_paths")

    for dataset_name in datasets:
        for model_name in model_names:
            for weights_dir_name in weights_dir_names:
                print_training_start(
                    f"White-box NTU ({dataset_name}/{model_name})",
                    {
                        "attack_method": attack_method,
                        "attack_type": attack_type,
                        "data_seed": data_seed,
                        "weights_dir_name": weights_dir_name,
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
                if eval_loader is None:
                    raise RuntimeError(f"Requested NTU eval split '{eval_split}' unavailable for {dataset_name}.")
                if train_loader is None:
                    raise RuntimeError(f"NTU train loader missing for {dataset_name}.")

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

                base_dir = resolve_project_path(weights_root) / dataset_name / weights_dir_name
                explicit_checkpoint = _nested_checkpoint_override(checkpoint_paths, dataset_name, model_name)
                explicit_checkpoint = ensure_optuna_clean_checkpoint(
                    domain="ntu",
                    model_name=model_name,
                    weights_dir_name=weights_dir_name,
                    weights_cfg=weights_cfg,
                    canonical_base_dir=base_dir,
                    alias_stems=[f"{dataset_name}_{model_name}"],
                    dataset_name=dataset_name,
                ) or explicit_checkpoint
                if shape is None:
                    inferred = infer_input_shape_from_loader(train_loader)
                    if inferred is None:
                        raise RuntimeError(f"Unable to infer input shape for {dataset_name}/{model_name}.")
                    channels, freq_bins, time_steps = inferred
                spec = DatasetSpec(dataset_name, channels=channels, freq_bins=freq_bins, time_steps=time_steps)
                checkpoint_path = _resolve_checkpoint(
                    base_dir=base_dir,
                    model_name=model_name,
                    selection=selection,
                    alias_stems=[f"{dataset_name}_{model_name}"],
                    checkpoint_glob=str(checkpoint_glob) if checkpoint_glob else None,
                    explicit_checkpoint=explicit_checkpoint,
                )
                model, _ = build_model_for_checkpoint(
                    domain="ntu",
                    model_name=model_name,
                    weights_dir_name=weights_dir_name,
                    weights_cfg=weights_cfg,
                    spec=spec,
                    dataset_key_for_model=dataset_name,
                    num_classes=num_classes,
                    arch_metadata=arch_metadata,
                    dataset_name=dataset_name,
                    checkpoint_path=str(checkpoint_path),
                )
                model = model.to(device)
                load_info = _load_checkpoint_into_model(
                    model=model,
                    checkpoint_path=str(checkpoint_path),
                    strict=strict_load,
                    device=device,
                )

                clean = _clean_metrics(model=model, loader=eval_loader, device=device)
                attack_kwargs_case = dict(attack_kwargs)
                attack_kwargs_case.setdefault("n_classes", int(num_classes))

                print(
                    f"[WB][START] domain=ntu dataset={dataset_name} model={model_name} "
                    f"weights_dir={weights_dir_name} attack={attack_method}/{attack_type} "
                    f"{_format_attack_plan(steps=steps, alpha=alpha, alpha_fraction=alpha_fraction, random_start=random_start, n_restarts=n_restarts, attack_kwargs=attack_kwargs_case)}"
                )
                res_list, _ = evaluate_pgdl2_over_budgets(
                    model,
                    eval_loader,
                    device,
                    dataset_name=dataset_name,
                    attack_type=attack_type,
                    budgets=budgets,
                    steps=steps,
                    alpha_fraction=alpha_fraction,
                    alpha=alpha,
                    random_start=random_start,
                    dataloader_workers=num_workers,
                    n_restarts=n_restarts,
                    max_batches=max_batches,
                    attack_method=attack_method,
                    attack_kwargs=attack_kwargs_case,
                    profile_kwargs=profile_kwargs_from_cfg(profile_cfg),
                )
                if not res_list:
                    print(
                        f"[WB][SKIP] domain=ntu dataset={dataset_name} model={model_name} "
                        f"reason=no_results_from_evaluation"
                    )
                for res in res_list:
                    diag = res.get("diag", {})
                    if not isinstance(diag, dict):
                        diag = {}
                    row: Dict[str, Any] = {
                        "dataset": dataset_name,
                        "domain": "ntu",
                        "model": model_name,
                        "weights_dir_name": weights_dir_name,
                        "checkpoint_path": str(checkpoint_path),
                        "eval_split": eval_split,
                        "attack_method": attack_method,
                        "attack_mode": attack_type,
                        "targeted": bool(attack_type == "targeted"),
                        "budget_label": str(res.get("budget_label")),
                        "eps_2": float(res.get("eps_2", 0.0)),
                        "steps": int(steps),
                        "alpha": float(alpha) if alpha is not None else None,
                        "alpha_fraction": float(alpha_fraction) if alpha_fraction is not None else None,
                        "n_restarts": int(n_restarts),
                        "random_start": bool(random_start),
                        "asr": float(res.get("asr", 0.0)),
                        "device": str(device),
                        "seed": int(applied_seed) if applied_seed is not None else None,
                        **clean,
                        **diag,
                        "checkpoint_missing_keys": len(load_info.get("missing_keys", [])),
                        "checkpoint_unexpected_keys": len(load_info.get("unexpected_keys", [])),
                    }
                    _apply_attack_kwargs_to_row(row, attack_kwargs_case)
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
                    results_rows.append(row)
                    print(
                        f"[WB][DONE] domain=ntu dataset={dataset_name} model={model_name} "
                        f"budget={str(res.get('budget_label'))} asr={float(res.get('asr', 0.0)):.4f} "
                        f"n_clean_correct={int(diag.get('n_clean_correct', 0))} "
                        f"{_format_budget_diag(diag)}"
                    )


def _run_csi_bench_whitebox(
    *,
    cfg: Mapping[str, Any],
    model_names: Sequence[str],
    device: torch.device,
    attack_method: str,
    attack_type: str,
    budgets: List[Tuple[str, float]],
    steps: int,
    alpha: Optional[float],
    alpha_fraction: Optional[float],
    random_start: bool,
    n_restarts: int,
    attack_kwargs: Dict[str, Any],
    results_rows: List[Dict[str, Any]],
    strict_load: bool,
    selection: str,
    weights_cfg: Mapping[str, Any],
    applied_seed: Optional[int],
    max_batches: Optional[int],
    profile_cfg: Mapping[str, Any],
) -> None:
    csi_cfg = dict(cfg.get("csi_bench", {}) or {})
    weights_csi = dict(weights_cfg.get("csi_bench", {}) or {})

    default_family = "hid" if attack_type == "targeted" else "har"
    task_names = parse_token_list(
        csi_cfg.get("task_names")
        or csi_cfg.get("tasks")
        or csi_cfg.get("task_name")
        or ",".join(_default_csi_tasks(default_family))
    )
    task_names = [normalize_csi_bench_task_name(name) for name in task_names]
    if not task_names:
        task_names = _default_csi_tasks(default_family)

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
    eval_split_name = str(csi_cfg.get("eval_split_name", "test_id"))
    canonical_split = _normalise_csi_split_name(eval_split_name)
    loader_key = _loader_key_for_csi_split(eval_split_name)

    weights_dataset_root = str(weights_csi.get("dataset_root", dataset_root))
    weights_dir_names = _as_weight_dir_list(
        weights_csi.get("weights_dir_names", weights_csi.get("weights_dir_name")),
        default=str(weights_cfg.get("weights_dir_name") or DEFAULT_CLEAN_WEIGHTS_DIR),
    )
    checkpoint_glob = weights_csi.get("checkpoint_glob")
    checkpoint_paths = weights_csi.get("checkpoint_paths")

    for task_name in task_names:
        for model_name in model_names:
            for weights_dir_name in weights_dir_names:
                print_training_start(
                    f"White-box CSI-Bench ({task_name}/{model_name})",
                    {
                        "attack_method": attack_method,
                        "attack_type": attack_type,
                        "weights_dir_name": weights_dir_name,
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
                    only_splits=parse_splits([canonical_split], default=["test_id"]),
                    splits_dir_name=splits_dir_name,
                    collate_fn=filter_none_collate,
                )
                if not bundle:
                    raise RuntimeError(f"Failed to load CSI-Bench task {task_name}")
                loaders = bundle["loaders"]
                eval_loader = loaders.get(loader_key)
                if eval_loader is None:
                    raise RuntimeError(
                        f"Requested CSI-Bench split '{eval_split_name}' unavailable for {task_name}. "
                        f"Available={sorted(loaders.keys())}"
                    )

                num_classes = int(bundle["num_classes"])
                shape = infer_input_shape_from_loader(eval_loader)
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
                base_dir = _resolve_csi_bench_checkpoint_dir(
                    weights_dataset_root=weights_dataset_root,
                    weights_dir_name=weights_dir_name,
                    task_name=task_name,
                )
                explicit_checkpoint = _nested_checkpoint_override(checkpoint_paths, task_name, model_name)
                explicit_checkpoint = ensure_optuna_clean_checkpoint(
                    domain="csi_bench",
                    model_name=model_name,
                    weights_dir_name=weights_dir_name,
                    weights_cfg=weights_cfg,
                    canonical_base_dir=base_dir,
                    alias_stems=[f"CSI-Bench_{model_name}"],
                    task_name=task_name,
                ) or explicit_checkpoint
                if shape is None:
                    inferred = infer_input_shape_from_loader(eval_loader)
                    if inferred is None:
                        raise RuntimeError(f"Unable to infer input shape for {task_name}/{model_name}.")
                    channels, freq_bins, time_steps = inferred
                spec = DatasetSpec("CSI-Bench", channels=channels, freq_bins=freq_bins, time_steps=time_steps)
                checkpoint_path = _resolve_checkpoint(
                    base_dir=base_dir,
                    model_name=model_name,
                    selection=selection,
                    alias_stems=[f"CSI-Bench_{model_name}"],
                    checkpoint_glob=str(checkpoint_glob) if checkpoint_glob else None,
                    explicit_checkpoint=explicit_checkpoint,
                )
                model, _ = build_model_for_checkpoint(
                    domain="csi_bench",
                    model_name=model_name,
                    weights_dir_name=weights_dir_name,
                    weights_cfg=weights_cfg,
                    spec=spec,
                    dataset_key_for_model="csi-bench",
                    num_classes=num_classes,
                    arch_metadata=arch_metadata,
                    task_name=task_name,
                    checkpoint_path=str(checkpoint_path),
                )
                model = model.to(device)
                load_info = _load_checkpoint_into_model(
                    model=model,
                    checkpoint_path=str(checkpoint_path),
                    strict=strict_load,
                    device=device,
                )

                clean = _clean_metrics(model=model, loader=eval_loader, device=device)
                attack_kwargs_case = dict(attack_kwargs)
                attack_kwargs_case.setdefault("n_classes", int(num_classes))

                print(
                    f"[WB][START] domain=csi_bench dataset={task_name} model={model_name} "
                    f"weights_dir={weights_dir_name} attack={attack_method}/{attack_type} "
                    f"{_format_attack_plan(steps=steps, alpha=alpha, alpha_fraction=alpha_fraction, random_start=random_start, n_restarts=n_restarts, attack_kwargs=attack_kwargs_case)}"
                )
                res_list, _ = evaluate_pgdl2_over_budgets(
                    model,
                    eval_loader,
                    device,
                    dataset_name=task_name,
                    attack_type=attack_type,
                    budgets=budgets,
                    steps=steps,
                    alpha_fraction=alpha_fraction,
                    alpha=alpha,
                    random_start=random_start,
                    dataloader_workers=num_workers,
                    n_restarts=n_restarts,
                    max_batches=max_batches,
                    attack_method=attack_method,
                    attack_kwargs=attack_kwargs_case,
                    profile_kwargs=profile_kwargs_from_cfg(profile_cfg),
                )
                if not res_list:
                    print(
                        f"[WB][SKIP] domain=csi_bench dataset={task_name} model={model_name} "
                        f"reason=no_results_from_evaluation"
                    )
                for res in res_list:
                    diag = res.get("diag", {})
                    if not isinstance(diag, dict):
                        diag = {}
                    row: Dict[str, Any] = {
                        "dataset": task_name,
                        "domain": "csi_bench",
                        "task_name": task_name,
                        "model": model_name,
                        "weights_dir_name": weights_dir_name,
                        "checkpoint_path": str(checkpoint_path),
                        "eval_split": eval_split_name,
                        "attack_method": attack_method,
                        "attack_mode": attack_type,
                        "targeted": bool(attack_type == "targeted"),
                        "budget_label": str(res.get("budget_label")),
                        "eps_2": float(res.get("eps_2", 0.0)),
                        "steps": int(steps),
                        "alpha": float(alpha) if alpha is not None else None,
                        "alpha_fraction": float(alpha_fraction) if alpha_fraction is not None else None,
                        "n_restarts": int(n_restarts),
                        "random_start": bool(random_start),
                        "asr": float(res.get("asr", 0.0)),
                        "device": str(device),
                        "seed": int(applied_seed) if applied_seed is not None else None,
                        **clean,
                        **diag,
                        "checkpoint_missing_keys": len(load_info.get("missing_keys", [])),
                        "checkpoint_unexpected_keys": len(load_info.get("unexpected_keys", [])),
                    }
                    _apply_attack_kwargs_to_row(row, attack_kwargs_case)
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
                    results_rows.append(row)
                    print(
                        f"[WB][DONE] domain=csi_bench dataset={task_name} model={model_name} "
                        f"budget={str(res.get('budget_label'))} asr={float(res.get('asr', 0.0)):.4f} "
                        f"n_clean_correct={int(diag.get('n_clean_correct', 0))} "
                        f"{_format_budget_diag(diag)}"
                    )


def main(
    config_name: Optional[str] = None,
    *,
    attack_method_override: Optional[str] = None,
    default_attack_type: str = "untargeted",
    default_task_family: str = "har",
    default_free_targeted: bool = False,
    default_config_relpath: Optional[str] = None,
) -> None:
    cfg_arg = config_name or os.environ.get("EXP_ATTACK_CONFIG")
    cfg_path = resolve_config_path_from_arg(cfg_arg)
    if cfg_path is None and default_config_relpath:
        cfg_path = resolve_config_path_from_arg(default_config_relpath)
    cfg = load_config(cfg_path)

    applied_seed = configure_from_env_and_config(cfg)
    configure_torch_performance(cfg)

    common_cfg = cfg.get("common", {})
    experiment_name = str(common_cfg.get("experiment_name", "whitebox_apgd"))

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

    attack_type = str(attack_cfg.get("type", default_attack_type)).strip().lower()
    if attack_type not in _SUPPORTED_ATTACK_TYPES:
        raise ValueError(f"Unsupported attack.type={attack_type!r}. Use one of {sorted(_SUPPORTED_ATTACK_TYPES)}.")

    task_family = str(attack_cfg.get("task_family", default_task_family)).strip().lower()
    if task_family not in {"har", "hid"}:
        raise ValueError("attack.task_family must be 'har' or 'hid'.")

    steps = int(attack_cfg.get("steps", 100))
    alpha = attack_cfg.get("alpha")
    alpha = float(alpha) if alpha is not None else None
    alpha_fraction = attack_cfg.get("alpha_fraction")
    alpha_fraction = float(alpha_fraction) if alpha_fraction is not None else None
    random_start = bool(attack_cfg.get("random_start", True))
    n_restarts = int(attack_cfg.get("n_restarts", 1))
    max_batches_raw = (cfg.get("eval", {}) or {}).get("max_batches")
    try:
        max_batches = int(max_batches_raw) if max_batches_raw is not None else None
        if max_batches is not None and max_batches <= 0:
            max_batches = None
    except Exception:
        max_batches = None

    budgets = _compute_budgets(attack_cfg)
    attack_kwargs = _attack_kwargs_from_cfg(
        attack_cfg,
        attack_method=attack_method,
        attack_type=attack_type,
        default_free_targeted=default_free_targeted,
    )

    weights_cfg = dict(cfg.get("weights", {}) or {})
    selection = _normalise_selection_mode(weights_cfg.get("selection"))
    strict_load = parse_bool(weights_cfg.get("strict_load"), default=True)
    results_cfg = dict(cfg.get("results", {}) or {})
    results_job_dir = results_cfg.get("job_dir")
    results_job_name = results_cfg.get("job_name")
    results_job_timestamp = results_cfg.get("job_timestamp")
    simple_paths = parse_bool(results_cfg.get("simple_paths"), default=True)

    if simple_paths:
        # Keep default output structure short: .../<experiment>/<mode>/<timestamp>
        relative_parts: List[str] = []
    else:
        # Opt-in verbose structure for users who want attack/task/seed as path parts.
        relative_parts = [
            attack_method,
            attack_type,
            task_family,
        ]
        if applied_seed is not None:
            relative_parts.append(f"seed_{int(applied_seed)}")
    out_dir, ts, cfg_name = prepare_results_dir(
        experiment_name=experiment_name,
        cfg_path=cfg_path,
        mode=f"{attack_type}_{attack_method}",
        job_dir=str(results_job_dir) if results_job_dir else None,
        job_name=str(results_job_name) if results_job_name else None,
        job_timestamp=str(results_job_timestamp) if results_job_timestamp else None,
        relative_parts=relative_parts,
    )
    save_config(out_dir, dict(cfg))

    print_training_start(
        "White-box robustness run",
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
            "attack_kwargs": attack_kwargs,
            "device": str(device),
            "seed": applied_seed,
        },
    )

    results_rows: List[Dict[str, Any]] = []
    if "ntu" in targets:
        _run_ntu_whitebox(
            cfg=cfg,
            model_names=model_names,
            device=device,
            attack_method=attack_method,
            attack_type=attack_type,
            budgets=budgets,
            steps=steps,
            alpha=alpha,
            alpha_fraction=alpha_fraction,
            random_start=random_start,
            n_restarts=n_restarts,
            attack_kwargs=attack_kwargs,
            results_rows=results_rows,
            strict_load=strict_load,
            selection=selection,
            weights_cfg=weights_cfg,
            applied_seed=applied_seed,
            max_batches=max_batches,
            profile_cfg=profile_cfg,
        )
    if "csi_bench" in targets:
        _run_csi_bench_whitebox(
            cfg=cfg,
            model_names=model_names,
            device=device,
            attack_method=attack_method,
            attack_type=attack_type,
            budgets=budgets,
            steps=steps,
            alpha=alpha,
            alpha_fraction=alpha_fraction,
            random_start=random_start,
            n_restarts=n_restarts,
            attack_kwargs=attack_kwargs,
            results_rows=results_rows,
            strict_load=strict_load,
            selection=selection,
            weights_cfg=weights_cfg,
            applied_seed=applied_seed,
            max_batches=max_batches,
            profile_cfg=profile_cfg,
        )

    save_results_rows(out_dir, results_rows, base_name="summary_rows", field_order=WB_ATTACK_FIELD_ORDER)
    ci_rows = compute_asr_ci_summary(results_rows, config_name=cfg_name)
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
            "num_rows": int(len(results_rows)),
        },
    )
    print(f"[INFO] Saved white-box results: {out_dir}")


if __name__ == "__main__":
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_name=arg)
