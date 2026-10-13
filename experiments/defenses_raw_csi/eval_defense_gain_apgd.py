"""Evaluate baseline vs defended checkpoints under APGD threat and report ASR gain."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from csi.attacks.utils.attack_utils import _compute_budgets
from csi.models.utils.experiment import load_config, resolve_config_path_from_arg
from csi.models.utils.gpu_utils import _resolve_device
from csi.models.utils.results import (
    prepare_results_dir,
    save_config,
    save_results_rows,
    save_run_meta,
)
from csi.models.utils.seeding import configure_from_env_and_config

from ..common.edge_profile import resolve_profile_cfg
from ..attacks_raw_csi.whitebox_attack_apgd import (
    _SUPPORTED_ATTACK_METHODS,
    _SUPPORTED_ATTACK_TYPES,
    _attack_kwargs_from_cfg,
    _format_attack_plan,
    _normalise_selection_mode,
    _run_csi_bench_whitebox,
    _run_ntu_whitebox,
)
from ..common.shared_utils import parse_bool, print_training_start
from ..common.train_models_utils import as_list, configure_torch_performance, resolve_supervised_models


def _role_iteration_order(roles_cfg: Mapping[str, Any]) -> List[str]:
    role_names = [str(name).strip() for name in roles_cfg.keys() if str(name).strip()]
    if "baseline" in role_names:
        role_names = ["baseline"] + [name for name in role_names if name != "baseline"]
    return role_names


def _canonical_row_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("domain"),
        row.get("dataset"),
        row.get("task_name"),
        row.get("model"),
        row.get("budget_label"),
        row.get("attack_method"),
        row.get("attack_mode"),
        row.get("targeted"),
        row.get("seed"),
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _collect_rows_for_role(
    *,
    cfg: Mapping[str, Any],
    role_name: str,
    role_weights_cfg: Mapping[str, Any],
    model_names: Sequence[str],
    device,
    attack_method: str,
    attack_type: str,
    budgets: List[Tuple[str, float]],
    steps: int,
    alpha: Optional[float],
    alpha_fraction: Optional[float],
    random_start: bool,
    n_restarts: int,
    attack_kwargs: Dict[str, Any],
    strict_load: bool,
    selection: str,
    applied_seed: Optional[int],
    max_batches: Optional[int],
    targets: Sequence[str],
    out_rows: List[Dict[str, Any]],
    profile_cfg: Mapping[str, Any],
) -> None:
    before = len(out_rows)
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
            attack_kwargs=dict(attack_kwargs),
            results_rows=out_rows,
            strict_load=strict_load,
            selection=selection,
            weights_cfg=role_weights_cfg,
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
            attack_kwargs=dict(attack_kwargs),
            results_rows=out_rows,
            strict_load=strict_load,
            selection=selection,
            weights_cfg=role_weights_cfg,
            applied_seed=applied_seed,
            max_batches=max_batches,
            profile_cfg=profile_cfg,
        )

    for row in out_rows[before:]:
        row["weights_role"] = role_name
        row["defense_type"] = role_name if role_name != "baseline" else "baseline"


def _compute_gain_rows(role_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    baseline_by_key: Dict[Tuple[Any, ...], Mapping[str, Any]] = {}
    for row in role_rows:
        if str(row.get("weights_role")) != "baseline":
            continue
        baseline_by_key[_canonical_row_key(row)] = row

    gain_rows: List[Dict[str, Any]] = []
    for row in role_rows:
        role = str(row.get("weights_role", ""))
        if role == "baseline":
            continue
        base = baseline_by_key.get(_canonical_row_key(row))
        if base is None:
            continue

        base_asr = _safe_float(base.get("asr"), default=0.0)
        defended_asr = _safe_float(row.get("asr"), default=0.0)
        base_robust = 1.0 - base_asr
        defended_robust = 1.0 - defended_asr

        gain_rows.append(
            {
                "domain": row.get("domain"),
                "dataset": row.get("dataset"),
                "task_name": row.get("task_name"),
                "model": row.get("model"),
                "budget_label": row.get("budget_label"),
                "eps_2": row.get("eps_2"),
                "seed": row.get("seed"),
                "attack_method": row.get("attack_method"),
                "attack_mode": row.get("attack_mode"),
                "targeted": row.get("targeted"),
                "defense_type": role,
                "baseline_weights_dir_name": base.get("weights_dir_name"),
                "defended_weights_dir_name": row.get("weights_dir_name"),
                "baseline_checkpoint_path": base.get("checkpoint_path"),
                "defended_checkpoint_path": row.get("checkpoint_path"),
                "baseline_asr": base_asr,
                "defended_asr": defended_asr,
                "asr_gain": base_asr - defended_asr,
                "baseline_robust_acc": base_robust,
                "defended_robust_acc": defended_robust,
                "robust_acc_gain": defended_robust - base_robust,
                "baseline_clean_acc": _safe_float(base.get("clean_acc"), default=0.0),
                "defended_clean_acc": _safe_float(row.get("clean_acc"), default=0.0),
                "clean_acc_delta": _safe_float(row.get("clean_acc"), default=0.0) - _safe_float(base.get("clean_acc"), default=0.0),
                "baseline_model_param_count": base.get("model_param_count"),
                "defended_model_param_count": row.get("model_param_count"),
                "model_param_count_delta": _safe_float(row.get("model_param_count"), default=0.0)
                - _safe_float(base.get("model_param_count"), default=0.0),
                "baseline_model_param_kib": base.get("model_param_kib"),
                "defended_model_param_kib": row.get("model_param_kib"),
                "model_param_kib_delta": _safe_float(row.get("model_param_kib"), default=0.0)
                - _safe_float(base.get("model_param_kib"), default=0.0),
                "baseline_latency_ms_per_example": base.get("latency_ms_per_example"),
                "defended_latency_ms_per_example": row.get("latency_ms_per_example"),
                "latency_ms_per_example_delta": _safe_float(row.get("latency_ms_per_example"), default=0.0)
                - _safe_float(base.get("latency_ms_per_example"), default=0.0),
                "baseline_edge_inference_ms_est": base.get("edge_inference_ms_est"),
                "defended_edge_inference_ms_est": row.get("edge_inference_ms_est"),
                "edge_inference_ms_est_delta": _safe_float(row.get("edge_inference_ms_est"), default=0.0)
                - _safe_float(base.get("edge_inference_ms_est"), default=0.0),
                "baseline_tinyml_param_feasible": base.get("tinyml_param_feasible"),
                "defended_tinyml_param_feasible": row.get("tinyml_param_feasible"),
                "baseline_realtime_inference_feasible": base.get("realtime_inference_feasible"),
                "defended_realtime_inference_feasible": row.get("realtime_inference_feasible"),
                "baseline_edge_inference_feasible": base.get("edge_inference_feasible"),
                "defended_edge_inference_feasible": row.get("edge_inference_feasible"),
            }
        )
    return gain_rows


def main(config_name: Optional[str] = None) -> None:
    cfg_arg = config_name or os.environ.get("EXP_DEFENSE_EVAL_CONFIG")
    cfg_path = resolve_config_path_from_arg(cfg_arg)
    if cfg_path is None:
        cfg_path = resolve_config_path_from_arg("experiments/config/campaign/defenses/eval/defense_eval_har_20db.yaml")
    cfg = load_config(cfg_path)

    applied_seed = configure_from_env_and_config(cfg)
    configure_torch_performance(cfg)

    common_cfg = dict(cfg.get("common", {}) or {})
    experiment_name = str(common_cfg.get("experiment_name", "defense_eval_apgd"))

    data_cfg = dict(cfg.get("data", {}) or {})
    device = _resolve_device(data_cfg.get("device"))
    profile_cfg = resolve_profile_cfg(cfg)

    model_cfg = dict(cfg.get("model", {}) or {})
    model_names = resolve_supervised_models(
        as_list(model_cfg.get("architectures") or model_cfg.get("models"), default=["ResNet18", "LeNet", "BiLSTM"])
    )

    run_cfg = dict(cfg.get("run", {}) or {})
    targets_raw = as_list(run_cfg.get("targets"), default=["ntu", "csi_bench"])
    targets = sorted({token.strip().lower().replace("-", "_") for token in targets_raw})

    attack_cfg = dict(cfg.get("attack", {}) or {})
    attack_method = str(attack_cfg.get("method", "apgd")).strip().lower()
    if attack_method not in _SUPPORTED_ATTACK_METHODS:
        raise ValueError(f"Unsupported attack.method={attack_method!r}. Use one of {sorted(_SUPPORTED_ATTACK_METHODS)}.")

    attack_type = str(attack_cfg.get("type", "untargeted")).strip().lower()
    if attack_type not in _SUPPORTED_ATTACK_TYPES:
        raise ValueError(f"Unsupported attack.type={attack_type!r}. Use one of {sorted(_SUPPORTED_ATTACK_TYPES)}.")

    task_family = str(attack_cfg.get("task_family", "har")).strip().lower()
    if task_family not in {"har", "hid"}:
        raise ValueError("attack.task_family must be 'har' or 'hid'.")

    steps = int(attack_cfg.get("steps", 100))
    alpha = attack_cfg.get("alpha")
    alpha = float(alpha) if alpha is not None else None
    alpha_fraction = attack_cfg.get("alpha_fraction")
    alpha_fraction = float(alpha_fraction) if alpha_fraction is not None else None
    random_start = bool(attack_cfg.get("random_start", True))
    n_restarts = int(attack_cfg.get("n_restarts", 3))
    budgets = _compute_budgets(attack_cfg)
    attack_kwargs = _attack_kwargs_from_cfg(
        attack_cfg,
        attack_method=attack_method,
        attack_type=attack_type,
        default_free_targeted=False,
    )

    max_batches_raw = (cfg.get("eval", {}) or {}).get("max_batches")
    try:
        max_batches = int(max_batches_raw) if max_batches_raw is not None else None
        if max_batches is not None and max_batches <= 0:
            max_batches = None
    except Exception:
        max_batches = None

    weights_cfg = dict(cfg.get("weights", {}) or {})
    selection = _normalise_selection_mode(weights_cfg.get("selection"))
    strict_load = parse_bool(weights_cfg.get("strict_load"), default=True)
    roles_cfg = OrderedDict((str(k), v) for (k, v) in dict(weights_cfg.get("roles", {}) or {}).items())
    role_names = _role_iteration_order(roles_cfg)
    if not role_names:
        raise ValueError("weights.roles must define at least one role (including 'baseline').")
    if "baseline" not in role_names:
        raise ValueError("weights.roles must include a 'baseline' role.")

    results_cfg = dict(cfg.get("results", {}) or {})
    results_job_dir = results_cfg.get("job_dir")
    results_job_name = results_cfg.get("job_name")
    results_job_timestamp = results_cfg.get("job_timestamp")
    simple_paths = parse_bool(results_cfg.get("simple_paths"), default=True)
    relative_parts = [] if simple_paths else [attack_method, attack_type, task_family]
    if (not simple_paths) and (applied_seed is not None):
        relative_parts.append(f"seed_{int(applied_seed)}")

    out_dir, ts, cfg_name = prepare_results_dir(
        experiment_name=experiment_name,
        cfg_path=cfg_path,
        mode=f"defense_eval_{attack_type}_{attack_method}",
        job_dir=str(results_job_dir) if results_job_dir else None,
        job_name=str(results_job_name) if results_job_name else None,
        job_timestamp=str(results_job_timestamp) if results_job_timestamp else None,
        relative_parts=relative_parts,
    )
    save_config(out_dir, dict(cfg))

    print_training_start(
        "Defense gain evaluation run",
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
            "roles": list(role_names),
            "targets": list(targets),
            "seed": applied_seed,
        },
    )

    role_rows: List[Dict[str, Any]] = []
    for role_name in role_names:
        role_weights_cfg = dict(roles_cfg.get(role_name, {}) or {})
        print(
            f"[DEF-EVAL][ROLE] role={role_name} attack={attack_method}/{attack_type} "
            f"{_format_attack_plan(steps=steps, alpha=alpha, alpha_fraction=alpha_fraction, random_start=random_start, n_restarts=n_restarts, attack_kwargs=attack_kwargs)}"
        )
        before_role_rows = len(role_rows)
        _collect_rows_for_role(
            cfg=cfg,
            role_name=role_name,
            role_weights_cfg=role_weights_cfg,
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
            attack_kwargs=dict(attack_kwargs),
            strict_load=strict_load,
            selection=selection,
            applied_seed=applied_seed,
            max_batches=max_batches,
            targets=targets,
            out_rows=role_rows,
            profile_cfg=profile_cfg,
        )
        print(
            f"[DEF-EVAL][ROLE][DONE] role={role_name} rows_added={int(len(role_rows) - before_role_rows)}"
        )

    gain_rows = _compute_gain_rows(role_rows)

    save_results_rows(out_dir, role_rows, base_name="role_summary_rows")
    save_results_rows(out_dir, gain_rows, base_name="defense_gain_rows")
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
                "budgets": [label for (label, _) in budgets],
                "kwargs": dict(attack_kwargs),
            },
            "weights": {
                "selection": selection,
                "strict_load": bool(strict_load),
                "roles": list(role_names),
            },
            "models": list(model_names),
            "targets": list(targets),
            "seed": int(applied_seed) if applied_seed is not None else None,
            "num_role_rows": int(len(role_rows)),
            "num_gain_rows": int(len(gain_rows)),
        },
    )
    print(f"[INFO] Saved defense evaluation results: {out_dir}")


if __name__ == "__main__":
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_name=arg)
