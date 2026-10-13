from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

JOB_DIR_ENV = "CSI_RESULTS_JOB_DIR"
JOB_NAME_ENV = "CSI_RESULTS_JOB_NAME"
JOB_TS_ENV = "CSI_RESULTS_JOB_TS"
JOB_RUN_PARTS_ENV = "CSI_RESULTS_RUN_PARTS"


def _experiments_root() -> str:
    repo_root = Path(__file__).resolve().parents[5]
    return str(repo_root / "experiments")


def _results_root() -> str:
    return os.path.join(_experiments_root(), "results")


def _sanitize_path_component(part: object) -> str:
    raw = str(part).strip()
    sanitised = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in raw)
    return sanitised or "unnamed"


def _sanitize_components(parts: Iterable[object]) -> List[str]:
    return [_sanitize_path_component(part) for part in parts if part is not None and str(part).strip() != ""]


def _job_run_parts_from_env() -> List[str]:
    raw = os.environ.get(JOB_RUN_PARTS_ENV, "")
    if not raw:
        return []
    normalised = raw.replace("\\", "/")
    return _sanitize_components(part for part in normalised.split("/") if part.strip())


def derive_config_parts(cfg_path: Optional[str]) -> List[str]:
    if not cfg_path:
        return []
    cfg_abs = os.path.abspath(cfg_path)
    cfg_root = os.path.join(_experiments_root(), "config")
    try:
        rel_cfg = os.path.relpath(cfg_abs, cfg_root)
    except Exception:
        rel_cfg = os.path.basename(cfg_abs)
    rel_no_ext = os.path.splitext(rel_cfg)[0]
    return [part for part in rel_no_ext.split(os.sep) if part and part not in (".", "..")]


try:
    from csi.attacks.utils.attack_utils import _snr_db_to_eps2  # type: ignore
except Exception:  # pragma: no cover
    _snr_db_to_eps2 = None

WB_ATTACK_FIELD_ORDER: Tuple[str, ...] = (
    "dataset",
    "model",
    "attack_method",
    "attack_mode",
    "targeted",
    "budget_label",
    "eps_2",
    "steps",
    "alpha",
    "alpha_fraction",
    "n_restarts",
    "random_start",
    "clean_acc",
    "clean_loss",
    "asr",
    "n_all",
    "n_clean_correct",
    "device",
    "weights_path",
    "seed",
)

ASR_CI_FIELD_ORDER: Tuple[str, ...] = (
    "dataset",
    "model",
    "attack_method",
    "attack_mode",
    "budget_label",
    "config_name",
    "n",
    "asr_mean",
    "asr_ci95",
)


def prepare_results_dir(
    experiment_name: str,
    cfg_path: Optional[str],
    *,
    mode: Optional[str] = None,
    job_name: Optional[str] = None,
    job_timestamp: Optional[str] = None,
    job_dir: Optional[str] = None,
    relative_parts: Optional[Sequence[object]] = None,
    include_timestamp: Optional[bool] = None,
    include_experiment: Optional[bool] = None,
) -> Tuple[str, str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_name = os.path.basename(cfg_path) if cfg_path else "inline"

    job_dir_effective: Optional[str] = None
    job_name_effective: Optional[str] = None
    job_ts_effective: Optional[str] = None

    if job_dir:
        job_dir_effective = os.path.abspath(job_dir)
        os.makedirs(job_dir_effective, exist_ok=True)
        job_name_effective = job_name or os.environ.get(JOB_NAME_ENV) or os.path.basename(job_dir_effective)
        job_ts_effective = job_timestamp or os.environ.get(JOB_TS_ENV) or ts
        os.environ[JOB_DIR_ENV] = job_dir_effective
        os.environ[JOB_NAME_ENV] = _sanitize_path_component(job_name_effective)
        os.environ[JOB_TS_ENV] = job_ts_effective
    else:
        env_dir = os.environ.get(JOB_DIR_ENV)
        env_name = os.environ.get(JOB_NAME_ENV)
        env_ts = os.environ.get(JOB_TS_ENV)
        if env_dir:
            job_dir_effective = env_dir
            job_name_effective = env_name
            job_ts_effective = env_ts or ts
        else:
            job_name_candidate = job_name or os.environ.get("CSI_JOB_NAME") or os.environ.get("SLURM_JOB_NAME")
            job_ts_candidate = job_timestamp or os.environ.get("CSI_JOB_TS")
            if job_name_candidate:
                job_dir_effective, job_name_effective, job_ts_effective = init_job_results_dir(
                    job_name_candidate,
                    timestamp=job_ts_candidate,
                )

    if job_dir_effective:
        base = job_dir_effective
        include_ts_default = False
    else:
        base = os.path.join(_results_root(), _sanitize_path_component(experiment_name))
        if mode:
            base = os.path.join(base, _sanitize_path_component(mode))
        include_ts_default = True

    include_ts_flag = include_timestamp if include_timestamp is not None else include_ts_default
    include_experiment_flag = include_experiment if include_experiment is not None else bool(job_dir_effective)

    path_components: List[str] = []
    if job_dir_effective:
        if include_experiment_flag:
            path_components.append(_sanitize_path_component(experiment_name))
        if mode:
            path_components.append(_sanitize_path_component(mode))

    if relative_parts is not None:
        path_components.extend(_sanitize_components(relative_parts))
    else:
        cfg_parts = derive_config_parts(cfg_path) if cfg_path else []
        sanitized_cfg_parts = _sanitize_components(cfg_parts)
        if sanitized_cfg_parts:
            path_components.extend(sanitized_cfg_parts)
        else:
            cfg_stub = os.path.splitext(cfg_name)[0] if cfg_name else "inline"
            path_components.append(_sanitize_path_component(cfg_stub))

    if include_ts_flag:
        path_components.append(ts)

    if job_dir_effective:
        path_components.extend(_job_run_parts_from_env())

    out_dir = os.path.join(base, *path_components) if path_components else base
    os.makedirs(out_dir, exist_ok=True)

    if job_dir_effective:
        if job_name_effective:
            os.environ[JOB_NAME_ENV] = _sanitize_path_component(job_name_effective)
        if job_ts_effective:
            os.environ[JOB_TS_ENV] = job_ts_effective

    return out_dir, ts, cfg_name


def init_job_results_dir(
    job_name: str,
    *,
    timestamp: Optional[str] = None,
    root: Optional[str] = None,
) -> Tuple[str, str, str]:
    sanitised_name = _sanitize_path_component(job_name)
    job_ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_root = os.path.abspath(root) if root else _results_root()
    os.makedirs(base_root, exist_ok=True)
    job_dir_name = f"{sanitised_name}_{job_ts}"
    job_dir = os.path.join(base_root, job_dir_name)
    os.makedirs(job_dir, exist_ok=True)
    os.environ[JOB_DIR_ENV] = job_dir
    os.environ[JOB_NAME_ENV] = sanitised_name
    os.environ[JOB_TS_ENV] = job_ts
    return job_dir, sanitised_name, job_ts


def save_config(out_dir: str, cfg: dict, filename: str = "config_resolved.json") -> str:
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def save_run_meta(out_dir: str, run_meta: dict, filename: str = "run_meta.json") -> str:
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(run_meta, f, indent=2)
    return path


def save_results_rows(
    out_dir: str,
    rows: List[Dict[str, object]],
    base_name: str = "summary_rows",
    *,
    field_order: Optional[Sequence[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    if not rows:
        return None, None
    all_keys = {k for r in rows for k in r.keys() if k is not None}
    if field_order:
        fieldnames = list(field_order)
        extras = [k for k in sorted(all_keys) if k not in fieldnames]
        fieldnames.extend(extras)
    else:
        fieldnames = sorted(all_keys)
    normalised_rows: List[Dict[str, object]] = [{field: row.get(field) for field in fieldnames} for row in rows]
    import csv as _csv

    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalised_rows)
    json_path = os.path.join(out_dir, f"{base_name}.json")
    with open(json_path, "w") as f:
        json.dump(normalised_rows, f, indent=2)
    return csv_path, json_path


def compute_ci(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    import numpy as np
    from scipy import stats

    if not data:
        return float("nan"), float("nan")
    arr = np.array(data)
    n = arr.size
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0
    sem = float(stats.sem(arr))
    h = float(sem * stats.t.ppf((1 + confidence) / 2.0, n - 1))
    return mean, h


def standardize_wb_attack_row(row: Dict[str, object]) -> Dict[str, object]:
    """Project attack/robustness rows onto the minimal schema."""

    normalised: Dict[str, object] = {}
    for field in WB_ATTACK_FIELD_ORDER:
        value = row.get(field)
        if value is None:
            continue
        if field in ("targeted", "random_start"):
            normalised[field] = bool(value)
            continue
        normalised[field] = value
    return normalised


def compute_asr_ci_summary(
    rows: List[Dict[str, object]],
    *,
    config_name: Optional[str] = None,
) -> List[Dict[str, object]]:
    from collections import defaultdict

    grouped: Dict[Tuple[object, ...], List[float]] = defaultdict(list)
    for row in rows:
        asr_val = row.get("asr")
        if asr_val is None:
            continue
        try:
            asr_float = float(asr_val)
        except (TypeError, ValueError):
            continue
        key = (
            row.get("dataset"),
            row.get("model"),
            row.get("attack_method"),
            row.get("attack_mode"),
            row.get("budget_label"),
        )
        grouped[key].append(asr_float)

    summary_rows: List[Dict[str, object]] = []
    for key, values in grouped.items():
        if not values:
            continue
        mean, half_width = compute_ci(values, confidence=0.95)
        dataset, model, attack_method, attack_mode, budget_label = key
        summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "attack_method": attack_method,
                "attack_mode": attack_mode,
                "budget_label": budget_label,
                "config_name": config_name,
                "n": int(len(values)),
                "asr_mean": float(mean),
                "asr_ci95": float(half_width),
            }
        )
    return summary_rows


__all__ = [
    "ASR_CI_FIELD_ORDER",
    "JOB_DIR_ENV",
    "JOB_NAME_ENV",
    "JOB_TS_ENV",
    "WB_ATTACK_FIELD_ORDER",
    "compute_asr_ci_summary",
    "compute_ci",
    "derive_config_parts",
    "init_job_results_dir",
    "prepare_results_dir",
    "save_config",
    "save_results_rows",
    "save_run_meta",
]
