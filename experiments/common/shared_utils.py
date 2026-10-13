from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


DEFAULT_CLEAN_WEIGHTS_DIR = "model_params_optuna_tiebreak_seed42"
DEFAULT_CLEAN_EXPORT_SEED = 42
# Optional pair of mirrored storage roots (e.g. workstation and cluster mounts of the
# same share). Absolute paths under one root are remapped to the other when needed.
WIFI_SENSING_ROOT_RESEARCH = Path(os.environ["CSI_STORAGE_ROOT"]) if os.environ.get("CSI_STORAGE_ROOT") else None
WIFI_SENSING_ROOT_CLUSTER = Path(os.environ["CSI_STORAGE_ROOT_ALT"]) if os.environ.get("CSI_STORAGE_ROOT_ALT") else None


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def remap_wifi_sensing_path(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute() or WIFI_SENSING_ROOT_RESEARCH is None or WIFI_SENSING_ROOT_CLUSTER is None:
        return path

    text = str(path)
    candidates = [path]
    if text == str(WIFI_SENSING_ROOT_RESEARCH) or text.startswith(str(WIFI_SENSING_ROOT_RESEARCH) + os.sep):
        suffix = text[len(str(WIFI_SENSING_ROOT_RESEARCH)) :]
        candidates.append(Path(str(WIFI_SENSING_ROOT_CLUSTER) + suffix))
    elif text == str(WIFI_SENSING_ROOT_CLUSTER) or text.startswith(str(WIFI_SENSING_ROOT_CLUSTER) + os.sep):
        suffix = text[len(str(WIFI_SENSING_ROOT_CLUSTER)) :]
        candidates.append(Path(str(WIFI_SENSING_ROOT_RESEARCH) + suffix))

    seen: set[Path] = set()
    ordered: List[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)

    for candidate in ordered:
        if candidate.exists():
            return candidate

    if WIFI_SENSING_ROOT_CLUSTER.exists():
        for candidate in ordered:
            if str(candidate).startswith(str(WIFI_SENSING_ROOT_CLUSTER)):
                return candidate
    if WIFI_SENSING_ROOT_RESEARCH.exists():
        for candidate in ordered:
            if str(candidate).startswith(str(WIFI_SENSING_ROOT_RESEARCH)):
                return candidate
    return ordered[0]


def parse_token_list(value: Optional[object]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        return [item for item in items if item]
    return []


def parse_splits(value: Optional[object], default: Sequence[str]) -> List[str]:
    parsed = parse_token_list(value)
    return parsed if parsed else list(default)


def parse_run_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    run_name = str(value).strip()
    return run_name or None


def resolve_ssl_run_selection(
    run_cfg: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[bool, bool, str]:
    only = run_cfg.get("only")
    if only is not None:
        key = str(only).strip().lower().replace("-", "_")
        if key in {"simclr", "simclr_only"}:
            return True, False, "simclr"
        if key in {"wimae", "mae", "wimae_only", "mae_only"}:
            return False, True, "wimae"
        if key in {"both", "all", "simclr_wimae", "simclr+wimae"}:
            return True, True, "simclr_wimae"
        raise ValueError("run.only must be one of: simclr, wimae (mae), both.")

    run_simclr = parse_bool(run_cfg.get("simclr", cfg.get("simclr", {}).get("enabled", True)), default=True)
    run_wimae = parse_bool(run_cfg.get("wimae", cfg.get("wimae", {}).get("enabled", False)), default=False)
    if run_simclr and run_wimae:
        return True, True, "simclr_wimae"
    if run_simclr:
        return True, False, "simclr"
    if run_wimae:
        return False, True, "wimae"
    return False, False, "none"


def normalize_csi_bench_task_name(value: str) -> str:
    key = value.strip().lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "har": "HumanActivityRecognition",
        "humanactivityrecognition": "HumanActivityRecognition",
        "human_activity_recognition": "HumanActivityRecognition",
        "human-activity-recognition": "HumanActivityRecognition",
        "hid": "HumanIdentification",
        "humanidentification": "HumanIdentification",
        "human_identification": "HumanIdentification",
        "human-identification": "HumanIdentification",
        "proximityrecognition": "ProximityRecognition",
        "proximity_recognition": "ProximityRecognition",
        "proximity-recognition": "ProximityRecognition",
        "falldetection": "FallDetection",
        "fall_detection": "FallDetection",
        "fall-detection": "FallDetection",
        "localization": "Localization",
        "localisation": "Localization",
        "breathingdetection": "BreathingDetection",
        "breathing_detection": "BreathingDetection",
        "breathing_detection_subset": "BreathingDetection",
        "breathingdetection_subset": "BreathingDetection",
        "motionsourcerecognition": "MotionSourceRecognition",
        "motion_source_recognition": "MotionSourceRecognition",
        "motion_source_recognition_updated": "MotionSourceRecognition",
        "motionsourcerecognitionupdated": "MotionSourceRecognition",
        "motion_source_recognition_fourclass": "MotionSourceRecognition",
        "motion_source_recognition_4class": "MotionSourceRecognition",
        "motionsourcerecognition_fourclass": "MotionSourceRecognition",
        "motionsourcerecognition_4class": "MotionSourceRecognition",
    }
    return mapping.get(key, value)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def prepare_run_dirs(out_dir: str, *, include_eval_outputs: bool = False) -> Dict[str, str]:
    """Create and return standard run directories used by training/poisoning scripts."""
    ensure_dir(out_dir)
    reports_dir = os.path.join(out_dir, "reports")
    figures_dir = os.path.join(out_dir, "figures")
    ensure_dir(reports_dir)
    ensure_dir(figures_dir)
    out: Dict[str, str] = {
        "out_dir": out_dir,
        "reports_dir": reports_dir,
        "figures_dir": figures_dir,
    }
    if include_eval_outputs:
        eval_outputs_dir = os.path.join(out_dir, "artifacts", "eval_outputs")
        ensure_dir(eval_outputs_dir)
        out["eval_outputs_dir"] = eval_outputs_dir
    return out


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def move_incomplete_run(out_dir: str) -> Optional[str]:
    if not os.path.exists(out_dir):
        return None
    parent_dir = os.path.dirname(out_dir)
    incomplete_dir = os.path.join(parent_dir, "incomplete")
    os.makedirs(incomplete_dir, exist_ok=True)
    target_dir = os.path.join(incomplete_dir, os.path.basename(out_dir))
    if os.path.exists(target_dir):
        return None
    shutil.move(out_dir, target_dir)
    return target_dir


def to_hw_pair(value: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return int(value), int(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Expected int or 2-element sequence, got {value}.")


def infer_input_shape_from_loader(loader) -> Optional[Tuple[int, int, int]]:
    for batch in loader:
        if not isinstance(batch, (list, tuple)) or len(batch) < 1:
            continue
        inputs = batch[0]
        if not torch.is_tensor(inputs):
            continue
        if inputs.ndim != 4 or inputs.size(0) == 0:
            continue
        return int(inputs.shape[1]), int(inputs.shape[2]), int(inputs.shape[3])
    return None


def history_rows_supervised(history) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for entry in history:
        rows.append(
            {
                "epoch": entry.epoch,
                "train_loss": entry.train_loss,
                "train_accuracy": entry.train_accuracy,
                "val_loss": entry.val_loss,
                "val_accuracy": entry.val_accuracy,
            }
        )
    return rows


def history_rows_simclr(history) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in history:
        rows.append(
            {
                "epoch": item.epoch,
                "lr": item.lr,
                "train_loss": item.train_loss,
                "train_acc_top1": item.train_acc_top1,
                "train_acc_top5": item.train_acc_top5,
                "train_acc_mean_pos": item.train_acc_mean_pos,
                "val_loss": item.val_loss,
                "val_acc_top1": item.val_acc_top1,
                "val_acc_top5": item.val_acc_top5,
                "val_acc_mean_pos": item.val_acc_mean_pos,
            }
        )
    return rows


def history_rows_wimae(history) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in history:
        rows.append(
            {
                "epoch": item.epoch,
                "lr": item.lr,
                "train_loss": item.train_loss,
                "val_loss": item.val_loss,
            }
        )
    return rows


def print_training_start(header: str, info: Mapping[str, Any]) -> None:
    print(f"\n[START] {header}")
    for key in sorted(info.keys()):
        print(f"[START] {key}={info[key]}")


def save_training_metadata(
    out_dir: str,
    payload: Mapping[str, Any],
    *,
    filename: str = "training_data_metadata.json",
) -> str:
    artifacts_dir = os.path.join(out_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    path = os.path.join(artifacts_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2)
    print(f"[INFO] Saved training metadata: {path}")
    return path


def _safe_split_count(path: str) -> Optional[int]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return len(payload)
    except Exception:
        return None
    return None


def _safe_csv_rows(path: str) -> Optional[int]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None)
            return sum(1 for _ in reader)
    except Exception:
        return None


def collect_csi_bench_task_metadata(
    *,
    dataset_root: str,
    task_name: str,
    splits_dir_name: str,
    split_names: Sequence[str],
    metadata_name: str = "sample_metadata.csv",
) -> Dict[str, Any]:
    task_dir = os.path.join(dataset_root, task_name)
    metadata_path = os.path.join(task_dir, "metadata", metadata_name)
    split_dir = os.path.join(task_dir, splits_dir_name)
    splits: Dict[str, Dict[str, Any]] = {}
    for split_name in split_names:
        split_path = os.path.join(split_dir, f"{split_name}.json")
        splits[split_name] = {
            "path": split_path,
            "exists": os.path.exists(split_path),
            "count": _safe_split_count(split_path),
        }
    return {
        "task_name": task_name,
        "task_dir": task_dir,
        "metadata_path": metadata_path,
        "metadata_exists": os.path.exists(metadata_path),
        "metadata_rows": _safe_csv_rows(metadata_path),
        "splits_dir": split_dir,
        "splits": splits,
    }


def _sanitize_file_token(value: Any, fallback: str = "na") -> str:
    token = "" if value is None else str(value).strip()
    if not token:
        return fallback
    out_chars: List[str] = []
    for ch in token:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out_chars.append(ch)
        else:
            out_chars.append("-")
    out = "".join(out_chars).strip("-_.")
    return out or fallback


def get_cluster_job_metadata() -> Dict[str, Optional[str]]:
    env = os.environ
    scheduler = "local"
    job_id = None
    task_id = None

    if env.get("SLURM_JOB_ID"):
        scheduler = "slurm"
        job_id = env.get("SLURM_JOB_ID")
        task_id = env.get("SLURM_ARRAY_TASK_ID")
    elif env.get("JOB_ID"):
        scheduler = "sge"
        job_id = env.get("JOB_ID")
        task_id = env.get("SGE_TASK_ID")
    elif env.get("PBS_JOBID"):
        scheduler = "pbs"
        job_id = env.get("PBS_JOBID")
        task_id = env.get("PBS_ARRAYID") or env.get("PBS_TASKNUM")
    elif env.get("LSB_JOBID"):
        scheduler = "lsf"
        job_id = env.get("LSB_JOBID")
        task_id = env.get("LSB_JOBINDEX")

    host = env.get("HOSTNAME") or env.get("COMPUTERNAME")
    return {
        "scheduler": scheduler,
        "job_id": job_id,
        "task_id": task_id,
        "host": host,
        "pid": str(os.getpid()),
    }


def _resolve_non_overwriting_path(base_path: str, strategy: str) -> Optional[str]:
    if not os.path.exists(base_path):
        return base_path
    normalized = str(strategy).strip().lower()
    if normalized == "replace":
        return base_path
    if normalized == "skip":
        return None

    stem, ext = os.path.splitext(base_path)
    counter = 1
    while True:
        candidate = f"{stem}__v{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def export_best_weight_to_datasets(
    *,
    source_checkpoint: str,
    datasets_root: str,
    domain: str,
    dataset_name: str,
    model_name: str,
    task_name: Optional[str] = None,
    run_output_dir: Optional[str] = None,
    run_timestamp: Optional[str] = None,
    config_name: Optional[str] = None,
    weights_dir_name: str = "model_params_optuna_best",
    overwrite_strategy: str = "unique",
    include_full_config: bool = True,
    config_payload: Optional[Mapping[str, Any]] = None,
    metadata_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    checkpoint_src = os.path.abspath(source_checkpoint)
    if not os.path.exists(checkpoint_src):
        raise FileNotFoundError(f"Missing checkpoint for export: {checkpoint_src}")

    datasets_root_abs = os.path.abspath(datasets_root)
    weights_dir_token = _sanitize_file_token(weights_dir_name, fallback="model_params_optuna_best")
    domain_key = str(domain).strip().lower().replace("-", "_")

    if domain_key == "csi_bench":
        task_norm = task_name or "unknown_task"
        export_dir = os.path.join(datasets_root_abs, "CSI-Bench", weights_dir_token, str(task_norm))
        dataset_part = "CSI-Bench"
        task_part = str(task_norm)
    else:
        export_dir = os.path.join(datasets_root_abs, str(dataset_name), weights_dir_token)
        dataset_part = str(dataset_name)
        task_part = None

    os.makedirs(export_dir, exist_ok=True)

    cluster_meta = get_cluster_job_metadata()
    run_tag = _sanitize_file_token(run_timestamp or _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    cfg_tag = _sanitize_file_token(config_name or "cfg")
    model_tag = _sanitize_file_token(model_name)
    dataset_tag = _sanitize_file_token(dataset_part)
    parts = [dataset_tag]
    if task_part:
        parts.append(_sanitize_file_token(task_part))
    parts.extend([model_tag, cfg_tag, run_tag])

    job_id = cluster_meta.get("job_id")
    task_id = cluster_meta.get("task_id")
    if job_id:
        parts.append(f"job{_sanitize_file_token(job_id)}")
    if task_id and str(task_id) != "undefined":
        parts.append(f"task{_sanitize_file_token(task_id)}")
    if not job_id:
        parts.append("local")

    base_name = "__".join(parts) + ".pt"
    requested_path = os.path.join(export_dir, base_name)
    target_path = _resolve_non_overwriting_path(requested_path, overwrite_strategy)
    if target_path is None:
        return {
            "exported": False,
            "reason": "skipped_existing",
            "source_checkpoint": checkpoint_src,
            "target_checkpoint": requested_path,
        }

    shutil.copy2(checkpoint_src, target_path)

    sidecar_payload: Dict[str, Any] = {
        "stage": "optuna_best_weight_export",
        "domain": domain_key,
        "dataset": dataset_part,
        "task_name": task_part,
        "model": model_name,
        "weights_dir_name": weights_dir_token,
        "source_checkpoint": checkpoint_src,
        "source_run_dir": os.path.abspath(run_output_dir) if run_output_dir else None,
        "target_checkpoint": target_path,
        "cluster": cluster_meta,
        "run_timestamp": run_timestamp,
        "config_name": config_name,
        "created_utc": _dt.datetime.utcnow().isoformat() + "Z",
    }
    if metadata_payload:
        sidecar_payload["run_metadata"] = dict(metadata_payload)
    if include_full_config and config_payload is not None:
        sidecar_payload["config"] = dict(config_payload)

    sidecar_path = f"{target_path}.meta.json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_payload, f, indent=2)

    return {
        "exported": True,
        "source_checkpoint": checkpoint_src,
        "target_checkpoint": target_path,
        "metadata_path": sidecar_path,
        "cluster": cluster_meta,
    }
