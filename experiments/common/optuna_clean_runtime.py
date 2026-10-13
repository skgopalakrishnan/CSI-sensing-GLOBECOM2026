from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import yaml

from csi.models.model_factory import build_model
from csi.models.utils.datasets import DatasetSpec
from csi.models.utils.experiment import sanitize_token
from csi.models.utils.optimisation import build_model_from_hparams

from ..common.shared_utils import (
    DEFAULT_CLEAN_EXPORT_SEED,
    DEFAULT_CLEAN_WEIGHTS_DIR,
    normalize_csi_bench_task_name,
    remap_wifi_sensing_path,
)
from ..train_models.optuna_seed_retrain import _coerce_hparams


DEFAULT_NTU_TIEBREAK_SUMMARY = "experiments/metadata/optuna_selection/ntu_optuna_tiebreak_summary.json"
DEFAULT_CSI_BENCH_TIEBREAK_SUMMARY = "experiments/metadata/optuna_selection/csi_bench_optuna_tiebreak_summary.json"
DEFAULT_NTU_BOOTSTRAP_CONFIG = "experiments/config/training/ntu/sl/seed_retrain_from_optuna.yaml"
DEFAULT_CSI_BENCH_BOOTSTRAP_CONFIG = "experiments/config/training/csi_bench/sl/seed_retrain_from_optuna.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def is_optuna_clean_weights_dir(weights_dir_name: str) -> bool:
    return sanitize_token(weights_dir_name) == sanitize_token(DEFAULT_CLEAN_WEIGHTS_DIR)


def _domain_weights_cfg(weights_cfg: Mapping[str, Any], domain: str) -> Mapping[str, Any]:
    value = weights_cfg.get(domain)
    return value if isinstance(value, Mapping) else {}


def resolve_optuna_summary_json(weights_cfg: Mapping[str, Any], domain: str) -> Path:
    domain_cfg = _domain_weights_cfg(weights_cfg, domain)
    env_key = "CSI_OPTUNA_SUMMARY_JSON_NTU" if domain == "ntu" else "CSI_OPTUNA_SUMMARY_JSON_CSI_BENCH"
    candidate = (
        domain_cfg.get("selection_summary_json")
        or weights_cfg.get("selection_summary_json")
        or os.environ.get(env_key)
        or (DEFAULT_NTU_TIEBREAK_SUMMARY if domain == "ntu" else DEFAULT_CSI_BENCH_TIEBREAK_SUMMARY)
    )
    return Path(str(candidate)).resolve()


def resolve_fallback_weights_root(weights_cfg: Mapping[str, Any]) -> Path:
    candidate = (
        weights_cfg.get("fallback_weights_root")
        or os.environ.get("CSI_FALLBACK_WEIGHTS_ROOT")
        or "/tmp/csi_optuna_clean_weights/datasets"
    )
    path = remap_wifi_sensing_path(candidate)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    except Exception:
        fallback = Path("/tmp/csi_optuna_clean_weights/datasets")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback.resolve()


def load_optuna_target(
    *,
    domain: str,
    model_name: str,
    weights_cfg: Mapping[str, Any],
    dataset_name: Optional[str] = None,
    task_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    summary_path = resolve_optuna_summary_json(weights_cfg, domain)
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    studies = payload.get("studies")
    if not isinstance(studies, list):
        return None
    task_key = normalize_csi_bench_task_name(task_name) if task_name else None
    for study in studies:
        if not isinstance(study, Mapping):
            continue
        if str(study.get("model")) != str(model_name):
            continue
        if domain == "ntu" and str(study.get("dataset")) != str(dataset_name):
            continue
        if domain == "csi_bench" and normalize_csi_bench_task_name(str(study.get("task_name"))) != task_key:
            continue
        selected = study.get("selected_trial")
        if not isinstance(selected, Mapping):
            continue
        params = selected.get("params")
        if not isinstance(params, Mapping):
            continue
        return {
            "study_dir": str(study.get("study_dir", "")),
            "dataset": (str(study.get("dataset")) if study.get("dataset") is not None else None),
            "task_name": (str(study.get("task_name")) if study.get("task_name") is not None else None),
            "model": str(model_name),
            "objective_metric": str(study.get("objective_metric", "val_accuracy")).strip().lower(),
            "direction": str(study.get("direction", "maximize")).strip().lower(),
            "selected_trial_number": selected.get("number"),
            "selected_trial_value": selected.get("value"),
            "params": _coerce_hparams(str(model_name), params),
            "summary_json": str(summary_path),
        }
    return None


def _checkpoint_params_to_arch_metadata(params: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = dict(params)

    if params.get("in_channels") is not None:
        metadata.setdefault("channels", params.get("in_channels"))
    if params.get("feat_size") is not None:
        metadata.setdefault("freq_bins", params.get("feat_size"))
    if params.get("freq_bins") is not None:
        metadata.setdefault("freq_bins", params.get("freq_bins"))
    if params.get("seq_len") is not None:
        metadata.setdefault("time_steps", params.get("seq_len"))
    if params.get("time_steps") is not None:
        metadata.setdefault("time_steps", params.get("time_steps"))
    if params.get("num_classes") is not None:
        metadata.setdefault("num_classes", params.get("num_classes"))

    channels = metadata.get("channels")
    freq_bins = metadata.get("freq_bins")
    time_steps = metadata.get("time_steps")
    if channels is not None and freq_bins is not None and time_steps is not None:
        try:
            metadata["input_shape"] = [int(channels), int(freq_bins), int(time_steps)]
        except Exception:
            pass
    return metadata


def _load_checkpoint_params(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    params = payload.get("params")
    if not isinstance(params, Mapping):
        return None
    return _checkpoint_params_to_arch_metadata(params)


def _source_weights_dir_candidates(weights_cfg: Mapping[str, Any], domain: str) -> Sequence[str]:
    candidates: list[str] = []
    domain_cfg = _domain_weights_cfg(weights_cfg, domain)

    def _extend(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            token = value.strip()
            if token:
                candidates.append(token)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                token = str(item).strip()
                if token:
                    candidates.append(token)

    for source in (weights_cfg, domain_cfg):
        if not isinstance(source, Mapping):
            continue
        _extend(source.get("source_weights_dir_name"))
        _extend(source.get("source_weights_dir_names"))

    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        key = sanitize_token(candidate)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    return ordered


def _load_weights_cfg_from_export_meta(checkpoint_path: str) -> Mapping[str, Any]:
    meta_path = Path(f"{checkpoint_path}.meta.json")
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    candidate_paths: list[Path] = []

    def _append_candidate(path_value: Any) -> None:
        if not path_value:
            return
        raw_path = Path(str(path_value))
        candidate_paths.append(raw_path)
        remapped = remap_wifi_sensing_path(str(path_value))
        if remapped != raw_path:
            candidate_paths.append(remapped)

    config_path = meta.get("config_path")
    if config_path:
        _append_candidate(config_path)
    source_run_dir = meta.get("source_run_dir")
    if source_run_dir:
        run_dir_candidates: list[Path] = []
        _append_candidate(source_run_dir)
        raw_run_dir = Path(str(source_run_dir))
        remapped_run_dir = remap_wifi_sensing_path(str(source_run_dir))
        run_dir_candidates.extend([raw_run_dir, remapped_run_dir])

        for run_dir in list(run_dir_candidates):
            run_dir_str = str(run_dir)
            if "/results/jobs/" not in run_dir_str:
                continue
            mirror = Path(run_dir_str.replace("/results/jobs/", "/campaign_results/"))
            run_dir_candidates.append(mirror)
            suffix = run_dir_str.split("/results/jobs/", 1)[1]
            run_dir_candidates.append(PROJECT_ROOT / "campaign_results" / suffix)

        for run_dir in run_dir_candidates:
            candidate_paths.append(run_dir / "config_resolved.json")

    seen: set[Path] = set()
    for candidate in candidate_paths:
        candidate = candidate.resolve() if candidate.exists() else candidate
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists():
            continue
        try:
            payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        weights_cfg = payload.get("weights")
        if isinstance(weights_cfg, Mapping):
            return weights_cfg
    return {}


def _should_use_optuna_target(
    *,
    domain: str,
    weights_dir_name: str,
    weights_cfg: Mapping[str, Any],
    checkpoint_path: Optional[str],
) -> bool:
    if is_optuna_clean_weights_dir(weights_dir_name):
        return True

    for candidate in _source_weights_dir_candidates(weights_cfg, domain):
        if is_optuna_clean_weights_dir(candidate):
            return True

    if checkpoint_path:
        export_weights_cfg = _load_weights_cfg_from_export_meta(checkpoint_path)
        for candidate in _source_weights_dir_candidates(export_weights_cfg, domain):
            if is_optuna_clean_weights_dir(candidate):
                return True
    return False


def build_model_for_checkpoint(
    *,
    domain: str,
    model_name: str,
    weights_dir_name: str,
    weights_cfg: Mapping[str, Any],
    spec: DatasetSpec,
    dataset_key_for_model: str,
    num_classes: int,
    arch_metadata: Mapping[str, Any],
    dataset_name: Optional[str] = None,
    task_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
):
    merged_arch_metadata = dict(arch_metadata)
    checkpoint_metadata = _load_checkpoint_params(checkpoint_path) if checkpoint_path else None
    if checkpoint_metadata is not None:
        merged_arch_metadata.update(checkpoint_metadata)
        model, _ = build_model(
            dataset_key_for_model,
            model_name,
            num_classes=int(num_classes),
            arch_metadata=merged_arch_metadata,
        )
        return model, {"source": "checkpoint_params", "params": checkpoint_metadata}

    target = None
    if _should_use_optuna_target(
        domain=domain,
        weights_dir_name=weights_dir_name,
        weights_cfg=weights_cfg,
        checkpoint_path=checkpoint_path,
    ):
        target = load_optuna_target(
            domain=domain,
            model_name=model_name,
            weights_cfg=weights_cfg,
            dataset_name=dataset_name,
            task_name=task_name,
        )
    if target is not None:
        model = build_model_from_hparams(spec, model_name, int(num_classes), dict(target["params"]))
        return model, target
    model, _ = build_model(
        dataset_key_for_model,
        model_name,
        num_classes=int(num_classes),
        arch_metadata=merged_arch_metadata,
    )
    return model, None


def _alias_name(domain: str, *, dataset_name: Optional[str], task_name: Optional[str], model_name: str) -> str:
    if domain == "ntu":
        return f"{dataset_name}_{model_name}.pt"
    _ = task_name
    return f"CSI-Bench_{model_name}.pt"


def _fallback_checkpoint_path(
    *,
    domain: str,
    fallback_root: Path,
    dataset_name: Optional[str],
    task_name: Optional[str],
    model_name: str,
) -> Path:
    alias = _alias_name(domain, dataset_name=dataset_name, task_name=task_name, model_name=model_name)
    if domain == "ntu":
        assert dataset_name is not None
        return fallback_root / dataset_name / DEFAULT_CLEAN_WEIGHTS_DIR / alias
    assert task_name is not None
    return fallback_root / "CSI-Bench" / DEFAULT_CLEAN_WEIGHTS_DIR / normalize_csi_bench_task_name(task_name) / alias


def ensure_optuna_clean_checkpoint(
    *,
    domain: str,
    model_name: str,
    weights_dir_name: str,
    weights_cfg: Mapping[str, Any],
    canonical_base_dir: Path,
    alias_stems: Sequence[str],
    dataset_name: Optional[str] = None,
    task_name: Optional[str] = None,
) -> Optional[str]:
    if not is_optuna_clean_weights_dir(weights_dir_name):
        return None
    for stem in alias_stems:
        alias_path = canonical_base_dir / f"{stem}.pt"
        if alias_path.exists():
            return None

    fallback_root = resolve_fallback_weights_root(weights_cfg)
    target_path = _fallback_checkpoint_path(
        domain=domain,
        fallback_root=fallback_root,
        dataset_name=dataset_name,
        task_name=task_name,
        model_name=model_name,
    )
    if target_path.exists():
        return str(target_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target_path.with_suffix(target_path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if target_path.exists():
            return str(target_path)

        summary_path = resolve_optuna_summary_json(weights_cfg, domain)
        if not summary_path.exists():
            raise FileNotFoundError(f"Optuna tiebreak summary not found for fallback bootstrap: {summary_path}")
        _bootstrap_optuna_clean_checkpoint(
            domain=domain,
            summary_path=summary_path,
            fallback_root=fallback_root,
            model_name=model_name,
            dataset_name=dataset_name,
            task_name=task_name,
        )
        if not target_path.exists():
            raise FileNotFoundError(f"Fallback checkpoint bootstrap did not produce expected alias: {target_path}")
        return str(target_path)


def _bootstrap_optuna_clean_checkpoint(
    *,
    domain: str,
    summary_path: Path,
    fallback_root: Path,
    model_name: str,
    dataset_name: Optional[str],
    task_name: Optional[str],
) -> None:
    base_config = Path(DEFAULT_NTU_BOOTSTRAP_CONFIG if domain == "ntu" else DEFAULT_CSI_BENCH_BOOTSTRAP_CONFIG).resolve()
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}

    selection_cfg = cfg.setdefault("selection", {})
    selection_cfg["summary_json"] = str(summary_path)

    run_cfg = cfg.setdefault("run", {})
    run_cfg["seeds"] = [DEFAULT_CLEAN_EXPORT_SEED]
    run_cfg["deterministic"] = False
    run_cfg["require_test_split"] = True
    run_cfg["continue_on_error"] = False
    run_cfg["include_models"] = [model_name]
    if domain == "ntu":
        if not dataset_name:
            raise ValueError("dataset_name is required for NTU bootstrap.")
        run_cfg["include_datasets"] = [dataset_name]
        run_cfg.pop("include_tasks", None)
    else:
        if not task_name:
            raise ValueError("task_name is required for CSI-Bench bootstrap.")
        run_cfg["include_tasks"] = [normalize_csi_bench_task_name(task_name)]
        run_cfg.pop("include_datasets", None)

    export_cfg = cfg.setdefault("export", {})
    export_cfg["enabled"] = True
    export_cfg["clean_weights_dir_name"] = DEFAULT_CLEAN_WEIGHTS_DIR
    export_cfg["canonical_seed"] = DEFAULT_CLEAN_EXPORT_SEED
    export_cfg["latest_alias"] = True
    if domain == "ntu":
        export_cfg["datasets_root"] = str(fallback_root)
    else:
        export_cfg["dataset_root"] = str((fallback_root / "CSI-Bench").resolve())

    common_cfg = cfg.setdefault("common", {})
    common_cfg["experiment_name"] = f"{common_cfg.get('experiment_name', 'optuna-seed-retrain')}-bootstrap"

    # Bootstrap retrains are launched from within active attack/eval jobs on the same GPU slot.
    # Keep their dataloaders conservative to avoid pin-memory/device contention.
    data_cfg = cfg.setdefault("data", {})
    data_cfg["num_workers"] = 0
    data_cfg["pin_memory"] = False
    data_cfg["persistent_workers"] = False
    data_cfg["prefetch_factor"] = None

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix=f"optuna_clean_bootstrap_{domain}_", delete=False, encoding="utf-8") as handle:
        tmp_cfg = Path(handle.name)
        yaml.safe_dump(cfg, handle, sort_keys=False)

    env = os.environ.copy()
    env.setdefault("CSI_JOB_NAME", f"optuna-clean-bootstrap-{domain}")
    env.setdefault("CSI_JOB_TS", "bootstrap")
    env.setdefault("CSI_RESULTS_JOB_DIR", f"results/jobs/optuna_clean_bootstrap_{domain}")
    try:
        subprocess.run(
            [sys.executable, "-u", "-m", "experiments.train_models.optuna_seed_retrain", str(tmp_cfg)],
            check=True,
            env=env,
        )
    finally:
        try:
            tmp_cfg.unlink()
        except Exception:
            pass
