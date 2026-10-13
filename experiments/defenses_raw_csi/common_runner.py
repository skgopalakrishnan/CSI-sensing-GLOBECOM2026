"""Shared runner utilities for defense experiments.

This module centralises checkpoint resolution/loading, split-name mapping,
model-eval helpers, and export helpers used by adversarial training and
randomized smoothing scripts.
"""

from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from csi.models.supervised import evaluate_classifier
from csi.models.utils.experiment import sanitize_token

from ..common.train_models_utils import resolve_project_path


def call_with_supported_kwargs(func, *args, **kwargs):
    """Call a function while dropping kwargs not in its signature."""
    try:
        sig = inspect.signature(func)
    except Exception:
        return func(*args, **kwargs)

    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return func(*args, **kwargs)
    supported = {
        name
        for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return func(*args, **filtered)


def normalise_selection_mode(value: Any) -> str:
    key = str(value or "auto").strip().lower()
    if key not in {"auto", "alias", "latest"}:
        raise ValueError("weights.selection must be one of: auto, alias, latest")
    return key


def as_weight_dir_list(value: Any, *, default: str) -> List[str]:
    if value is None:
        return [default]
    if isinstance(value, str):
        out = [token.strip() for token in value.split(",") if token.strip()]
        return out if out else [default]
    if isinstance(value, (list, tuple)):
        out = [str(token).strip() for token in value if str(token).strip()]
        return out if out else [default]
    return [default]


def extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], Mapping):
            return payload["model_state_dict"]  # type: ignore[return-value]
        if "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
            return payload["state_dict"]  # type: ignore[return-value]
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload  # type: ignore[return-value]
    raise ValueError(f"Unsupported checkpoint format: {type(payload)!r}")


def load_checkpoint_into_model(
    *,
    model: nn.Module,
    checkpoint_path: str,
    strict: bool,
    device: torch.device,
    eval_mode: bool = False,
) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = extract_state_dict(payload)
    result = model.load_state_dict(state, strict=bool(strict))
    model.to(device)
    if eval_mode:
        model.eval()
    return {
        "checkpoint_path": checkpoint_path,
        "strict": bool(strict),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def nested_checkpoint_override(mapping: Any, key_1: str, key_2: str) -> Optional[str]:
    if not isinstance(mapping, Mapping):
        return None
    level_1 = mapping.get(key_1)
    if not isinstance(level_1, Mapping):
        return None
    value = level_1.get(key_2)
    return str(value) if value else None


def resolve_checkpoint(
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

    selection_key = normalise_selection_mode(selection)
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
    if not token_hits:
        available = sorted(path.name for path in candidates)
        preview = ", ".join(available[:8])
        if len(available) > 8:
            preview += f", ... (+{len(available) - 8} more)"
        raise FileNotFoundError(
            f"No checkpoint for model={model_name!r} found in {base_dir} "
            f"(pattern='{pattern}'). Available: [{preview}]"
        )
    return max(token_hits, key=lambda path: path.stat().st_mtime)


def resolve_csi_bench_checkpoint_dir(
    *,
    weights_dataset_root: str,
    weights_dir_name: str,
    task_name: str,
    model_name: Optional[str] = None,
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
        if not candidate.exists():
            continue
        if model_name:
            alias = candidate / f"CSI-Bench_{model_name}.pt"
            if alias.exists():
                return candidate
            model_token = sanitize_token(model_name).lower()
            try:
                for path in candidate.glob("*.pt"):
                    if model_token in sanitize_token(path.stem).lower():
                        return candidate
            except Exception:
                pass
        else:
            return candidate

    for candidate in ordered:
        if candidate.exists():
            return candidate
    return ordered[0]


def normalise_csi_split_name(split_name: str) -> str:
    key = str(split_name).strip().lower().replace("-", "_")
    if key in {"train", "train_id"}:
        return "train_id"
    if key in {"val", "validation", "val_id"}:
        return "val_id"
    if key in {"test", "test_id"}:
        return "test_id"
    return key


def loader_key_for_csi_split(split_name: str) -> str:
    split_key = normalise_csi_split_name(split_name)
    if split_key == "train_id":
        return "train"
    if split_key == "val_id":
        return "val"
    if split_key == "test_id":
        return "test"
    if split_key.startswith("test_"):
        return split_key
    return f"test_{split_key}"


def default_ntu_datasets(task_family: str) -> List[str]:
    if str(task_family).strip().lower() == "hid":
        return ["NTU-Fi-HumanID"]
    return ["NTU-Fi_HAR"]


def default_csi_tasks(task_family: str) -> List[str]:
    if str(task_family).strip().lower() == "hid":
        return ["HumanIdentification"]
    return ["HumanActivityRecognition"]


def clean_metrics(*, model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    metrics, _ = evaluate_classifier(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        use_tqdm=False,
        collect_predictions=False,
        collect_probabilities=False,
        desc="defense_clean_eval",
    )
    return {
        "clean_acc": float(metrics.get("accuracy", 0.0)),
        "clean_loss": float(metrics.get("loss", 0.0)),
        "clean_f1_macro": float(metrics.get("f1_macro", 0.0)),
        "clean_f1_weighted": float(metrics.get("f1_weighted", 0.0)),
    }


def resolve_checkpoint_epochs(training_cfg: Mapping[str, Any], epochs: int) -> List[int]:
    value = training_cfg.get("checkpoint_epochs")
    resolved: List[int] = []
    if value is None:
        resolved = [epochs]
        if epochs >= 10:
            resolved.append(10)
    else:
        if isinstance(value, (list, tuple)):
            for item in value:
                try:
                    epoch = int(item)
                except Exception:
                    continue
                if 1 <= epoch <= epochs:
                    resolved.append(epoch)
        else:
            try:
                epoch = int(value)
            except Exception:
                epoch = epochs
            if 1 <= epoch <= epochs:
                resolved.append(epoch)
    if epochs not in resolved:
        resolved.append(epochs)
    return sorted(set(resolved))


def safe_token(value: Any, fallback: str = "na") -> str:
    token = sanitize_token(value)
    return token if token and token != "na" else fallback


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2)


def build_checkpoint_payload(model: nn.Module) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model_state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    }
    params = getattr(model, "params", None)
    if isinstance(params, Mapping):
        payload["params"] = dict(params)
    return payload


def export_ntu_checkpoint(
    *,
    datasets_root: str,
    export_weights_dir_name: str,
    dataset_name: str,
    model_name: str,
    source_checkpoint: str,
    run_timestamp: str,
    run_dir: str,
    config_path: Optional[str],
    stage: str,
    latest_alias: bool = True,
    checkpoint_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    root = resolve_project_path(datasets_root)
    export_dir = root / dataset_name / export_weights_dir_name
    export_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{dataset_name}_{model_name}"
    ts_token = safe_token(run_timestamp, fallback="run")
    timestamped = export_dir / f"{stem}__{ts_token}.pt"
    latest = export_dir / f"{stem}.pt"

    if checkpoint_payload is None:
        shutil.copy2(source_checkpoint, timestamped)
        if latest_alias:
            shutil.copy2(source_checkpoint, latest)
    else:
        torch.save(dict(checkpoint_payload), timestamped)
        if latest_alias:
            torch.save(dict(checkpoint_payload), latest)

    meta = {
        "stage": stage,
        "domain": "ntu",
        "dataset": dataset_name,
        "model": model_name,
        "run_timestamp": run_timestamp,
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "source_run_dir": str(Path(run_dir).resolve()),
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "latest_alias_updated": bool(latest_alias),
        "latest_path": str(latest),
    }
    write_json(Path(f"{timestamped}.meta.json"), meta)
    if latest_alias:
        write_json(Path(f"{latest}.meta.json"), {**meta, "variant": "latest_alias"})
    return {"timestamped_path": str(timestamped), "latest_path": str(latest)}


def export_csi_checkpoint(
    *,
    dataset_root: str,
    export_weights_dir_name: str,
    task_name: str,
    model_name: str,
    source_checkpoint: str,
    run_timestamp: str,
    run_dir: str,
    config_path: Optional[str],
    stage: str,
    latest_alias: bool = True,
    checkpoint_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    root = resolve_project_path(dataset_root)
    csi_root = root if root.name == "CSI-Bench" else (root / "CSI-Bench")
    export_dir = csi_root / export_weights_dir_name / task_name
    export_dir.mkdir(parents=True, exist_ok=True)

    stem = f"CSI-Bench_{model_name}"
    ts_token = safe_token(run_timestamp, fallback="run")
    timestamped = export_dir / f"{stem}__{ts_token}.pt"
    latest = export_dir / f"{stem}.pt"

    if checkpoint_payload is None:
        shutil.copy2(source_checkpoint, timestamped)
        if latest_alias:
            shutil.copy2(source_checkpoint, latest)
    else:
        torch.save(dict(checkpoint_payload), timestamped)
        if latest_alias:
            torch.save(dict(checkpoint_payload), latest)

    meta = {
        "stage": stage,
        "domain": "csi_bench",
        "dataset": "CSI-Bench",
        "task_name": task_name,
        "model": model_name,
        "run_timestamp": run_timestamp,
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "source_run_dir": str(Path(run_dir).resolve()),
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "latest_alias_updated": bool(latest_alias),
        "latest_path": str(latest),
    }
    write_json(Path(f"{timestamped}.meta.json"), meta)
    if latest_alias:
        write_json(Path(f"{latest}.meta.json"), {**meta, "variant": "latest_alias"})
    return {"timestamped_path": str(timestamped), "latest_path": str(latest)}
