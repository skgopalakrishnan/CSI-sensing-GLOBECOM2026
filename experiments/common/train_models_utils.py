from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from csi.models.autoencoders import HEAD_ALIASES
from csi.models.supervised import evaluate_classifier
from csi.models.supervised.models import MODEL_BUILDERS
from csi.models.utils.datasets import DatasetSpec
from csi.models.utils.model_utils import infer_num_classes

from .shared_utils import infer_input_shape_from_loader, parse_bool, remap_wifi_sensing_path


DEFAULT_CSI_BENCH_SUBCARRIERS = 56
DEFAULT_CSI_BENCH_TIMESTEPS = 250


def as_list(value: Any, default: Sequence[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        out = [token.strip() for token in value.split(",") if token.strip()]
        return out if out else list(default)
    if isinstance(value, (list, tuple)):
        out = [str(token).strip() for token in value if str(token).strip()]
        return out if out else list(default)
    return list(default)


def state_name(state: Any) -> str:
    name = getattr(state, "name", None)
    if isinstance(name, str):
        return name
    return str(state)


def json_dump(path: str, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return remap_wifi_sensing_path(path)
    return (repo_root() / path).resolve()


def write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2)


def configure_torch_performance(cfg: Mapping[str, Any]) -> None:
    perf_cfg = dict(cfg.get("performance", {}) or {})
    data_cfg = dict(cfg.get("data", {}) or {})

    tf32_enabled = parse_bool(
        perf_cfg.get("tf32", data_cfg.get("tf32")),
        default=True,
    )
    if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32_enabled)
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = bool(tf32_enabled)
        except Exception:
            pass
        cudnn_benchmark = parse_bool(
            perf_cfg.get("cudnn_benchmark", data_cfg.get("cudnn_benchmark")),
            default=True,
        )
        try:
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        except Exception:
            pass

    precision = perf_cfg.get("matmul_precision")
    if precision is None:
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        precision_key = str(precision).strip().lower()
        if precision_key in {"highest", "high", "medium"}:
            try:
                torch.set_float32_matmul_precision(precision_key)
            except Exception:
                pass


def normalise_model_name(name: str) -> str:
    key = str(name).lower().replace(" ", "").replace("-", "").replace("_", "")
    aliases = {
        "timesformer": "csitimesformer",
        "timesformer1d": "csitimesformer",
    }
    return aliases.get(key, key)


def resolve_supervised_models(requested: Sequence[str]) -> List[str]:
    available = set(MODEL_BUILDERS.keys())
    available.update(normalise_model_name(name) for name in HEAD_ALIASES.keys())
    if not requested:
        requested = ["ResNet18"]
    resolved: List[str] = []
    seen_keys = set()
    invalid: List[str] = []
    for raw_name in requested:
        key = normalise_model_name(raw_name)
        if key not in available:
            invalid.append(str(raw_name))
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        resolved.append(str(raw_name))
    if invalid:
        raise ValueError(
            f"Unsupported model(s): {invalid}. "
            f"Available classification models: {sorted(available)}"
        )
    return resolved


def infer_ntu_shape(bundle_info: Mapping[str, Any], train_loader) -> Optional[Tuple[int, int, int]]:
    input_shape = bundle_info.get("input_shape")
    if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
        return int(input_shape[0]), int(input_shape[1]), int(input_shape[2])
    return infer_input_shape_from_loader(train_loader)


def resolve_ntu_num_classes(bundle_info: Mapping[str, Any], train_loader) -> int:
    value = bundle_info.get("num_classes")
    if value is not None:
        try:
            resolved = int(value)
            if resolved > 0:
                return resolved
        except Exception:
            pass
    inferred = infer_num_classes(train_loader.dataset)
    if inferred is None or int(inferred) <= 0:
        raise RuntimeError("Unable to infer number of classes from bundle info or dataset labels.")
    return int(inferred)


def infer_spec_from_bundle(dataset_name: str, bundle_info: Mapping[str, Any], train_loader) -> DatasetSpec:
    shape = infer_ntu_shape(bundle_info, train_loader)
    if shape is not None:
        channels, freq_bins, time_steps = shape
        return DatasetSpec(dataset_name, channels=channels, freq_bins=freq_bins, time_steps=time_steps)
    raise RuntimeError("Unable to infer input shape from DataBundle info or train loader.")


def evaluate_splits(
    *,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    use_tqdm: bool,
    split_loaders: Mapping[str, Any],
    stage: Optional[str] = None,
    amp: Optional[bool] = None,
    amp_dtype: str = "float16",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split_name, loader in split_loaders.items():
        if loader is None:
            continue
        metrics, _ = evaluate_classifier(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            use_tqdm=use_tqdm,
            collect_predictions=False,
            collect_probabilities=False,
            desc=f"{stage}:{split_name}" if stage else f"Eval:{split_name}",
            amp=amp,
            amp_dtype=amp_dtype,
        )
        row: Dict[str, Any] = {
            "split": split_name,
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "loss": float(metrics.get("loss", 0.0)),
            "f1_macro": float(metrics.get("f1_macro", 0.0)),
            "f1_weighted": float(metrics.get("f1_weighted", 0.0)),
        }
        if stage:
            row["stage"] = stage
        rows.append(row)
    return rows


def save_model_weights(*, out_dir: str, filename_stem: str, model: nn.Module) -> str:
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    safe_name = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in filename_stem.strip())
    if not safe_name:
        safe_name = "model"
    weights_path = os.path.join(checkpoint_dir, f"{safe_name}.pt")
    torch.save(model.state_dict(), weights_path)
    return weights_path


def iter_final_eval_splits(val_loader, test_loader) -> Iterable[Tuple[str, Any]]:
    if val_loader is not None:
        yield "val", val_loader
    if test_loader is not None:
        yield "test", test_loader
