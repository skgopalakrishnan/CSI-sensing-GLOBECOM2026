from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Union

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[5]
_EXPERIMENTS_DIR = _REPO_ROOT / "experiments"
# Optional pair of mirrored storage roots (e.g. workstation and cluster mounts of the
# same share). Absolute paths under one root are remapped to the other when needed.
_WIFI_SENSING_ROOT_RESEARCH = Path(os.environ["CSI_STORAGE_ROOT"]) if os.environ.get("CSI_STORAGE_ROOT") else None
_WIFI_SENSING_ROOT_CLUSTER = Path(os.environ["CSI_STORAGE_ROOT_ALT"]) if os.environ.get("CSI_STORAGE_ROOT_ALT") else None


def sanitize_token(value: Union[str, os.PathLike, int, float, None]) -> str:
    if value is None:
        return "na"
    token = str(value).strip()
    if not token:
        return "na"
    allowed = []
    for ch in token:
        if ch.isalnum() or ch in ("-", "_"):
            allowed.append(ch)
        elif ch in (".", " "):
            allowed.append("_")
    return "".join(allowed) or "na"


def _remap_wifi_sensing_path(path_value: Union[str, os.PathLike]) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute() or _WIFI_SENSING_ROOT_RESEARCH is None or _WIFI_SENSING_ROOT_CLUSTER is None:
        return path

    text = str(path)
    candidates = [path]
    if text == str(_WIFI_SENSING_ROOT_RESEARCH) or text.startswith(str(_WIFI_SENSING_ROOT_RESEARCH) + os.sep):
        suffix = text[len(str(_WIFI_SENSING_ROOT_RESEARCH)) :]
        candidates.append(Path(str(_WIFI_SENSING_ROOT_CLUSTER) + suffix))
    elif text == str(_WIFI_SENSING_ROOT_CLUSTER) or text.startswith(str(_WIFI_SENSING_ROOT_CLUSTER) + os.sep):
        suffix = text[len(str(_WIFI_SENSING_ROOT_CLUSTER)) :]
        candidates.append(Path(str(_WIFI_SENSING_ROOT_RESEARCH) + suffix))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    if _WIFI_SENSING_ROOT_CLUSTER.exists():
        for candidate in candidates:
            if str(candidate).startswith(str(_WIFI_SENSING_ROOT_CLUSTER)):
                return candidate
    if _WIFI_SENSING_ROOT_RESEARCH.exists():
        for candidate in candidates:
            if str(candidate).startswith(str(_WIFI_SENSING_ROOT_RESEARCH)):
                return candidate
    return candidates[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: experiments/config/training/train_csi_autoencoder.yaml)",
    )
    return parser.parse_args()


def _load_config(path: Optional[str] = None) -> dict:
    default_cfg = _EXPERIMENTS_DIR / "config" / "train_one_model.yaml"
    cfg_path = str(_remap_wifi_sensing_path(path)) if path else str(default_cfg)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to load the config. Please install pyyaml.")
    with open(cfg_path, "r") as f:
        data = f.read()
    cfg = yaml.safe_load(data) or {}

    model_cfg = cfg.setdefault("model", {})
    weights_override = os.environ.get("CSI_MODEL_WEIGHTS_DIR_OVERRIDE")
    if weights_override:
        model_cfg["weights_dir_name"] = weights_override
    adv_weights_override = os.environ.get("CSI_MODEL_ADV_WEIGHTS_DIR_OVERRIDE")
    if adv_weights_override:
        model_cfg["adv_weights_dir_name"] = adv_weights_override

    group_override_env = os.environ.get("CSI_MODEL_GROUP_WEIGHTS_OVERRIDES")
    if group_override_env:
        groups = cfg.get("model_groups")
        if isinstance(groups, list):
            mapping: Dict[str, str] = {}
            for entry in group_override_env.replace(",", ";").split(";"):
                if not entry or "=" not in entry:
                    continue
                name, path_val = entry.split("=", 1)
                name = name.strip()
                path_val = path_val.strip()
                if name and path_val:
                    mapping[name] = path_val
            if mapping:
                for group in groups:
                    if not isinstance(group, dict):
                        continue
                    name = str(group.get("name") or "").strip()
                    if name and name in mapping:
                        group["weights_dir_name"] = mapping[name]

    return cfg


def _resolve_config_path_from_arg(arg: Optional[str]) -> Optional[str]:
    if not arg:
        return None
    if os.path.isabs(arg):
        return str(_remap_wifi_sensing_path(arg))
    if os.sep not in arg and not os.path.exists(arg):
        candidate = _EXPERIMENTS_DIR / "config" / arg
        if os.path.exists(candidate):
            return str(candidate)
    return arg


def parse_args() -> argparse.Namespace:
    return _parse_args()


def load_config(path: Optional[str] = None) -> dict:
    return _load_config(path)


def resolve_config_path_from_arg(arg: Optional[str]) -> Optional[str]:
    return _resolve_config_path_from_arg(arg)


def normalize_dataset_name(dataset: Optional[Union[str, os.PathLike]], default: Optional[str] = None) -> str:
    if dataset is None:
        if default is None:
            raise ValueError("Dataset name must be provided")
        return default
    value = str(dataset).strip()
    if not value:
        if default is not None:
            return default
        raise ValueError("Dataset name cannot be empty")
    key = value.lower().replace(" ", "_")
    canonical = {
        "ntu_fi": "NTU-Fi_HAR",
        "ntu-fi": "NTU-Fi_HAR",
        "ntu_fi_har": "NTU-Fi_HAR",
        "ntu-fi_har": "NTU-Fi_HAR",
        "ntu_fi_humanid": "NTU-Fi-HumanID",
        "ntu-fi_humanid": "NTU-Fi-HumanID",
        "ntu-fi-humanid": "NTU-Fi-HumanID",
        "ntu_fi-humanid": "NTU-Fi-HumanID",
        "ntu_fi_human_id": "NTU-Fi-HumanID",
        "ntu-fi_human_id": "NTU-Fi-HumanID",
        "ntu-fi_har_amp": "NTU-Fi_HAR",
        "csi-bench": "CSI-Bench",
        "csi_bench": "CSI-Bench",
    }
    return canonical.get(key, value)


def _normalize_dataset_name(dataset: str) -> str:
    return normalize_dataset_name(dataset)


def deep_update(target: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping):
            node = target.setdefault(key, {})
            if not isinstance(node, MutableMapping):
                raise TypeError(f"Cannot merge mapping into non-mapping at key '{key}'")
            deep_update(node, value)
        else:
            target[key] = value
    return target


def checkpoint_with_head_suffix(path: Union[str, os.PathLike], head_type: Optional[str]) -> Path:
    head_token = str(head_type).strip() if head_type else "head"
    head_suffix = head_token.upper()
    try:
        resolved = Path(path)
    except Exception:
        resolved = Path(str(path))
    try:
        if resolved.suffix:
            if resolved.stem.endswith(f"_{head_suffix}"):
                return resolved
            return resolved.with_name(f"{resolved.stem}_{head_suffix}{resolved.suffix}")
        base_name = resolved.name
        target_name = f"{base_name}.pt" if base_name.endswith(f"_{head_suffix}") else f"{base_name}_{head_suffix}.pt"
        return resolved.with_name(target_name)
    except Exception:
        return resolved


def _sanitize_tag_component(value: Any) -> str:
    token = str(value).strip()
    if not token:
        return ""
    return token.replace(" ", "").replace(".", "p")


def is_model_compatible_with_dataset(model_name: str, dataset_name: str) -> bool:
    model_key = model_name.strip().lower()
    dataset_key = dataset_name.strip().lower()

    if model_key.startswith("aeheadntu_"):
        return dataset_key.startswith("ntu")
    return True


def format_attack_budget_tag(attack_config: Optional[Mapping[str, Any]]) -> str:
    if not attack_config:
        return "budgetNA"
    snr_cfg = attack_config.get("snr_db") if isinstance(attack_config, Mapping) else None
    if isinstance(snr_cfg, (list, tuple)):
        snr_cfg = snr_cfg[0] if snr_cfg else None
    if snr_cfg is not None:
        return f"snr{_sanitize_tag_component(snr_cfg)}dB"
    eps_cfg = attack_config.get("epsilon") if isinstance(attack_config, Mapping) else None
    if eps_cfg is not None:
        return f"eps{_sanitize_tag_component(eps_cfg)}"
    return "budgetNA"


def format_seed_tag(seed: Optional[Union[int, str]]) -> str:
    if seed is None:
        return "seedNA"
    if isinstance(seed, str):
        seed_str = seed.strip()
        if not seed_str:
            return "seedNA"
        try:
            seed_int = int(seed_str)
        except ValueError:
            return f"seed{seed_str}"
        return f"seed{seed_int}"
    try:
        seed_int = int(seed)
    except (TypeError, ValueError):
        return f"seed{str(seed)}"
    return f"seed{seed_int}"


def build_adv_checkpoint_path(
    *,
    root: str,
    dataset_name: str,
    model_name: str,
    weights_dir: str,
    epoch_idx: int,
    attack_cfg: Optional[Mapping[str, Any]] = None,
    seed: Optional[Union[int, str]] = None,
    budget_tag: Optional[str] = None,
    seed_tag: Optional[str] = None,
    extra_tags: Optional[Sequence[str]] = None,
    create_dir: bool = True,
) -> str:
    budget_component = budget_tag or format_attack_budget_tag(attack_cfg)
    seed_component = seed_tag or format_seed_tag(seed)
    tags = [tag for tag in (budget_component, seed_component) if tag]
    if extra_tags:
         tags.extend(_sanitize_tag_component(tag) for tag in extra_tags if tag)
    folder = os.path.join(root, dataset_name, weights_dir)
    if tags:
        folder = os.path.join(folder, "_".join(tags))
    if create_dir:
        os.makedirs(folder, exist_ok=True)
    filename = f"{dataset_name}_{model_name}_epoch{epoch_idx:02d}.pt"
    return os.path.join(folder, filename)


__all__ = [
    "load_config",
    "parse_args",
    "resolve_config_path_from_arg",
    "_resolve_config_path_from_arg",
    "build_adv_checkpoint_path",
    "checkpoint_with_head_suffix",
    "deep_update",
    "format_attack_budget_tag",
    "format_seed_tag",
    "is_model_compatible_with_dataset",
    "normalize_dataset_name",
    "sanitize_token",
]
