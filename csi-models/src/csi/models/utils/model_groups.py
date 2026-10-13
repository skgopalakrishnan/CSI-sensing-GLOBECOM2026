from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass
class ModelGroup:
    name: str
    architectures: List[str]
    weights_dir: str


def _as_list(value: Optional[Any]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return []


def _unique_preserve(items: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def extract_model_groups(cfg: Mapping[str, Any]) -> List[ModelGroup]:
    groups: List[ModelGroup] = []

    raw_groups = cfg.get("model_groups")
    if isinstance(raw_groups, Mapping):
        raw_groups = [raw_groups]
    if isinstance(raw_groups, list):
        for idx, entry in enumerate(raw_groups):
            if not isinstance(entry, Mapping):
                continue
            architectures = entry.get("architectures") or entry.get("models") or []
            architectures_list = [str(a) for a in _as_list(architectures)]
            if not architectures_list:
                continue
            name = str(entry.get("name") or f"group_{idx + 1}")
            weights_dir = str(entry.get("weights_dir_name") or entry.get("weights_dir") or "").strip()
            groups.append(ModelGroup(name=name, architectures=architectures_list, weights_dir=weights_dir))

    if not groups:
        model_cfg = cfg.get("model")
        if isinstance(model_cfg, Mapping):
            architectures = model_cfg.get("architectures") or model_cfg.get("models") or []
            architectures_list = [str(a) for a in _as_list(architectures)]
            if architectures_list:
                weights_dir = str(model_cfg.get("weights_dir_name") or model_cfg.get("weights_dir") or "").strip()
                name = str(model_cfg.get("name") or "models")
                groups.append(ModelGroup(name=name, architectures=architectures_list, weights_dir=weights_dir))

    for legacy_key, default_name in (
        ("models", "models"),
        ("sensefi_models", "sensefi"),
        ("csi_bench_models", "csi-bench"),
    ):
        section = cfg.get(legacy_key)
        if section is None or any(g.name == default_name for g in groups):
            continue
        if isinstance(section, Mapping):
            architectures = section.get("architectures") or section.get("models") or []
            if architectures:
                weights_dir = str(section.get("weights_dir_name") or section.get("weights_dir") or "").strip()
                groups.append(
                    ModelGroup(
                        name=str(section.get("name") or default_name),
                        architectures=[str(a) for a in architectures],
                        weights_dir=weights_dir,
                    )
                )
        elif isinstance(section, list) and section:
            groups.append(ModelGroup(name=default_name, architectures=[str(a) for a in section], weights_dir=""))

    if not groups:
        raise ValueError("No model groups configured. Please define 'model_groups' in the config.")
    return groups


def build_model_weight_map(groups: List[ModelGroup]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for group in groups:
        for arch in group.architectures:
            mapping[arch] = group.weights_dir
    return mapping


def resolve_model_list(
    explicit: Optional[Any],
    group_names: Optional[Any],
    *,
    groups: List[ModelGroup],
    fallback: Sequence[str],
) -> List[str]:
    resolved: List[str] = []
    for item in _as_list(explicit):
        group_match = next((g for g in groups if g.name == item), None)
        if group_match is not None:
            resolved.extend(group_match.architectures)
        else:
            resolved.append(item)
    for group_name in _as_list(group_names):
        group_match = next((g for g in groups if g.name == group_name), None)
        if group_match is not None:
            resolved.extend(group_match.architectures)
    if not resolved:
        resolved = list(fallback)
    return _unique_preserve([str(item) for item in resolved])


def resolve_weights_dir(model_name: str, weight_map: Dict[str, str], default_dir: str) -> str:
    candidate = weight_map.get(model_name)
    if candidate:
        return candidate
    if candidate == "":
        return default_dir
    return default_dir


__all__ = [
    "ModelGroup",
    "build_model_weight_map",
    "extract_model_groups",
    "resolve_model_list",
    "resolve_weights_dir",
]
