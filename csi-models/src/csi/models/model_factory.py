"""Factory utilities for constructing tailored CSI models."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from torch.nn import Module

from .autoencoders import (
    AutoencoderClassifier,
    HEAD_ALIASES,
    build_autoencoder_classifier,
    build_autoencoder_from_config,
    build_classifier_head,
)
from .supervised.models import MODEL_BUILDERS
from .utils.datasets import DatasetSpec, get_dataset_spec
from .utils.model_utils import get_num_classes


__all__ = [
    "build_model",
    "infer_model_technique",
]


_DEFAULT_EPOCHS = 100
_AUTOENCODER_ALIASES: Dict[str, str] = {
    "autoencoder": "tiny_f_stride",
    "ae": "tiny_f_stride",
    "tiny": "tiny_f_stride",
    "tinyfstride": "tiny_f_stride",
    "tinyfstrideautoencoder": "tiny_f_stride",
    "tiny_f_stride": "tiny_f_stride",
}


def _normalise_dataset_key(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _normalise_model_name(name: str) -> str:
    key = name.lower().replace(" ", "").replace("-", "")
    aliases = {
        "timesformer": "csitimesformer",
        "timesformer1d": "csitimesformer",
        "tinyfstride": "tiny_f_stride",
        "tinyfstrideautoencoder": "tiny_f_stride",
    }
    return aliases.get(key, key)


def _normalise_technique(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "sl": "supervised",
        "classification": "supervised",
        "classifier": "supervised",
        "auto_encoder": "autoencoder",
        "reconstruction": "autoencoder",
        "ae_classifier": "autoencoder_classifier",
        "autoencoder_head": "autoencoder_classifier",
    }
    return aliases.get(key, key)


_BUILDER_ALLOWED_KWARGS: Dict[str, Tuple[str, ...]] = {
    "mlp": ("hidden_dim", "num_layers", "dropout"),
    "lenet": ("base_filters", "fc_hidden_dim", "pool_size", "dropout"),
    "resnet18": ("pretrained",),
    "bilstm": ("hidden_dim", "num_layers", "bidirectional", "dropout"),
    "csitimesformer": (
        "patch_f",
        "patch_t",
        "emb_dim",
        "depth",
        "num_heads",
        "mlp_ratio",
        "dropout",
    ),
}


_BUILDER_CASTS: Dict[str, Dict[str, Any]] = {
    "mlp": {
        "hidden_dim": int,
        "num_layers": int,
        "dropout": float,
    },
    "lenet": {
        "base_filters": int,
        "fc_hidden_dim": int,
        "pool_size": int,
        "dropout": float,
    },
    "resnet18": {
        "pretrained": bool,
    },
    "bilstm": {
        "hidden_dim": int,
        "num_layers": int,
        "bidirectional": bool,
        "dropout": float,
    },
    "csitimesformer": {
        "patch_f": int,
        "patch_t": int,
        "emb_dim": int,
        "depth": int,
        "num_heads": int,
        "mlp_ratio": float,
        "dropout": float,
    },
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _largest_divisor_at_most(value: int, ceiling: int) -> int:
    upper = min(max(1, int(ceiling)), int(value))
    divisors = [d for d in range(1, upper + 1) if value % d == 0]
    return max(divisors) if divisors else 1


def _infer_builder_defaults(spec: DatasetSpec, model_key: str) -> Dict[str, Any]:
    if model_key == "bilstm":
        feature_dim = spec.channels * spec.freq_bins
        if feature_dim <= 512:
            hidden_dim = 128
            num_layers = 1
            dropout = 0.1
        elif feature_dim <= 768:
            hidden_dim = 192
            num_layers = 1
            dropout = 0.12
        else:
            hidden_dim = min(256, (feature_dim // 64) * 64)
            num_layers = 2
            dropout = 0.15

        return {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }

    if model_key == "csitimesformer":
        patch_f = _largest_divisor_at_most(spec.freq_bins, 16)
        patch_t = _largest_divisor_at_most(spec.time_steps, 32)
        return {
            "patch_f": patch_f,
            "patch_t": patch_t,
            "emb_dim": 128,
            "depth": 4,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
        }

    return {}


def _override_spec_with_metadata(spec: DatasetSpec, arch_metadata: Optional[Mapping[str, Any]]) -> DatasetSpec:
    if not isinstance(arch_metadata, Mapping):
        return spec
    channels = spec.channels
    freq_bins = spec.freq_bins
    time_steps = spec.time_steps
    num_classes = spec.num_classes
    task_name = spec.task_name

    if "channels" in arch_metadata:
        try:
            channels = int(arch_metadata["channels"])
        except Exception:
            pass
    if "freq_bins" in arch_metadata:
        try:
            freq_bins = int(arch_metadata["freq_bins"])
        except Exception:
            pass
    if "time_steps" in arch_metadata:
        try:
            time_steps = int(arch_metadata["time_steps"])
        except Exception:
            pass
    if arch_metadata.get("num_classes") is not None:
        try:
            num_classes = int(arch_metadata["num_classes"])
        except Exception:
            pass
    if arch_metadata.get("task_name") is not None:
        task_name = str(arch_metadata["task_name"])

    input_shape = arch_metadata.get("input_shape")
    if isinstance(input_shape, (list, tuple)):
        if len(input_shape) >= 1:
            try:
                channels = int(input_shape[0])
            except Exception:
                pass
        if len(input_shape) >= 2:
            try:
                freq_bins = int(input_shape[1])
            except Exception:
                pass
        if len(input_shape) >= 3:
            try:
                time_steps = int(input_shape[2])
            except Exception:
                pass

    return DatasetSpec(
        spec.name,
        int(channels),
        int(freq_bins),
        int(time_steps),
        num_classes=num_classes,
        task_name=task_name,
    )


def _extract_builder_kwargs(spec: DatasetSpec, model_key: str, arch_metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    overrides = dict(_infer_builder_defaults(spec, model_key))
    allowed = _BUILDER_ALLOWED_KWARGS.get(model_key, ())
    casts = _BUILDER_CASTS.get(model_key, {})

    sources: Tuple[Mapping[str, Any], ...] = ()
    if isinstance(arch_metadata, Mapping):
        sources += (arch_metadata,)
        for nested_key in ("model", "config", "params"):
            nested = arch_metadata.get(nested_key)
            if isinstance(nested, Mapping):
                sources += (nested,)
    for source in sources:
        for key in allowed:
            if key not in source:
                continue
            overrides[key] = source[key]

    for key, caster in casts.items():
        if key not in overrides:
            continue
        try:
            if caster is bool:
                overrides[key] = _as_bool(overrides[key])
            else:
                overrides[key] = caster(overrides[key])
        except Exception:
            overrides.pop(key, None)
    return overrides


def _resolve_sample_shape(
    arch_metadata: Optional[Mapping[str, Any]],
    *,
    channels: int,
    freq_bins: int,
    time_steps: int,
) -> Tuple[int, int, int]:
    if arch_metadata is None:
        return channels, freq_bins, time_steps
    input_shape = arch_metadata.get("input_shape")
    if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
        return int(input_shape[0]), int(input_shape[1]), int(input_shape[2])
    return channels, freq_bins, time_steps


def _resolve_num_classes(
    dataset: str,
    spec: DatasetSpec,
    num_classes: Optional[int],
    arch_metadata: Optional[Mapping[str, Any]],
) -> int:
    if num_classes is not None:
        return int(num_classes)
    return get_num_classes(
        dataset,
        task_name=spec.task_name,
        arch_metadata=arch_metadata,
    )


def _extract_autoencoder_config(
    model_key: str,
    arch_metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if isinstance(arch_metadata, Mapping):
        encoder_section = arch_metadata.get("encoder")
        if isinstance(encoder_section, Mapping):
            nested = encoder_section.get("config")
            if isinstance(nested, Mapping):
                cfg.update(dict(nested))
            elif isinstance(encoder_section.get("model"), Mapping):
                cfg.update(dict(encoder_section["model"]))
            else:
                for key, value in encoder_section.items():
                    if key not in {"type", "config", "model"}:
                        cfg[key] = value
                if encoder_section.get("type") is not None:
                    cfg["type"] = encoder_section["type"]

        model_section = arch_metadata.get("model")
        if isinstance(model_section, Mapping):
            nested = model_section.get("config")
            if isinstance(nested, Mapping):
                cfg.update(dict(nested))
            else:
                for key, value in model_section.items():
                    if key != "config":
                        cfg[key] = value

        top_level_cfg = arch_metadata.get("config")
        if isinstance(top_level_cfg, Mapping):
            cfg.update(dict(top_level_cfg))

    cfg.pop("in_channels", None)
    cfg["type"] = cfg.get("type", _AUTOENCODER_ALIASES.get(model_key, "tiny_f_stride"))
    return cfg


def _resolve_head_type(model_key: str, arch_metadata: Optional[Mapping[str, Any]]) -> str:
    if model_key in HEAD_ALIASES:
        return HEAD_ALIASES[model_key]
    if isinstance(arch_metadata, Mapping):
        head_section = arch_metadata.get("head")
        if isinstance(head_section, Mapping) and head_section.get("type") is not None:
            return str(head_section["type"]).strip().lower()
    return "tcn"


def infer_model_technique(
    model_name: str,
    *,
    arch_metadata: Optional[Mapping[str, Any]] = None,
    technique: Optional[str] = None,
) -> str:
    resolved_technique = _normalise_technique(technique)
    if resolved_technique is not None:
        return resolved_technique

    if isinstance(arch_metadata, Mapping):
        for key in ("technique", "training_technique", "model_technique"):
            value = _normalise_technique(arch_metadata.get(key))
            if value is not None:
                return value

    model_key = _normalise_model_name(model_name)
    if model_key in MODEL_BUILDERS:
        return "supervised"
    if model_key in HEAD_ALIASES:
        return "autoencoder_classifier"
    if model_key in _AUTOENCODER_ALIASES:
        return "autoencoder"

    if isinstance(arch_metadata, Mapping):
        if isinstance(arch_metadata.get("head"), Mapping):
            return "autoencoder_classifier"
        if isinstance(arch_metadata.get("encoder"), Mapping):
            return "autoencoder"

    raise ValueError(f"Unsupported model '{model_name}'.")


def build_model(
    dataset: str,
    model_name: str,
    num_classes: Optional[int] = None,
    *,
    arch_metadata: Optional[Mapping[str, Any]] = None,
    technique: Optional[str] = None,
) -> Tuple[Module, int]:
    """Construct a dataset-aware model and return it with a suggested epoch budget."""

    metadata_mapping: Optional[Mapping[str, Any]] = arch_metadata if isinstance(arch_metadata, Mapping) else None
    spec = get_dataset_spec(
        dataset,
        task_name=metadata_mapping.get("task_name") if metadata_mapping else None,
        metadata=metadata_mapping,
    )
    spec = _override_spec_with_metadata(spec, metadata_mapping)

    model_key = _normalise_model_name(model_name)
    resolved_technique = infer_model_technique(
        model_name,
        arch_metadata=metadata_mapping,
        technique=technique,
    )

    if resolved_technique == "supervised":
        resolved_classes = _resolve_num_classes(dataset, spec, num_classes, metadata_mapping)
        builder_kwargs = _extract_builder_kwargs(spec, model_key, metadata_mapping)
        model = MODEL_BUILDERS[model_key](spec, num_classes=resolved_classes, **builder_kwargs)
        return model, _DEFAULT_EPOCHS

    sample_shape = _resolve_sample_shape(
        metadata_mapping,
        channels=spec.channels,
        freq_bins=spec.freq_bins,
        time_steps=spec.time_steps,
    )

    if resolved_technique == "autoencoder":
        encoder_cfg = _extract_autoencoder_config(model_key, metadata_mapping)
        model = build_autoencoder_from_config(encoder_cfg, sample_shape=sample_shape)
        return model, _DEFAULT_EPOCHS

    if resolved_technique != "autoencoder_classifier":
        raise ValueError(
            f"Unsupported technique '{resolved_technique}' for model '{model_name}'."
        )

    resolved_classes = _resolve_num_classes(dataset, spec, num_classes, metadata_mapping)
    head_type = _resolve_head_type(model_key, metadata_mapping)
    if not metadata_mapping or not isinstance(metadata_mapping.get("encoder"), Mapping):
        model = build_autoencoder_classifier(spec, resolved_classes, head_type=head_type)
        return model, _DEFAULT_EPOCHS

    encoder_cfg = _extract_autoencoder_config(model_key, metadata_mapping)
    encoder = build_autoencoder_from_config(encoder_cfg, sample_shape=sample_shape)

    head_section = metadata_mapping.get("head")
    if not isinstance(head_section, Mapping):
        head_section = {}
    project_dim = head_section.get("project_dim")
    dropout = head_section.get("dropout", 0.0)
    head = build_classifier_head(
        head_type=head_type,
        in_channels=int(getattr(encoder, "latent_channels", 0)),
        num_classes=resolved_classes,
        project_dim=int(project_dim) if project_dim is not None else None,
        dropout=float(dropout),
        tcn_cfg=head_section.get("tcn"),
        gru_cfg=head_section.get("gru"),
        ssm_cfg=head_section.get("ssm"),
    )
    model = AutoencoderClassifier(encoder, head)
    return model, _DEFAULT_EPOCHS
