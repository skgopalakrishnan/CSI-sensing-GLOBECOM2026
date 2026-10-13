from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch

from csi.attacks.evaluation import _collect_model_stats, _sample_input_tensor

from .shared_utils import parse_bool


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _profile_input_tensor(sample_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if sample_tensor is None:
        return None
    sample = sample_tensor.detach()
    if sample.ndim > 0 and sample.size(0) > 1:
        sample = sample[:1].detach()
    return sample


def resolve_profile_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    profile_cfg = dict(cfg.get("profiling", {}) or {})
    realtime_deadline_ms = _safe_float(profile_cfg.get("realtime_deadline_ms", 10.0))
    tinyml_memory_budget_kib = _safe_float(
        profile_cfg.get("tinyml_memory_budget_kib", profile_cfg.get("memory_budget_kib", 512.0))
    )
    return {
        "enabled": parse_bool(profile_cfg.get("enabled"), default=True),
        "latency_warmup": max(0, int(profile_cfg.get("latency_warmup", 3))),
        "latency_repeats": max(1, int(profile_cfg.get("latency_repeats", 10))),
        "realtime_deadline_ms": realtime_deadline_ms,
        "edge_device_flops_per_s": _safe_float(
            profile_cfg.get("edge_device_flops_per_s", profile_cfg.get("edge_flops_per_s"))
        ),
        "tinyml_memory_budget_kib": tinyml_memory_budget_kib,
    }


def profile_kwargs_from_cfg(profile_cfg: Mapping[str, Any], *, enabled: Optional[bool] = None) -> Dict[str, Any]:
    enabled_value = profile_cfg.get("enabled", True) if enabled is None else enabled
    return {
        "enabled": bool(enabled_value),
        "latency_warmup": max(0, int(profile_cfg.get("latency_warmup", 3))),
        "latency_repeats": max(1, int(profile_cfg.get("latency_repeats", 10))),
    }


def build_model_profile(
    *,
    model: torch.nn.Module,
    device: Any,
    profile_cfg: Mapping[str, Any],
    loader: Any = None,
    sample_tensor: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    raw_sample = sample_tensor if sample_tensor is not None else _sample_input_tensor(loader)
    profile_sample = _profile_input_tensor(raw_sample) if bool(profile_cfg.get("enabled", True)) else None
    return _collect_model_stats(
        model,
        profile_sample,
        device,
        latency_warmup=max(0, int(profile_cfg.get("latency_warmup", 3))),
        latency_repeats=max(1, int(profile_cfg.get("latency_repeats", 10))),
    )


def prefix_profile_keys(profile: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in dict(profile or {}).items()}


def _augment_prefixed_model_profile(row: Dict[str, Any], *, prefix: str, profile_cfg: Mapping[str, Any]) -> None:
    param_bytes = _safe_float(row.get(f"{prefix}model_param_bytes"))
    if param_bytes is not None:
        row.setdefault(f"{prefix}model_param_kib", float(param_bytes / 1024.0))
        row.setdefault(f"{prefix}model_param_mib", float(param_bytes / (1024.0 * 1024.0)))
        tinyml_budget_kib = _safe_float(profile_cfg.get("tinyml_memory_budget_kib"))
        if tinyml_budget_kib is not None:
            param_kib = _safe_float(row.get(f"{prefix}model_param_kib"))
            if param_kib is not None:
                row.setdefault(f"{prefix}tinyml_param_budget_kib", float(tinyml_budget_kib))
                row.setdefault(f"{prefix}tinyml_param_gap_kib", float(param_kib - tinyml_budget_kib))
                row.setdefault(f"{prefix}tinyml_param_feasible", bool(param_kib <= tinyml_budget_kib))

    model_flops = _safe_float(row.get(f"{prefix}model_flops"))
    if model_flops is not None:
        row.setdefault(f"{prefix}model_gflops", float(model_flops / 1e9))

    latency_ms_mean = _safe_float(row.get(f"{prefix}latency_ms_mean"))
    if latency_ms_mean is not None:
        latency_batch_size = max(1, _safe_int(row.get(f"{prefix}latency_batch_size")) or 1)
        latency_ms_per_example = float(latency_ms_mean / latency_batch_size)
        row.setdefault(f"{prefix}latency_ms_per_example", latency_ms_per_example)

        realtime_deadline_ms = _safe_float(profile_cfg.get("realtime_deadline_ms"))
        if realtime_deadline_ms is not None:
            row.setdefault(f"{prefix}realtime_deadline_ms", float(realtime_deadline_ms))
            row.setdefault(
                f"{prefix}realtime_inference_gap_ms",
                float(latency_ms_per_example - realtime_deadline_ms),
            )
            row.setdefault(
                f"{prefix}realtime_inference_feasible",
                bool(latency_ms_per_example <= realtime_deadline_ms),
            )

        edge_device_flops_per_s = _safe_float(profile_cfg.get("edge_device_flops_per_s"))
        if edge_device_flops_per_s is not None and edge_device_flops_per_s > 0.0 and model_flops is not None:
            edge_inference_ms_est = float((model_flops / edge_device_flops_per_s) * 1000.0)
            row.setdefault(f"{prefix}edge_device_flops_per_s", float(edge_device_flops_per_s))
            row.setdefault(f"{prefix}edge_inference_ms_est", edge_inference_ms_est)
            if realtime_deadline_ms is not None:
                row.setdefault(
                    f"{prefix}edge_inference_gap_ms",
                    float(edge_inference_ms_est - realtime_deadline_ms),
                )
                row.setdefault(
                    f"{prefix}edge_inference_feasible",
                    bool(edge_inference_ms_est <= realtime_deadline_ms),
                )


def _augment_attack_ratios(
    row: Dict[str, Any],
    *,
    prefix: str,
) -> None:
    attack_ms = _safe_float(row.get(f"{prefix}attack_generation_time_per_example_ms"))
    if attack_ms is None:
        attack_total_s = _safe_float(row.get(f"{prefix}attack_generation_time_s_total"))
        attack_num_examples = _safe_int(row.get(f"{prefix}attack_num_examples"))
        if attack_total_s is not None and attack_num_examples is not None and attack_num_examples > 0:
            attack_ms = float((attack_total_s * 1000.0) / attack_num_examples)
            row.setdefault(f"{prefix}attack_generation_time_per_example_ms", attack_ms)

    projection_ms = _safe_float(row.get(f"{prefix}projection_time_per_example_ms"))
    if projection_ms is None:
        projection_total_s = _safe_float(row.get(f"{prefix}projection_time_s_total"))
        attack_num_examples = _safe_int(row.get(f"{prefix}attack_num_examples"))
        if projection_total_s is not None and attack_num_examples is not None and attack_num_examples > 0:
            projection_ms = float((projection_total_s * 1000.0) / attack_num_examples)
            row.setdefault(f"{prefix}projection_time_per_example_ms", projection_ms)

    inference_ms = _safe_float(row.get(f"{prefix}latency_ms_per_example"))
    if inference_ms is None:
        return

    ratio_prefix = f"{prefix}" if prefix else ""
    edge_inference_ms_est = _safe_float(row.get(f"{prefix}edge_inference_ms_est"))

    if attack_ms is not None:
        attack_ratio = float(attack_ms / max(inference_ms, 1e-12))
        row.setdefault(f"{ratio_prefix}attack_to_inference_ratio", attack_ratio)
        if edge_inference_ms_est is not None:
            row.setdefault(f"{ratio_prefix}edge_attack_generation_ms_est", float(attack_ratio * edge_inference_ms_est))

    if projection_ms is not None:
        projection_ratio = float(projection_ms / max(inference_ms, 1e-12))
        row.setdefault(f"{ratio_prefix}projection_to_inference_ratio", projection_ratio)
        if edge_inference_ms_est is not None:
            row.setdefault(f"{ratio_prefix}edge_projection_time_ms_est", float(projection_ratio * edge_inference_ms_est))


def augment_profile_row(row: Dict[str, Any], profile_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    _augment_prefixed_model_profile(row, prefix="", profile_cfg=profile_cfg)
    _augment_prefixed_model_profile(row, prefix="source_", profile_cfg=profile_cfg)
    _augment_attack_ratios(row, prefix="")
    _augment_attack_ratios(row, prefix="source_")
    return row
