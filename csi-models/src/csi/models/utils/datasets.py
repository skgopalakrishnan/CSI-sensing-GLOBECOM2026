from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "DatasetSpec",
    "get_dataset_spec",
]


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset-specific shape metadata used by the model builders."""

    name: str
    channels: int
    freq_bins: int
    time_steps: int
    num_classes: Optional[int] = None
    task_name: Optional[str] = None


def _normalise_token(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _normalise_task_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = _normalise_token(value)
    mapping = {
        "har": "HumanActivityRecognition",
        "humanactivityrecognition": "HumanActivityRecognition",
        "human_activity_recognition": "HumanActivityRecognition",
        "hid": "HumanIdentification",
        "humanidentification": "HumanIdentification",
        "human_identification": "HumanIdentification",
        "humanid": "HumanIdentification",
        "human_id": "HumanIdentification",
        "reconstruction": "reconstruction",
        "classification": "classification",
    }
    return mapping.get(key, str(value))


def _spec(
    name: str,
    *,
    channels: int,
    freq_bins: int,
    time_steps: int,
    num_classes: Optional[int] = None,
    task_name: Optional[str] = None,
) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        channels=int(channels),
        freq_bins=int(freq_bins),
        time_steps=int(time_steps),
        num_classes=int(num_classes) if num_classes is not None else None,
        task_name=_normalise_task_name(task_name),
    )


_DATASET_SPECS: Dict[str, DatasetSpec] = {
    "ntu_fi": _spec("NTU-Fi_HAR", channels=3, freq_bins=114, time_steps=500, num_classes=6, task_name="har"),
    "ntu_fi_har": _spec("NTU-Fi_HAR", channels=3, freq_bins=114, time_steps=500, num_classes=6, task_name="har"),
    "ntu_har": _spec("NTU-Fi_HAR", channels=3, freq_bins=114, time_steps=500, num_classes=6, task_name="har"),
    "ntu": _spec("NTU-Fi_HAR", channels=3, freq_bins=114, time_steps=500, num_classes=6, task_name="har"),
    "ntu_fi_humanid": _spec("NTU-Fi-HumanID", channels=3, freq_bins=114, time_steps=500, num_classes=14, task_name="hid"),
    "ntu_fi_human_id": _spec("NTU-Fi-HumanID", channels=3, freq_bins=114, time_steps=500, num_classes=14, task_name="hid"),
    "ntu_hid": _spec("NTU-Fi-HumanID", channels=3, freq_bins=114, time_steps=500, num_classes=14, task_name="hid"),
    "ntu_humanid": _spec("NTU-Fi-HumanID", channels=3, freq_bins=114, time_steps=500, num_classes=14, task_name="hid"),
    "ntu_human_id": _spec("NTU-Fi-HumanID", channels=3, freq_bins=114, time_steps=500, num_classes=14, task_name="hid"),
    "csi_bench": _spec("CSI-Bench", channels=1, freq_bins=232, time_steps=500),
    "csi_bench_har": _spec(
        "CSI-Bench",
        channels=1,
        freq_bins=232,
        time_steps=500,
        task_name="HumanActivityRecognition",
    ),
    "csi_bench_hid": _spec(
        "CSI-Bench",
        channels=1,
        freq_bins=232,
        time_steps=500,
        task_name="HumanIdentification",
    ),
    "csi_bench_humanactivityrecognition": _spec(
        "CSI-Bench",
        channels=1,
        freq_bins=232,
        time_steps=500,
        task_name="HumanActivityRecognition",
    ),
    "csi_bench_human_identification": _spec(
        "CSI-Bench",
        channels=1,
        freq_bins=232,
        time_steps=500,
        task_name="HumanIdentification",
    ),
    "csi_bench_humanidentification": _spec(
        "CSI-Bench",
        channels=1,
        freq_bins=232,
        time_steps=500,
        task_name="HumanIdentification",
    ),
}


def _resolve_task_name(
    dataset: str,
    *,
    task_name: Optional[str],
    metadata: Optional[Mapping[str, Any]],
) -> Optional[str]:
    explicit = _normalise_task_name(task_name)
    if explicit is not None:
        return explicit
    if metadata:
        for key in ("task_name", "task", "task_family"):
            candidate = _normalise_task_name(metadata.get(key)) if metadata.get(key) is not None else None
            if candidate is not None:
                return candidate
    dataset_key = _normalise_token(dataset)
    if dataset_key.endswith("_har"):
        return "HumanActivityRecognition"
    if dataset_key.endswith("_hid") or "humanid" in dataset_key or "human_id" in dataset_key:
        return "HumanIdentification"
    return None


def _resolve_lookup_key(dataset: str, *, task_name: Optional[str]) -> str:
    dataset_key = _normalise_token(dataset)
    if task_name is None:
        return dataset_key
    task_key = _normalise_token(task_name)
    combined = f"{dataset_key}_{task_key}"
    if combined in _DATASET_SPECS:
        return combined
    if dataset_key == "csi_bench":
        if task_key in {"humanactivityrecognition", "human_activity_recognition"}:
            return "csi_bench_har"
        if task_key in {"humanidentification", "human_identification"}:
            return "csi_bench_hid"
    if dataset_key in {"ntu", "ntu_fi"}:
        if task_key in {"humanactivityrecognition", "human_activity_recognition"}:
            return "ntu_fi_har"
        if task_key in {"humanidentification", "human_identification"}:
            return "ntu_fi_humanid"
    return dataset_key


def get_dataset_spec(
    dataset: str,
    *,
    task_name: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> DatasetSpec:
    resolved_task = _resolve_task_name(dataset, task_name=task_name, metadata=metadata)
    lookup_key = _resolve_lookup_key(dataset, task_name=resolved_task)
    if lookup_key not in _DATASET_SPECS:
        raise ValueError(f"Unsupported dataset '{dataset}' for tailored models")

    spec = _DATASET_SPECS[lookup_key]
    if not metadata:
        return spec

    num_classes = metadata.get("num_classes")
    if num_classes is not None:
        try:
            return DatasetSpec(
                name=spec.name,
                channels=spec.channels,
                freq_bins=spec.freq_bins,
                time_steps=spec.time_steps,
                num_classes=int(num_classes),
                task_name=spec.task_name,
            )
        except Exception:
            return spec
    return spec
