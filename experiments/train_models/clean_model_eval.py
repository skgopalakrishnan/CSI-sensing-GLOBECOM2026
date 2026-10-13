"""Evaluate saved supervised model checkpoints on NTU and CSI-Bench splits."""

from __future__ import annotations

import os
import sys
import traceback
import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from csi.data import get_experiment_loaders
from csi.data.utils import filter_none_collate, load_csi_bench_supervised
from csi.models.supervised import evaluate_classifier
from csi.models.utils.datasets import DatasetSpec
from csi.models.utils.experiment import load_config, resolve_config_path_from_arg, sanitize_token
from csi.models.utils.gpu_utils import _resolve_device
from csi.models.utils.model_utils import load_arch_metadata
from csi.models.utils.results import prepare_results_dir, save_config, save_results_rows, save_run_meta
from csi.models.utils.seeding import configure_from_env_and_config
from ..common.shared_utils import (
    DEFAULT_CLEAN_WEIGHTS_DIR,
    collect_csi_bench_task_metadata,
    infer_input_shape_from_loader,
    normalize_csi_bench_task_name,
    parse_bool,
    parse_splits,
    parse_token_list,
    prepare_run_dirs,
    print_training_start,
    save_json,
)
from ..common.optuna_clean_runtime import build_model_for_checkpoint, ensure_optuna_clean_checkpoint
from ..common.edge_profile import augment_profile_row, build_model_profile, resolve_profile_cfg
from ..common.train_models_utils import (
    DEFAULT_CSI_BENCH_SUBCARRIERS,
    DEFAULT_CSI_BENCH_TIMESTEPS,
    as_list,
    configure_torch_performance,
    infer_ntu_shape,
    resolve_ntu_num_classes,
    resolve_project_path,
    resolve_supervised_models,
)


def _save_classification_artifacts(
    *,
    out_dir: str,
    split_name: str,
    targets: torch.Tensor,
    preds: torch.Tensor,
) -> Dict[str, str]:
    run_dirs = prepare_run_dirs(out_dir)
    reports_dir = run_dirs["reports_dir"]
    figures_dir = run_dirs["figures_dir"]

    y_true = targets.detach().cpu().numpy().tolist()
    y_pred = preds.detach().cpu().numpy().tolist()
    label_values = sorted({int(value) for value in y_true} | {int(value) for value in y_pred})

    report = classification_report(
        y_true,
        y_pred,
        labels=label_values,
        output_dict=True,
        zero_division=0,
    )
    report_json_path = os.path.join(reports_dir, f"classification_report_{split_name}.json")
    report_csv_path = os.path.join(reports_dir, f"classification_report_{split_name}.csv")
    save_json(report_json_path, report)  # type: ignore[arg-type]
    with open(report_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "precision", "recall", "f1-score", "support"])
        for label, stats in report.items():
            if not isinstance(stats, Mapping):
                continue
            writer.writerow(
                [
                    label,
                    stats.get("precision"),
                    stats.get("recall"),
                    stats.get("f1-score"),
                    stats.get("support"),
                ]
            )

    cm = confusion_matrix(y_true, y_pred, labels=label_values)
    cm_counts_json_path = os.path.join(reports_dir, f"confusion_matrix_{split_name}_counts.json")
    cm_counts_csv_path = os.path.join(reports_dir, f"confusion_matrix_{split_name}_counts.csv")
    save_json(
        cm_counts_json_path,
        {
            "labels": label_values,
            "matrix": cm.tolist(),
        },
    )
    with open(cm_counts_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + [str(label) for label in label_values])
        for idx, row in enumerate(cm.tolist()):
            writer.writerow([str(label_values[idx])] + row)

    row_totals = cm.sum(axis=1, keepdims=True)
    cm_row_norm = cm.astype(float)
    nonzero_mask = row_totals > 0
    cm_row_norm[nonzero_mask[:, 0]] /= row_totals[nonzero_mask[:, 0]]
    cm_row_norm_json_path = os.path.join(reports_dir, f"confusion_matrix_{split_name}_row_normalized.json")
    cm_row_norm_csv_path = os.path.join(reports_dir, f"confusion_matrix_{split_name}_row_normalized.csv")
    save_json(
        cm_row_norm_json_path,
        {
            "labels": label_values,
            "matrix": cm_row_norm.tolist(),
        },
    )
    with open(cm_row_norm_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + [str(label) for label in label_values])
        for idx, row in enumerate(cm_row_norm.tolist()):
            writer.writerow([str(label_values[idx])] + row)

    cm_counts_png_path = os.path.join(figures_dir, f"confusion_matrix_{split_name}_counts.png")
    cm_row_norm_png_path = os.path.join(figures_dir, f"confusion_matrix_{split_name}_row_normalized.png")
    try:
        import matplotlib.pyplot as plt

        display_labels = [str(label) for label in label_values]
        disp_counts = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp_counts.plot(cmap="Blues", xticks_rotation=45, values_format="d")
        if disp_counts.figure_ is not None:
            disp_counts.figure_.tight_layout()
            disp_counts.figure_.savefig(cm_counts_png_path, dpi=180, bbox_inches="tight")
            disp_counts.figure_.clf()
            plt.close(disp_counts.figure_)

        disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_row_norm, display_labels=display_labels)
        disp_norm.plot(cmap="Blues", xticks_rotation=45, values_format=".2f")
        if disp_norm.figure_ is not None:
            disp_norm.figure_.tight_layout()
            disp_norm.figure_.savefig(cm_row_norm_png_path, dpi=180, bbox_inches="tight")
            disp_norm.figure_.clf()
            plt.close(disp_norm.figure_)
    except Exception as plot_exc:
        print(f"[WARN] Failed to save confusion matrix figures for split '{split_name}': {plot_exc}")

    return {
        "classification_report_json": report_json_path,
        "classification_report_csv": report_csv_path,
        "confusion_matrix_counts_json": cm_counts_json_path,
        "confusion_matrix_counts_csv": cm_counts_csv_path,
        "confusion_matrix_row_normalized_json": cm_row_norm_json_path,
        "confusion_matrix_row_normalized_csv": cm_row_norm_csv_path,
        "confusion_matrix_counts_png": cm_counts_png_path,
        "confusion_matrix_row_normalized_png": cm_row_norm_png_path,
    }


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    if "model_state_dict" in payload and isinstance(payload["model_state_dict"], Mapping):
        return payload["model_state_dict"]  # type: ignore[return-value]
    if "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
        return payload["state_dict"]  # type: ignore[return-value]
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload  # type: ignore[return-value]
    raise ValueError("Checkpoint does not contain a readable state_dict.")


def _load_checkpoint_into_model(
    *,
    model: nn.Module,
    checkpoint_path: str,
    strict: bool,
) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = _extract_state_dict(payload)
    load_result = model.load_state_dict(state, strict=strict)
    return {
        "strict": bool(strict),
        "num_state_keys": len(state),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def _collect_model_profile(
    *,
    model: nn.Module,
    loader: Any,
    device: torch.device,
    cfg: Mapping[str, Any],
    checkpoint_path: Path,
) -> Dict[str, Any]:
    profile_cfg = resolve_profile_cfg(cfg)
    profile = build_model_profile(
        model=model,
        device=device,
        profile_cfg=profile_cfg,
        loader=loader,
    )
    profile = augment_profile_row(dict(profile), profile_cfg)

    total_params = int(sum(param.numel() for param in model.parameters()))
    trainable_params = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    total_bytes = int(sum(int(param.numel()) * int(param.element_size()) for param in model.parameters()))
    trainable_bytes = int(
        sum(int(param.numel()) * int(param.element_size()) for param in model.parameters() if param.requires_grad)
    )
    profile.setdefault("model_param_count", total_params)
    profile["model_trainable_param_count"] = trainable_params
    profile.setdefault("model_param_bytes", total_bytes)
    profile["model_trainable_param_bytes"] = trainable_bytes
    profile["model_trainable_param_mib"] = float(trainable_bytes / (1024.0 * 1024.0))

    if checkpoint_path.exists():
        checkpoint_size_bytes = int(checkpoint_path.stat().st_size)
        profile["checkpoint_size_bytes"] = checkpoint_size_bytes
        profile["checkpoint_size_mib"] = float(checkpoint_size_bytes / (1024.0 * 1024.0))

    profile["profile_device"] = str(device)
    profile["profile_device_type"] = str(device.type)
    if device.type == "cuda" and torch.cuda.is_available():
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        profile["profile_device_index"] = int(device_index)
        profile["profile_device_name"] = str(props.name)
        profile["profile_device_total_memory_bytes"] = int(props.total_memory)
        profile["profile_cuda_capability"] = f"{props.major}.{props.minor}"

    return profile


def _normalise_selection_mode(value: Any) -> str:
    key = str(value or "auto").strip().lower()
    if key not in {"auto", "alias", "latest"}:
        raise ValueError("weights.selection must be one of: auto, alias, latest")
    return key


def _resolve_checkpoint(
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

    selection_key = _normalise_selection_mode(selection)
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
        raise FileNotFoundError(f"No checkpoints matched pattern '{pattern}' in: {base_dir}")

    token = sanitize_token(model_name).lower()
    token_matches = [path for path in candidates if token in sanitize_token(path.stem).lower()]
    pool = token_matches if token_matches else candidates
    return max(pool, key=lambda path: path.stat().st_mtime)


def _nested_checkpoint_override(
    mapping: Any,
    key_1: str,
    key_2: str,
) -> Optional[str]:
    if not isinstance(mapping, Mapping):
        return None
    first = mapping.get(key_1)
    if isinstance(first, Mapping):
        second = first.get(key_2)
        return str(second) if second else None
    return None


def _merge_arch_metadata(base: Optional[Mapping[str, Any]], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base or {})
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_arch_metadata(current, value)
        else:
            merged[key] = value
    return merged


def _resolve_csi_bench_checkpoint_dir(
    *,
    weights_dataset_root: str,
    weights_dir_name: str,
    task_name: str,
) -> Path:
    root = resolve_project_path(weights_dataset_root)
    # Support both current and legacy export layouts:
    # 1) <...>/CSI-Bench/model_params.../<Task>
    # 2) <...>/CSI-Bench/CSI-Bench/model_params.../<Task>
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
            seen.add(candidate)
            ordered.append(candidate)
    for candidate in ordered:
        if candidate.exists():
            return candidate
    return ordered[0]


def _as_weight_dir_list(value: Any, *, default: str) -> List[str]:
    if value is None:
        return [default]
    if isinstance(value, str):
        parts = [token.strip() for token in value.split(",") if token.strip()]
        return parts if parts else [default]
    if isinstance(value, (list, tuple)):
        out = [str(token).strip() for token in value if str(token).strip()]
        return out if out else [default]
    return [default]


def _resolve_ntu_eval_splits(
    requested: Sequence[str],
    *,
    val_loader: Any,
    test_loader: Any,
) -> List[Tuple[str, Any]]:
    loaders = {
        "val": val_loader,
        "validation": val_loader,
        "test": test_loader,
    }
    resolved: List[Tuple[str, Any]] = []
    for raw in requested:
        key = str(raw).strip().lower().replace("-", "_")
        if key in {"val_id", "validation_id"}:
            key = "val"
        if key in {"test_id"}:
            key = "test"
        loader = loaders.get(key)
        if loader is None:
            continue
        label = "val" if key.startswith("val") else "test"
        resolved.append((label, loader))
    dedup: Dict[str, Any] = {}
    for name, loader in resolved:
        dedup[name] = loader
    return list(dedup.items())


def _normalise_csi_split_name(split_name: str) -> str:
    key = str(split_name).strip()
    lower = key.lower().replace("-", "_")
    if lower in {"train", "train_id"}:
        return "train_id"
    if lower in {"val", "validation", "val_id"}:
        return "val_id"
    if lower in {"test", "test_id"}:
        return "test_id"
    return key


def _loader_key_for_csi_split(split_name: str) -> str:
    key = _normalise_csi_split_name(split_name)
    if key == "train_id":
        return "train"
    if key == "val_id":
        return "val"
    if key == "test_id":
        return "test"
    if key.startswith("test_"):
        return key
    return f"test_{key}"


def _run_ntu_eval(
    *,
    cfg: Mapping[str, Any],
    config_path: Optional[str],
    experiment_name: str,
    model_names: Sequence[str],
    device: torch.device,
    weights_cfg: Mapping[str, Any],
) -> None:
    ntu_cfg = dict(cfg.get("ntu", {}) or {})
    weights_ntu = dict(weights_cfg.get("ntu", {}) or {})

    datasets = as_list(ntu_cfg.get("datasets") or ntu_cfg.get("dataset"), default=["NTU-Fi_HAR", "NTU-Fi-HumanID"])
    root = str(ntu_cfg.get("root", cfg.get("common", {}).get("root", "datasets")))
    task = str(ntu_cfg.get("task", "classification"))
    timesteps = int(ntu_cfg.get("timesteps", 250))
    batch_sizes = ntu_cfg.get("batch_sizes")
    base_normalisation = str(ntu_cfg.get("base_normalisation", "none"))
    canon_normalisation = str(ntu_cfg.get("canon_normalisation", "per_subcarrier"))
    variant = str(ntu_cfg.get("variant", "raw"))
    smoothing = ntu_cfg.get("smoothing")
    num_workers_raw = ntu_cfg.get("num_workers")
    num_workers = int(num_workers_raw) if num_workers_raw is not None else None
    pin_memory_raw = ntu_cfg.get("pin_memory")
    pin_memory = None if pin_memory_raw is None else parse_bool(pin_memory_raw, default=(device.type == "cuda"))
    persistent_workers_raw = ntu_cfg.get("persistent_workers")
    persistent_workers = None if persistent_workers_raw is None else parse_bool(persistent_workers_raw, default=True)
    prefetch_factor_raw = ntu_cfg.get("prefetch_factor")
    prefetch_factor = int(prefetch_factor_raw) if prefetch_factor_raw is not None else None
    augment_train = dict(ntu_cfg.get("augment", {}) or {})
    augment_eval = dict(ntu_cfg.get("augment_eval", augment_train) or augment_train)
    data_seed = int(ntu_cfg.get("seed", cfg.get("data", {}).get("seed", 42)))
    eval_splits = parse_token_list(ntu_cfg.get("eval_splits")) or ["test"]
    strict_load = parse_bool(weights_cfg.get("strict_load"), default=True)
    selection = _normalise_selection_mode(weights_cfg.get("selection"))
    checkpoint_glob = weights_ntu.get("checkpoint_glob")
    checkpoint_paths = weights_ntu.get("checkpoint_paths")
    weights_root = str(weights_ntu.get("datasets_root", root))
    weights_dir_names = _as_weight_dir_list(
        weights_ntu.get("weights_dir_names", weights_ntu.get("weights_dir_name")),
        default=str(weights_cfg.get("weights_dir_name") or DEFAULT_CLEAN_WEIGHTS_DIR),
    )

    criterion = nn.CrossEntropyLoss()
    for dataset_name in datasets:
        for model_name in model_names:
            for weights_dir_name in weights_dir_names:
                model_token = sanitize_token(model_name)
                weights_token = sanitize_token(weights_dir_name)
                out_dir, ts, cfg_name = prepare_results_dir(
                    experiment_name=experiment_name,
                    cfg_path=config_path,
                    relative_parts=["ntu", dataset_name, weights_token, model_token],
                )
                try:
                    save_run_meta(
                        out_dir,
                        {
                            "domain": "ntu",
                            "dataset": dataset_name,
                            "model": model_name,
                            "eval_splits": eval_splits,
                            "weights_dir_name": weights_dir_name,
                            "strict_load": strict_load,
                            "selection": selection,
                        },
                    )
                    save_config(out_dir, dict(cfg))
                    print_training_start(
                        f"Clean eval NTU ({dataset_name}/{model_name})",
                        {
                            "output_dir": out_dir,
                            "dataset": dataset_name,
                            "model": model_name,
                            "eval_splits": eval_splits,
                            "weights_dir_name": weights_dir_name,
                            "device": str(device),
                        },
                    )

                    bundle = get_experiment_loaders(
                        dataset_name,
                        root,
                        task=task,
                        timesteps=timesteps,
                        batch_sizes=batch_sizes,
                        device=device,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        persistent_workers=persistent_workers,
                        prefetch_factor=prefetch_factor,
                        base_normalisation=base_normalisation,
                        canon_normalisation=canon_normalisation,
                        variant=variant,
                        smoothing=smoothing,
                        augment_train=augment_train,
                        augment_eval=augment_eval,
                        seed=data_seed,
                    )
                    train_loader = bundle.train_loader
                    val_loader = bundle.val_loader
                    test_loader = bundle.test_loader
                    if train_loader is None:
                        raise RuntimeError(f"Missing NTU train loader for {dataset_name}")
                    eval_loaders = _resolve_ntu_eval_splits(
                        eval_splits,
                        val_loader=val_loader,
                        test_loader=test_loader,
                    )
                    if not eval_loaders:
                        raise RuntimeError(
                            f"No requested NTU eval splits available for {dataset_name}. "
                            f"Requested={eval_splits}"
                        )

                    bundle_info = bundle.info or {}
                    shape = infer_ntu_shape(bundle_info, train_loader)
                    num_classes = resolve_ntu_num_classes(bundle_info, train_loader)
                    arch_metadata: Dict[str, Any] = {"num_classes": num_classes}
                    if shape is not None:
                        channels, freq_bins, time_steps = shape
                        arch_metadata.update(
                            {
                                "channels": channels,
                                "freq_bins": freq_bins,
                                "time_steps": time_steps,
                                "input_shape": [channels, freq_bins, time_steps],
                            }
                        )
                    base_dir = resolve_project_path(weights_root) / dataset_name / weights_dir_name
                    explicit_checkpoint = _nested_checkpoint_override(checkpoint_paths, dataset_name, model_name)
                    explicit_checkpoint = ensure_optuna_clean_checkpoint(
                        domain="ntu",
                        model_name=model_name,
                        weights_dir_name=weights_dir_name,
                        weights_cfg=weights_cfg,
                        canonical_base_dir=base_dir,
                        alias_stems=[f"{dataset_name}_{model_name}"],
                        dataset_name=dataset_name,
                    ) or explicit_checkpoint
                    checkpoint_path = _resolve_checkpoint(
                        base_dir=base_dir,
                        model_name=model_name,
                        selection=selection,
                        alias_stems=[f"{dataset_name}_{model_name}"],
                        checkpoint_glob=str(checkpoint_glob) if checkpoint_glob else None,
                        explicit_checkpoint=explicit_checkpoint,
                    )
                    checkpoint_meta = load_arch_metadata(checkpoint_path)
                    resolved_arch_metadata = _merge_arch_metadata(checkpoint_meta, arch_metadata)
                    if shape is None:
                        inferred = infer_input_shape_from_loader(train_loader)
                        if inferred is None:
                            raise RuntimeError(f"Unable to infer input shape for {dataset_name}/{model_name}.")
                        channels, freq_bins, time_steps = inferred
                    spec = DatasetSpec(dataset_name, channels=channels, freq_bins=freq_bins, time_steps=time_steps)
                    model, _ = build_model_for_checkpoint(
                        domain="ntu",
                        model_name=model_name,
                        weights_dir_name=weights_dir_name,
                        weights_cfg=weights_cfg,
                        spec=spec,
                        dataset_key_for_model=dataset_name,
                        num_classes=num_classes,
                        arch_metadata=resolved_arch_metadata,
                        dataset_name=dataset_name,
                    )
                    model = model.to(device)
                    load_info = _load_checkpoint_into_model(
                        model=model,
                        checkpoint_path=str(checkpoint_path),
                        strict=strict_load,
                    )
                    model_profile = _collect_model_profile(
                        model=model,
                        loader=eval_loaders[0][1],
                        device=device,
                        cfg=cfg,
                        checkpoint_path=checkpoint_path,
                    )

                    rows: List[Dict[str, Any]] = []
                    for split_name, loader in eval_loaders:
                        metrics, extras = evaluate_classifier(
                            model=model,
                            loader=loader,
                            criterion=criterion,
                            device=device,
                            use_tqdm=False,
                            collect_predictions=True,
                            collect_probabilities=False,
                            desc=f"eval_{split_name}",
                        )
                        artifact_paths: Dict[str, str] = {}
                        preds = extras.get("preds")
                        targets = extras.get("targets")
                        if preds is not None and targets is not None:
                            artifact_paths = _save_classification_artifacts(
                                out_dir=out_dir,
                                split_name=split_name,
                                preds=preds,
                                targets=targets,
                            )
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "model_name": model_name,
                                "split": split_name,
                                "weights_dir_name": weights_dir_name,
                                "checkpoint_path": str(checkpoint_path),
                                "loss": float(metrics.get("loss", 0.0)),
                                "accuracy": float(metrics.get("accuracy", 0.0)),
                                "f1_macro": float(metrics.get("f1_macro", 0.0)),
                                "f1_weighted": float(metrics.get("f1_weighted", 0.0)),
                                "classification_report_json": artifact_paths.get("classification_report_json"),
                                "confusion_matrix_counts_json": artifact_paths.get("confusion_matrix_counts_json"),
                                "confusion_matrix_row_normalized_json": artifact_paths.get("confusion_matrix_row_normalized_json"),
                                **model_profile,
                            }
                        )
                    save_results_rows(out_dir, rows, base_name="summary_rows")
                    save_run_meta(
                        out_dir,
                        {
                            "domain": "ntu",
                            "dataset": dataset_name,
                            "model": model_name,
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_load": load_info,
                            "weights_dir_name": weights_dir_name,
                            "strict_load": strict_load,
                            "selection": selection,
                            "model_profile": model_profile,
                            "timestamp": ts,
                            "config_name": cfg_name,
                        },
                        filename="eval_meta.json",
                    )
                    print(f"[RUN] Completed eval: {out_dir}")
                except BaseException as exc:
                    print(f"[ERROR] NTU eval failed for {dataset_name}/{model_name} ({weights_dir_name}): {exc}")
                    traceback.print_exc()
                    raise


def _run_csi_bench_eval(
    *,
    cfg: Mapping[str, Any],
    config_path: Optional[str],
    experiment_name: str,
    model_names: Sequence[str],
    device: torch.device,
    weights_cfg: Mapping[str, Any],
) -> None:
    csi_cfg = dict(cfg.get("csi_bench", {}) or {})
    weights_csi = dict(weights_cfg.get("csi_bench", {}) or {})

    dataset_root = str(csi_cfg.get("dataset_root", cfg.get("common", {}).get("dataset_root", "datasets/CSI-Bench")))
    task_names = parse_token_list(
        csi_cfg.get("task_names")
        or csi_cfg.get("tasks")
        or csi_cfg.get("task_name")
        or "HumanActivityRecognition,HumanIdentification"
    )
    task_names = [normalize_csi_bench_task_name(name) for name in task_names]
    if not task_names:
        task_names = ["HumanActivityRecognition", "HumanIdentification"]
    batch_size = int(csi_cfg.get("batch_size", 256))
    num_workers_raw = csi_cfg.get("num_workers")
    num_workers = int(num_workers_raw) if num_workers_raw is not None else None
    pin_memory_raw = csi_cfg.get("pin_memory")
    pin_memory = None if pin_memory_raw is None else parse_bool(pin_memory_raw, default=(device.type == "cuda"))
    persistent_workers_raw = csi_cfg.get("persistent_workers")
    persistent_workers = None if persistent_workers_raw is None else parse_bool(persistent_workers_raw, default=True)
    prefetch_factor_raw = csi_cfg.get("prefetch_factor")
    prefetch_factor = int(prefetch_factor_raw) if prefetch_factor_raw is not None else None
    normalisation = str(csi_cfg.get("normalisation", "zscore"))
    splits_dir_name = str(csi_cfg.get("splits_dir", "splits"))
    target_subcarriers = int(csi_cfg.get("target_subcarriers", DEFAULT_CSI_BENCH_SUBCARRIERS))
    target_timesteps = int(csi_cfg.get("target_timesteps", DEFAULT_CSI_BENCH_TIMESTEPS))
    eval_split_names = parse_splits(
        csi_cfg.get("eval_split_names") or csi_cfg.get("eval_splits") or csi_cfg.get("split_names"),
        default=["test_id"],
    )

    strict_load = parse_bool(weights_cfg.get("strict_load"), default=True)
    selection = _normalise_selection_mode(weights_cfg.get("selection"))
    checkpoint_glob = weights_csi.get("checkpoint_glob")
    checkpoint_paths = weights_csi.get("checkpoint_paths")
    weights_dataset_root = str(weights_csi.get("dataset_root", dataset_root))
    weights_dir_names = _as_weight_dir_list(
        weights_csi.get("weights_dir_names", weights_csi.get("weights_dir_name")),
        default=str(weights_cfg.get("weights_dir_name") or DEFAULT_CLEAN_WEIGHTS_DIR),
    )

    criterion = nn.CrossEntropyLoss()
    for task_name in task_names:
        for model_name in model_names:
            for weights_dir_name in weights_dir_names:
                model_token = sanitize_token(model_name)
                weights_token = sanitize_token(weights_dir_name)
                out_dir, ts, cfg_name = prepare_results_dir(
                    experiment_name=experiment_name,
                    cfg_path=config_path,
                    relative_parts=["csi_bench", task_name, weights_token, model_token],
                )
                try:
                    save_run_meta(
                        out_dir,
                        {
                            "domain": "csi_bench",
                            "task_name": task_name,
                            "model": model_name,
                            "eval_split_names": eval_split_names,
                            "weights_dir_name": weights_dir_name,
                            "strict_load": strict_load,
                            "selection": selection,
                        },
                    )
                    save_config(out_dir, dict(cfg))
                    print_training_start(
                        f"Clean eval CSI-Bench ({task_name}/{model_name})",
                        {
                            "output_dir": out_dir,
                            "task_name": task_name,
                            "model": model_name,
                            "splits_dir": splits_dir_name,
                            "eval_split_names": eval_split_names,
                            "weights_dir_name": weights_dir_name,
                            "device": str(device),
                        },
                    )

                    canonical_splits = [_normalise_csi_split_name(split) for split in eval_split_names]
                    bundle = load_csi_bench_supervised(
                        dataset_root=dataset_root,
                        task_name=task_name,
                        batch_size=batch_size,
                        normalisation=normalisation,
                        target_subcarriers=target_subcarriers,
                        target_timesteps=target_timesteps,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        persistent_workers=persistent_workers,
                        prefetch_factor=prefetch_factor,
                        only_splits=canonical_splits,
                        splits_dir_name=splits_dir_name,
                        collate_fn=filter_none_collate,
                    )
                    if bundle is False or bundle is None:
                        raise RuntimeError(f"Failed to load CSI-Bench task {task_name}")

                    loaders = bundle["loaders"]
                    eval_loaders: List[Tuple[str, Any]] = []
                    for requested in eval_split_names:
                        loader_key = _loader_key_for_csi_split(requested)
                        loader = loaders.get(loader_key)
                        if loader is not None:
                            eval_loaders.append((loader_key, loader))
                    if not eval_loaders:
                        raise RuntimeError(
                            f"No requested CSI-Bench eval splits available for {task_name}. "
                            f"Requested={eval_split_names} Available={sorted(loaders.keys())}"
                        )

                    num_classes = int(bundle["num_classes"])
                    shape = infer_input_shape_from_loader(eval_loaders[0][1])
                    arch_metadata: Dict[str, Any] = {"num_classes": num_classes}
                    if shape is not None:
                        channels, freq_bins, time_steps = shape
                        arch_metadata.update(
                            {
                                "channels": channels,
                                "freq_bins": freq_bins,
                                "time_steps": time_steps,
                                "input_shape": [channels, freq_bins, time_steps],
                            }
                        )

                    base_dir = _resolve_csi_bench_checkpoint_dir(
                        weights_dataset_root=weights_dataset_root,
                        weights_dir_name=weights_dir_name,
                        task_name=task_name,
                    )
                    explicit_checkpoint = _nested_checkpoint_override(checkpoint_paths, task_name, model_name)
                    explicit_checkpoint = ensure_optuna_clean_checkpoint(
                        domain="csi_bench",
                        model_name=model_name,
                        weights_dir_name=weights_dir_name,
                        weights_cfg=weights_cfg,
                        canonical_base_dir=base_dir,
                        alias_stems=[f"CSI-Bench_{model_name}"],
                        task_name=task_name,
                    ) or explicit_checkpoint
                    checkpoint_path = _resolve_checkpoint(
                        base_dir=base_dir,
                        model_name=model_name,
                        selection=selection,
                        alias_stems=[f"CSI-Bench_{model_name}"],
                        checkpoint_glob=str(checkpoint_glob) if checkpoint_glob else None,
                        explicit_checkpoint=explicit_checkpoint,
                    )
                    checkpoint_meta = load_arch_metadata(checkpoint_path)
                    resolved_arch_metadata = _merge_arch_metadata(
                        checkpoint_meta,
                        dict(arch_metadata, task_name=task_name),
                    )
                    if shape is None:
                        inferred = infer_input_shape_from_loader(eval_loaders[0][1])
                        if inferred is None:
                            raise RuntimeError(f"Unable to infer input shape for {task_name}/{model_name}.")
                        channels, freq_bins, time_steps = inferred
                    spec = DatasetSpec("CSI-Bench", channels=channels, freq_bins=freq_bins, time_steps=time_steps)
                    model, _ = build_model_for_checkpoint(
                        domain="csi_bench",
                        model_name=model_name,
                        weights_dir_name=weights_dir_name,
                        weights_cfg=weights_cfg,
                        spec=spec,
                        dataset_key_for_model="csi-bench",
                        num_classes=num_classes,
                        arch_metadata=resolved_arch_metadata,
                        task_name=task_name,
                    )
                    model = model.to(device)
                    load_info = _load_checkpoint_into_model(
                        model=model,
                        checkpoint_path=str(checkpoint_path),
                        strict=strict_load,
                    )
                    model_profile = _collect_model_profile(
                        model=model,
                        loader=eval_loaders[0][1],
                        device=device,
                        cfg=cfg,
                        checkpoint_path=checkpoint_path,
                    )

                    rows: List[Dict[str, Any]] = []
                    for split_name, loader in eval_loaders:
                        metrics, extras = evaluate_classifier(
                            model=model,
                            loader=loader,
                            criterion=criterion,
                            device=device,
                            use_tqdm=False,
                            collect_predictions=True,
                            collect_probabilities=False,
                            desc=f"eval_{split_name}",
                        )
                        artifact_paths: Dict[str, str] = {}
                        preds = extras.get("preds")
                        targets = extras.get("targets")
                        if preds is not None and targets is not None:
                            artifact_paths = _save_classification_artifacts(
                                out_dir=out_dir,
                                split_name=split_name,
                                preds=preds,
                                targets=targets,
                            )
                        rows.append(
                            {
                                "dataset": "CSI-Bench",
                                "task_name": task_name,
                                "model_name": model_name,
                                "split": split_name,
                                "weights_dir_name": weights_dir_name,
                                "checkpoint_path": str(checkpoint_path),
                                "loss": float(metrics.get("loss", 0.0)),
                                "accuracy": float(metrics.get("accuracy", 0.0)),
                                "f1_macro": float(metrics.get("f1_macro", 0.0)),
                                "f1_weighted": float(metrics.get("f1_weighted", 0.0)),
                                "classification_report_json": artifact_paths.get("classification_report_json"),
                                "confusion_matrix_counts_json": artifact_paths.get("confusion_matrix_counts_json"),
                                "confusion_matrix_row_normalized_json": artifact_paths.get("confusion_matrix_row_normalized_json"),
                                **model_profile,
                            }
                        )
                    save_results_rows(out_dir, rows, base_name="summary_rows")
                    save_run_meta(
                        out_dir,
                        {
                            "domain": "csi_bench",
                            "task_name": task_name,
                            "model": model_name,
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_load": load_info,
                            "weights_dir_name": weights_dir_name,
                            "strict_load": strict_load,
                            "selection": selection,
                            "splits_dir_name": splits_dir_name,
                            "eval_split_names": eval_split_names,
                            "model_profile": model_profile,
                            "task_metadata": collect_csi_bench_task_metadata(
                                dataset_root=dataset_root,
                                task_name=task_name,
                                splits_dir_name=splits_dir_name,
                                split_names=canonical_splits,
                            ),
                            "timestamp": ts,
                            "config_name": cfg_name,
                        },
                        filename="eval_meta.json",
                    )
                    print(f"[RUN] Completed eval: {out_dir}")
                except BaseException as exc:
                    print(f"[ERROR] CSI-Bench eval failed for {task_name}/{model_name} ({weights_dir_name}): {exc}")
                    traceback.print_exc()
                    raise


def main(config_path: Optional[str] = None) -> None:
    cfg = load_config(config_path)
    configure_from_env_and_config(cfg)
    configure_torch_performance(cfg)

    common_cfg = cfg.get("common", {})
    experiment_name = str(common_cfg.get("experiment_name", "clean-model-eval"))
    model_cfg = cfg.get("model", {})
    model_names = resolve_supervised_models(
        as_list(
            model_cfg.get("architectures") or model_cfg.get("models"),
            default=["MLP", "ResNet18", "LeNet", "BiLSTM", "CSITimeSformer"],
        )
    )
    data_cfg = cfg.get("data", {})
    device = _resolve_device(data_cfg.get("device"))

    run_cfg = cfg.get("run", {})
    targets_raw = as_list(run_cfg.get("targets"), default=["ntu", "csi_bench"])
    targets = {token.strip().lower().replace("-", "_") for token in targets_raw}
    weights_cfg = dict(cfg.get("weights", {}) or {})

    if "ntu" in targets:
        _run_ntu_eval(
            cfg=cfg,
            config_path=config_path,
            experiment_name=experiment_name,
            model_names=model_names,
            device=device,
            weights_cfg=weights_cfg,
        )
    if "csi_bench" in targets:
        _run_csi_bench_eval(
            cfg=cfg,
            config_path=config_path,
            experiment_name=experiment_name,
            model_names=model_names,
            device=device,
            weights_cfg=weights_cfg,
        )


if __name__ == "__main__":
    cfg_path = os.environ.get("EXP_TRAIN_CONFIG")
    if cfg_path is None and len(sys.argv) > 1:
        cfg_path = resolve_config_path_from_arg(sys.argv[1])
    main(cfg_path)
