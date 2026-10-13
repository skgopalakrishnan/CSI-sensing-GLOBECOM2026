"""Two-stage supervised transfer runs (HAR <-> HID) for NTU and CSI-Bench."""

from __future__ import annotations

import csv
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from csi.data import get_experiment_loaders
from csi.data.utils import filter_none_collate, load_csi_bench_supervised
from csi.models.model_factory import build_model
from csi.models.supervised import train
from csi.models.utils.experiment import load_config, resolve_config_path_from_arg, sanitize_token
from csi.models.utils.gpu_utils import _resolve_device
from csi.models.utils.results import (
    prepare_results_dir,
    save_config,
    save_results_rows,
    save_run_meta,
)
from csi.models.utils.seeding import configure_from_env_and_config
from ..common.shared_utils import (
    collect_csi_bench_task_metadata,
    history_rows_supervised,
    infer_input_shape_from_loader,
    move_incomplete_run,
    normalize_csi_bench_task_name,
    parse_bool,
    parse_splits,
    parse_token_list,
    print_training_start,
    save_training_metadata,
)
from ..common.train_models_utils import (
    DEFAULT_CSI_BENCH_SUBCARRIERS,
    DEFAULT_CSI_BENCH_TIMESTEPS,
    as_list,
    configure_torch_performance,
    evaluate_splits,
    infer_ntu_shape,
    resolve_ntu_num_classes,
    resolve_project_path,
    resolve_supervised_models,
    save_model_weights,
    write_json_file,
)


def _extract_state_dict_from_payload(payload: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint payload must be a mapping, got {type(payload)!r}")
    if "model_state_dict" in payload and isinstance(payload["model_state_dict"], Mapping):
        return payload["model_state_dict"]  # type: ignore[return-value]
    if "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
        return payload["state_dict"]  # type: ignore[return-value]
    if payload and all(torch.is_tensor(v) for v in payload.values()):
        return payload  # type: ignore[return-value]
    raise ValueError("Unsupported checkpoint format (expected state_dict or model_state_dict).")


def _load_state_dict_file(path: str) -> Mapping[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return _extract_state_dict_from_payload(payload)


def _load_compatible_weights(
    model: nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    target_state = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []

    for key, value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            skipped.append(key)
            continue
        if tuple(target_value.shape) != tuple(value.shape):
            skipped.append(key)
            continue
        compatible[key] = value

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "loaded_keys": len(compatible),
        "skipped_keys": len(skipped),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "missing_key_names": list(missing),
        "unexpected_key_names": list(unexpected),
    }


def _is_classifier_head_param(name: str) -> bool:
    return (
        name.startswith("head.")
        or name.startswith("fc.")
        or name.startswith("model.fc.")
    )


def _apply_finetune_mode(model: nn.Module, mode: str) -> Dict[str, int]:
    key = str(mode).strip().lower()
    if key in {"full", "all", "full_finetune"}:
        for _, param in model.named_parameters():
            param.requires_grad = True
    elif key in {"linear_probe", "head_only", "frozen_backbone"}:
        any_trainable = False
        for name, param in model.named_parameters():
            is_head = _is_classifier_head_param(name)
            param.requires_grad = bool(is_head)
            any_trainable = any_trainable or bool(is_head)
        if not any_trainable:
            raise RuntimeError(
                "linear_probe mode found no head parameters. "
                "Expected names prefixed by one of: head., fc., model.fc."
            )
    else:
        raise ValueError(f"Unsupported transfer finetune mode '{mode}'.")

    total = 0
    trainable = 0
    for _, param in model.named_parameters():
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
    return {"trainable_params": trainable, "total_params": total}


def _stage_train_cfg(
    *,
    train_cfg: Mapping[str, Any],
    transfer_cfg: Mapping[str, Any],
    stage: str,
) -> Dict[str, Any]:
    stage_cfg = dict(transfer_cfg.get(stage, {}) or {})
    prefix = f"{stage}_"

    def _pick(name: str, default: Any) -> Any:
        if name in stage_cfg:
            return stage_cfg[name]
        prefixed = f"{prefix}{name}"
        if prefixed in transfer_cfg:
            return transfer_cfg[prefixed]
        return train_cfg.get(name, default)

    patience_raw = _pick("patience", train_cfg.get("patience"))
    grad_clip_raw = _pick("grad_clip", train_cfg.get("grad_clip"))
    log_every_raw = _pick("log_every_epochs", train_cfg.get("log_every_epochs"))

    return {
        "learning_rate": float(_pick("learning_rate", 1e-3)),
        "epochs": int(_pick("epochs", 100)),
        "progress_bar": parse_bool(_pick("progress_bar", False), default=False),
        "patience": int(patience_raw) if patience_raw is not None else None,
        "min_epochs": int(_pick("min_epochs", 1)),
        "monitor_metric": str(_pick("monitor_metric", "accuracy")),
        "min_delta": float(_pick("min_delta", 0.0)),
        "weight_decay": float(_pick("weight_decay", 0.0)),
        "grad_clip": float(grad_clip_raw) if grad_clip_raw is not None else None,
        "optimizer_cfg": _pick("optimizer", None),
        "scheduler_cfg": _pick("scheduler", None),
        "amp": _pick("amp", None),
        "amp_dtype": str(_pick("amp_dtype", "float16")),
        "log_every_epochs": int(log_every_raw) if log_every_raw is not None else None,
    }


def _run_training_stage(
    *,
    model: nn.Module,
    criterion: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    stage_label: str,
    cfg: Mapping[str, Any],
) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary = train(
        model=model,
        tensor_loader=train_loader,
        num_epochs=int(cfg["epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        criterion=criterion,
        device=device,
        use_tqdm=bool(cfg["progress_bar"]),
        val_loader=val_loader,
        patience=cfg["patience"] if val_loader is not None else None,
        min_epochs=int(cfg["min_epochs"]),
        monitor_metric=str(cfg["monitor_metric"]),
        min_delta=float(cfg["min_delta"]),
        weight_decay=float(cfg["weight_decay"]),
        grad_clip=cfg["grad_clip"],
        optimizer_cfg=cfg["optimizer_cfg"],
        scheduler_cfg=cfg["scheduler_cfg"],
        amp=cfg["amp"],
        amp_dtype=str(cfg["amp_dtype"]),
        log_every_epochs=cfg["log_every_epochs"],
    )

    history_rows = history_rows_supervised(summary.history)
    for row in history_rows:
        row["stage"] = stage_label

    eval_rows = evaluate_splits(
        model=model,
        criterion=criterion,
        device=device,
        use_tqdm=bool(cfg["progress_bar"]),
        split_loaders={"val": val_loader, "test": test_loader},
        stage=stage_label,
    )
    return summary, history_rows, eval_rows


def _export_ntu_checkpoint(
    *,
    datasets_root: str,
    weights_dir_name: str,
    dataset_name: str,
    model_name: str,
    transfer_tag: str,
    source_checkpoint: str,
    source_out_dir: str,
    config_path: Optional[str],
    run_timestamp: Optional[str],
) -> str:
    root = resolve_project_path(datasets_root)
    export_dir = root / dataset_name / weights_dir_name
    export_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_name}_{model_name}_{transfer_tag}"
    ts_token = sanitize_token(run_timestamp or "") or sanitize_token(Path(source_out_dir).name) or "run"
    export_path = export_dir / f"{stem}__{ts_token}.pt"
    latest_path = export_dir / f"{stem}.pt"

    shutil.copy2(source_checkpoint, export_path)
    shutil.copy2(source_checkpoint, latest_path)
    write_json_file(
        Path(f"{export_path}.meta.json"),
        {
            "stage": "sl_trial_transfer_export",
            "domain": "ntu",
            "dataset": dataset_name,
            "model": model_name,
            "transfer_tag": transfer_tag,
            "variant": "timestamped",
            "run_timestamp": run_timestamp,
            "latest_path": str(latest_path),
            "source_checkpoint": str(Path(source_checkpoint).resolve()),
            "source_run_dir": str(Path(source_out_dir).resolve()),
            "config_path": str(Path(config_path).resolve()) if config_path else None,
        },
    )
    write_json_file(
        Path(f"{latest_path}.meta.json"),
        {
            "stage": "sl_trial_transfer_export",
            "domain": "ntu",
            "dataset": dataset_name,
            "model": model_name,
            "transfer_tag": transfer_tag,
            "variant": "latest_alias",
            "run_timestamp": run_timestamp,
            "timestamped_path": str(export_path),
            "source_checkpoint": str(Path(source_checkpoint).resolve()),
            "source_run_dir": str(Path(source_out_dir).resolve()),
            "config_path": str(Path(config_path).resolve()) if config_path else None,
        },
    )
    return str(export_path)


def _export_csi_bench_checkpoint(
    *,
    datasets_root: str,
    weights_dir_name: str,
    task_name: str,
    model_name: str,
    transfer_tag: str,
    source_checkpoint: str,
    source_out_dir: str,
    config_path: Optional[str],
    run_timestamp: Optional[str],
) -> str:
    root = resolve_project_path(datasets_root)
    csi_root = root if root.name == "CSI-Bench" else (root / "CSI-Bench")
    export_dir = csi_root / weights_dir_name / task_name
    export_dir.mkdir(parents=True, exist_ok=True)
    stem = f"CSI-Bench_{model_name}_{transfer_tag}"
    ts_token = sanitize_token(run_timestamp or "") or sanitize_token(Path(source_out_dir).name) or "run"
    export_path = export_dir / f"{stem}__{ts_token}.pt"
    latest_path = export_dir / f"{stem}.pt"

    shutil.copy2(source_checkpoint, export_path)
    shutil.copy2(source_checkpoint, latest_path)
    write_json_file(
        Path(f"{export_path}.meta.json"),
        {
            "stage": "sl_trial_transfer_export",
            "domain": "csi_bench",
            "dataset": "CSI-Bench",
            "task_name": task_name,
            "model": model_name,
            "transfer_tag": transfer_tag,
            "variant": "timestamped",
            "run_timestamp": run_timestamp,
            "latest_path": str(latest_path),
            "source_checkpoint": str(Path(source_checkpoint).resolve()),
            "source_run_dir": str(Path(source_out_dir).resolve()),
            "config_path": str(Path(config_path).resolve()) if config_path else None,
        },
    )
    write_json_file(
        Path(f"{latest_path}.meta.json"),
        {
            "stage": "sl_trial_transfer_export",
            "domain": "csi_bench",
            "dataset": "CSI-Bench",
            "task_name": task_name,
            "model": model_name,
            "transfer_tag": transfer_tag,
            "variant": "latest_alias",
            "run_timestamp": run_timestamp,
            "timestamped_path": str(export_path),
            "source_checkpoint": str(Path(source_checkpoint).resolve()),
            "source_run_dir": str(Path(source_out_dir).resolve()),
            "config_path": str(Path(config_path).resolve()) if config_path else None,
        },
    )
    return str(export_path)


def _safe_csv_column_values(path: Path, column: str) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get(column)
            if value is not None:
                out.add(str(value))
    return out


def _csi_bench_har_hid_overlap(dataset_root: str) -> Dict[str, Any]:
    base = resolve_project_path(dataset_root)
    har_meta = base / "HumanActivityRecognition" / "metadata" / "sample_metadata.csv"
    hid_meta = base / "HumanIdentification" / "metadata" / "sample_metadata.csv"
    har_paths = _safe_csv_column_values(har_meta, "file_path")
    hid_paths = _safe_csv_column_values(hid_meta, "file_path")
    return {
        "har_metadata": str(har_meta),
        "hid_metadata": str(hid_meta),
        "har_file_paths": len(har_paths),
        "hid_file_paths": len(hid_paths),
        "intersection": len(har_paths & hid_paths),
        "har_only": len(har_paths - hid_paths),
        "hid_only": len(hid_paths - har_paths),
    }


def _ntu_har_hid_overlap(root: str) -> Dict[str, Any]:
    base = resolve_project_path(root)
    har_root = base / "NTU-Fi_HAR"
    hid_root = base / "NTU-Fi-HumanID"

    out: Dict[str, Any] = {"root": str(base)}
    for split in ("train_amp", "test_amp"):
        har_split = har_root / split
        hid_split = hid_root / split
        har_rel = {p.relative_to(har_split).as_posix() for p in har_split.rglob("*.mat")} if har_split.exists() else set()
        hid_rel = {p.relative_to(hid_split).as_posix() for p in hid_split.rglob("*.mat")} if hid_split.exists() else set()
        out[split] = {
            "har_count": len(har_rel),
            "hid_count": len(hid_rel),
            "intersection_relpath": len(har_rel & hid_rel),
            "har_only_relpath": len(har_rel - hid_rel),
            "hid_only_relpath": len(hid_rel - har_rel),
        }
    return out


def _resolve_transfer_directions(transfer_cfg: Mapping[str, Any]) -> set[str]:
    directions = parse_token_list(transfer_cfg.get("directions") or transfer_cfg.get("direction"))
    if not directions:
        return {"har_to_hid", "hid_to_har"}
    normalised: set[str] = set()
    for item in directions:
        key = item.strip().lower().replace("-", "_")
        if key in {"all", "both"}:
            return {"har_to_hid", "hid_to_har"}
        normalised.add(key)
    return normalised


def _resolve_ntu_pairs(transfer_cfg: Mapping[str, Any]) -> List[Dict[str, str]]:
    directions = _resolve_transfer_directions(transfer_cfg)
    defaults = {
        "har_to_hid": {
            "name": "har_to_hid",
            "source_dataset": "NTU-Fi_HAR",
            "target_dataset": "NTU-Fi-HumanID",
        },
        "hid_to_har": {
            "name": "hid_to_har",
            "source_dataset": "NTU-Fi-HumanID",
            "target_dataset": "NTU-Fi_HAR",
        },
    }

    cfg_pairs = transfer_cfg.get("ntu_pairs")
    if isinstance(cfg_pairs, list) and cfg_pairs:
        out: List[Dict[str, str]] = []
        for idx, raw in enumerate(cfg_pairs):
            if not isinstance(raw, Mapping):
                continue
            name = str(raw.get("name", f"pair_{idx}")).strip()
            source = str(raw.get("source_dataset", raw.get("source", ""))).strip()
            target = str(raw.get("target_dataset", raw.get("target", ""))).strip()
            if not name or not source or not target:
                continue
            if directions and name.lower().replace("-", "_") not in directions:
                continue
            out.append({"name": name, "source_dataset": source, "target_dataset": target})
        if out:
            return out

    return [value for key, value in defaults.items() if key in directions]


def _resolve_csi_pairs(transfer_cfg: Mapping[str, Any]) -> List[Dict[str, str]]:
    directions = _resolve_transfer_directions(transfer_cfg)
    defaults = {
        "har_to_hid": {
            "name": "har_to_hid",
            "source_task": "HumanActivityRecognition",
            "target_task": "HumanIdentification",
        },
        "hid_to_har": {
            "name": "hid_to_har",
            "source_task": "HumanIdentification",
            "target_task": "HumanActivityRecognition",
        },
    }

    cfg_pairs = transfer_cfg.get("csi_pairs")
    if isinstance(cfg_pairs, list) and cfg_pairs:
        out: List[Dict[str, str]] = []
        for idx, raw in enumerate(cfg_pairs):
            if not isinstance(raw, Mapping):
                continue
            name = str(raw.get("name", f"pair_{idx}")).strip()
            source = normalize_csi_bench_task_name(str(raw.get("source_task", raw.get("source", ""))).strip())
            target = normalize_csi_bench_task_name(str(raw.get("target_task", raw.get("target", ""))).strip())
            if not name or not source or not target:
                continue
            if directions and name.lower().replace("-", "_") not in directions:
                continue
            out.append({"name": name, "source_task": source, "target_task": target})
        if out:
            return out

    return [value for key, value in defaults.items() if key in directions]


def _run_ntu_transfer(
    *,
    cfg: Mapping[str, Any],
    config_path: Optional[str],
    experiment_name: str,
    model_names: Sequence[str],
    save_weights: bool,
    dataset_weights_dir_name: str,
    train_cfg: Mapping[str, Any],
    transfer_cfg: Mapping[str, Any],
    device: torch.device,
    data_seed: int,
) -> None:
    ntu_cfg = dict(cfg.get("ntu", {}) or {})
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
    ntu_seed = int(ntu_cfg.get("seed", data_seed))

    finetune_mode = str(transfer_cfg.get("finetune_mode", "full"))
    base_model_mode = str(transfer_cfg.get("base_model_mode", "source_supervised")).strip().lower()
    source_checkpoint = transfer_cfg.get("source_checkpoint")

    source_stage_cfg = _stage_train_cfg(train_cfg=train_cfg, transfer_cfg=transfer_cfg, stage="source")
    target_stage_cfg = _stage_train_cfg(train_cfg=train_cfg, transfer_cfg=transfer_cfg, stage="target")
    pairs = _resolve_ntu_pairs(transfer_cfg)
    criterion = nn.CrossEntropyLoss()
    overlap_info = _ntu_har_hid_overlap(root)

    if not pairs:
        raise ValueError("No NTU transfer pairs resolved from transfer.directions / transfer.ntu_pairs.")

    for pair in pairs:
        source_dataset = pair["source_dataset"]
        target_dataset = pair["target_dataset"]
        pair_name = pair["name"]
        transfer_tag = sanitize_token(f"{source_dataset}_to_{target_dataset}")

        for model_name in model_names:
            model_token = sanitize_token(model_name)
            out_dir, ts, cfg_name = prepare_results_dir(
                experiment_name=experiment_name,
                cfg_path=config_path,
                relative_parts=["ntu", pair_name, model_token],
            )
            completed = False
            try:
                save_run_meta(
                    out_dir,
                    {
                        "domain": "ntu",
                        "transfer_pair": pair_name,
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "model": model_name,
                        "timestamp": ts,
                        "config_name": cfg_name,
                        "root": root,
                        "timesteps": timesteps,
                        "batch_sizes": batch_sizes,
                        "num_workers": num_workers,
                        "pin_memory": pin_memory,
                        "persistent_workers": persistent_workers,
                        "prefetch_factor": prefetch_factor,
                        "device": str(device),
                        "base_model_mode": base_model_mode,
                        "finetune_mode": finetune_mode,
                    },
                )
                save_config(out_dir, dict(cfg))
                print_training_start(
                    f"NTU transfer ({pair_name}/{model_name})",
                    {
                        "output_dir": out_dir,
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "model": model_name,
                        "device": str(device),
                        "base_model_mode": base_model_mode,
                        "finetune_mode": finetune_mode,
                        "source_epochs": source_stage_cfg["epochs"],
                        "target_epochs": target_stage_cfg["epochs"],
                    },
                )

                source_bundle = get_experiment_loaders(
                    source_dataset,
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
                    seed=ntu_seed,
                )
                target_bundle = get_experiment_loaders(
                    target_dataset,
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
                    seed=ntu_seed,
                )

                source_train = source_bundle.train_loader
                source_val = source_bundle.val_loader
                source_test = source_bundle.test_loader
                target_train = target_bundle.train_loader
                target_val = target_bundle.val_loader
                target_test = target_bundle.test_loader

                if source_train is None or target_train is None:
                    raise RuntimeError("Missing train loader in transfer setup.")

                source_info = source_bundle.info or {}
                target_info = target_bundle.info or {}
                source_shape = infer_ntu_shape(source_info, source_train)
                target_shape = infer_ntu_shape(target_info, target_train)
                source_classes = resolve_ntu_num_classes(source_info, source_train)
                target_classes = resolve_ntu_num_classes(target_info, target_train)

                source_metadata: Dict[str, Any] = {"num_classes": source_classes}
                if source_shape is not None:
                    c, f, t = source_shape
                    source_metadata.update({"channels": c, "freq_bins": f, "time_steps": t, "input_shape": [c, f, t]})
                target_metadata: Dict[str, Any] = {"num_classes": target_classes}
                if target_shape is not None:
                    c, f, t = target_shape
                    target_metadata.update({"channels": c, "freq_bins": f, "time_steps": t, "input_shape": [c, f, t]})

                source_state: Optional[Mapping[str, torch.Tensor]] = None
                source_checkpoint_path: Optional[str] = None
                source_load_stats: Dict[str, Any] = {}
                stage_history_rows: List[Dict[str, Any]] = []
                stage_eval_rows: List[Dict[str, Any]] = []

                if base_model_mode in {"source_supervised", "pretrain_source"}:
                    source_model, _ = build_model(
                        source_dataset,
                        model_name,
                        num_classes=source_classes,
                        arch_metadata=source_metadata,
                    )
                    source_model = source_model.to(device)
                    _, source_hist, source_eval = _run_training_stage(
                        model=source_model,
                        criterion=criterion,
                        train_loader=source_train,
                        val_loader=source_val,
                        test_loader=source_test,
                        device=device,
                        stage_label="source_pretrain",
                        cfg=source_stage_cfg,
                    )
                    stage_history_rows.extend(source_hist)
                    stage_eval_rows.extend(source_eval)
                    source_checkpoint_path = save_model_weights(
                        out_dir=out_dir,
                        filename_stem=f"{model_name}_source_pretrain",
                        model=source_model,
                    )
                    source_state = {k: v.detach().cpu() for k, v in source_model.state_dict().items()}
                    del source_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                elif base_model_mode == "source_checkpoint":
                    if not source_checkpoint:
                        raise ValueError("transfer.source_checkpoint must be set when base_model_mode=source_checkpoint.")
                    resolved = str(resolve_project_path(str(source_checkpoint)))
                    source_state = _load_state_dict_file(resolved)
                    source_checkpoint_path = resolved
                elif base_model_mode in {"scratch", "none"}:
                    source_state = None
                else:
                    raise ValueError(
                        f"Unsupported transfer.base_model_mode '{base_model_mode}'. "
                        "Use source_supervised, source_checkpoint, or scratch."
                    )

                target_model, _ = build_model(
                    target_dataset,
                    model_name,
                    num_classes=target_classes,
                    arch_metadata=target_metadata,
                )
                target_model = target_model.to(device)

                if source_state is not None:
                    source_load_stats = _load_compatible_weights(target_model, source_state)
                freeze_stats = _apply_finetune_mode(target_model, finetune_mode)

                _, target_hist, target_eval = _run_training_stage(
                    model=target_model,
                    criterion=criterion,
                    train_loader=target_train,
                    val_loader=target_val,
                    test_loader=target_test,
                    device=device,
                    stage_label="target_finetune",
                    cfg=target_stage_cfg,
                )
                stage_history_rows.extend(target_hist)
                stage_eval_rows.extend(target_eval)

                save_results_rows(out_dir, stage_history_rows, base_name="train_history")
                save_results_rows(out_dir, stage_eval_rows, base_name="summary_rows")

                target_checkpoint = save_model_weights(
                    out_dir=out_dir,
                    filename_stem=f"{model_name}_target_finetune",
                    model=target_model,
                )
                exported_weights_path = None
                if save_weights:
                    exported_weights_path = _export_ntu_checkpoint(
                        datasets_root=root,
                        weights_dir_name=dataset_weights_dir_name,
                        dataset_name=target_dataset,
                        model_name=model_name,
                        transfer_tag=transfer_tag,
                        source_checkpoint=target_checkpoint,
                        source_out_dir=out_dir,
                        config_path=config_path,
                        run_timestamp=ts,
                    )

                save_training_metadata(
                    out_dir,
                    {
                        "stage": "sl_trial_transfer_ntu",
                        "transfer_pair": pair_name,
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "model_name": model_name,
                        "device": str(device),
                        "base_model_mode": base_model_mode,
                        "finetune_mode": finetune_mode,
                        "source_stage_cfg": source_stage_cfg,
                        "target_stage_cfg": target_stage_cfg,
                        "source_checkpoint_path": source_checkpoint_path,
                        "target_checkpoint_path": target_checkpoint,
                        "source_load_stats": source_load_stats,
                        "freeze_stats": freeze_stats,
                        "dataset_overlap_snapshot": overlap_info,
                        "dataset_export_weights_path": exported_weights_path,
                        "dataset_export_weights_dir_name": dataset_weights_dir_name,
                    },
                )

                completed = True
                print(f"[RUN] Completed transfer: {out_dir}")
                if exported_weights_path:
                    print(f"[INFO] Exported transfer checkpoint: {exported_weights_path}")

            except BaseException as exc:
                print(f"[ERROR] NTU transfer failed for {pair_name}/{model_name}: {exc}")
                traceback.print_exc()
                raise
            finally:
                if not completed:
                    target_dir = move_incomplete_run(out_dir)
                    if target_dir:
                        print(f"[INFO] Moved incomplete run to: {target_dir}")


def _run_csi_bench_transfer(
    *,
    cfg: Mapping[str, Any],
    config_path: Optional[str],
    experiment_name: str,
    model_names: Sequence[str],
    save_weights: bool,
    dataset_weights_dir_name: str,
    train_cfg: Mapping[str, Any],
    transfer_cfg: Mapping[str, Any],
    device: torch.device,
) -> None:
    csi_cfg = dict(cfg.get("csi_bench", {}) or {})
    dataset_root = str(csi_cfg.get("dataset_root", cfg.get("common", {}).get("dataset_root", "datasets/CSI-Bench")))
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
    only_splits = parse_splits(csi_cfg.get("only_splits") or csi_cfg.get("splits"), default=["train_id", "val_id", "test_id"])

    finetune_mode = str(transfer_cfg.get("finetune_mode", "full"))
    base_model_mode = str(transfer_cfg.get("base_model_mode", "source_supervised")).strip().lower()
    source_checkpoint = transfer_cfg.get("source_checkpoint")

    source_stage_cfg = _stage_train_cfg(train_cfg=train_cfg, transfer_cfg=transfer_cfg, stage="source")
    target_stage_cfg = _stage_train_cfg(train_cfg=train_cfg, transfer_cfg=transfer_cfg, stage="target")
    pairs = _resolve_csi_pairs(transfer_cfg)
    criterion = nn.CrossEntropyLoss()
    overlap_info = _csi_bench_har_hid_overlap(dataset_root)

    if not pairs:
        raise ValueError("No CSI-Bench transfer pairs resolved from transfer.directions / transfer.csi_pairs.")

    for pair in pairs:
        source_task = pair["source_task"]
        target_task = pair["target_task"]
        pair_name = pair["name"]
        transfer_tag = sanitize_token(f"{source_task}_to_{target_task}")

        for model_name in model_names:
            model_token = sanitize_token(model_name)
            out_dir, ts, cfg_name = prepare_results_dir(
                experiment_name=experiment_name,
                cfg_path=config_path,
                relative_parts=["csi_bench", pair_name, model_token],
            )
            completed = False
            try:
                save_run_meta(
                    out_dir,
                    {
                        "domain": "csi_bench",
                        "transfer_pair": pair_name,
                        "source_task": source_task,
                        "target_task": target_task,
                        "model": model_name,
                        "timestamp": ts,
                        "config_name": cfg_name,
                        "dataset_root": dataset_root,
                        "splits": only_splits,
                        "batch_size": batch_size,
                        "num_workers": num_workers,
                        "pin_memory": pin_memory,
                        "persistent_workers": persistent_workers,
                        "prefetch_factor": prefetch_factor,
                        "device": str(device),
                        "base_model_mode": base_model_mode,
                        "finetune_mode": finetune_mode,
                    },
                )
                save_config(out_dir, dict(cfg))
                print_training_start(
                    f"CSI-Bench transfer ({pair_name}/{model_name})",
                    {
                        "output_dir": out_dir,
                        "source_task": source_task,
                        "target_task": target_task,
                        "model": model_name,
                        "device": str(device),
                        "base_model_mode": base_model_mode,
                        "finetune_mode": finetune_mode,
                        "source_epochs": source_stage_cfg["epochs"],
                        "target_epochs": target_stage_cfg["epochs"],
                    },
                )

                source_bundle = load_csi_bench_supervised(
                    dataset_root=dataset_root,
                    task_name=source_task,
                    batch_size=batch_size,
                    normalisation=normalisation,
                    target_subcarriers=target_subcarriers,
                    target_timesteps=target_timesteps,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    prefetch_factor=prefetch_factor,
                    only_splits=only_splits,
                    splits_dir_name=splits_dir_name,
                    collate_fn=filter_none_collate,
                )
                target_bundle = load_csi_bench_supervised(
                    dataset_root=dataset_root,
                    task_name=target_task,
                    batch_size=batch_size,
                    normalisation=normalisation,
                    target_subcarriers=target_subcarriers,
                    target_timesteps=target_timesteps,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    prefetch_factor=prefetch_factor,
                    only_splits=only_splits,
                    splits_dir_name=splits_dir_name,
                    collate_fn=filter_none_collate,
                )
                if source_bundle is False or source_bundle is None:
                    raise RuntimeError(f"Failed to load CSI-Bench source task {source_task}")
                if target_bundle is False or target_bundle is None:
                    raise RuntimeError(f"Failed to load CSI-Bench target task {target_task}")

                source_loaders = source_bundle["loaders"]
                target_loaders = target_bundle["loaders"]
                source_train = source_loaders.get("train")
                source_val = source_loaders.get("val")
                source_test = source_loaders.get("test")
                target_train = target_loaders.get("train")
                target_val = target_loaders.get("val")
                target_test = target_loaders.get("test")
                if source_train is None or target_train is None:
                    raise RuntimeError("Missing train loader in CSI-Bench transfer setup.")

                source_classes = int(source_bundle["num_classes"])
                target_classes = int(target_bundle["num_classes"])
                source_shape = infer_input_shape_from_loader(source_train)
                target_shape = infer_input_shape_from_loader(target_train)

                source_metadata: Dict[str, Any] = {"num_classes": source_classes}
                if source_shape is not None:
                    c, f, t = source_shape
                    source_metadata.update({"channels": c, "freq_bins": f, "time_steps": t, "input_shape": [c, f, t]})
                target_metadata: Dict[str, Any] = {"num_classes": target_classes}
                if target_shape is not None:
                    c, f, t = target_shape
                    target_metadata.update({"channels": c, "freq_bins": f, "time_steps": t, "input_shape": [c, f, t]})

                source_state: Optional[Mapping[str, torch.Tensor]] = None
                source_checkpoint_path: Optional[str] = None
                source_load_stats: Dict[str, Any] = {}
                stage_history_rows: List[Dict[str, Any]] = []
                stage_eval_rows: List[Dict[str, Any]] = []

                if base_model_mode in {"source_supervised", "pretrain_source"}:
                    source_model, _ = build_model(
                        "csi-bench",
                        model_name,
                        num_classes=source_classes,
                        arch_metadata=source_metadata,
                    )
                    source_model = source_model.to(device)
                    _, source_hist, source_eval = _run_training_stage(
                        model=source_model,
                        criterion=criterion,
                        train_loader=source_train,
                        val_loader=source_val,
                        test_loader=source_test,
                        device=device,
                        stage_label="source_pretrain",
                        cfg=source_stage_cfg,
                    )
                    stage_history_rows.extend(source_hist)
                    stage_eval_rows.extend(source_eval)
                    source_checkpoint_path = save_model_weights(
                        out_dir=out_dir,
                        filename_stem=f"{model_name}_source_pretrain",
                        model=source_model,
                    )
                    source_state = {k: v.detach().cpu() for k, v in source_model.state_dict().items()}
                    del source_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                elif base_model_mode == "source_checkpoint":
                    if not source_checkpoint:
                        raise ValueError("transfer.source_checkpoint must be set when base_model_mode=source_checkpoint.")
                    resolved = str(resolve_project_path(str(source_checkpoint)))
                    source_state = _load_state_dict_file(resolved)
                    source_checkpoint_path = resolved
                elif base_model_mode in {"scratch", "none"}:
                    source_state = None
                else:
                    raise ValueError(
                        f"Unsupported transfer.base_model_mode '{base_model_mode}'. "
                        "Use source_supervised, source_checkpoint, or scratch."
                    )

                target_model, _ = build_model(
                    "csi-bench",
                    model_name,
                    num_classes=target_classes,
                    arch_metadata=target_metadata,
                )
                target_model = target_model.to(device)

                if source_state is not None:
                    source_load_stats = _load_compatible_weights(target_model, source_state)
                freeze_stats = _apply_finetune_mode(target_model, finetune_mode)

                _, target_hist, target_eval = _run_training_stage(
                    model=target_model,
                    criterion=criterion,
                    train_loader=target_train,
                    val_loader=target_val,
                    test_loader=target_test,
                    device=device,
                    stage_label="target_finetune",
                    cfg=target_stage_cfg,
                )
                stage_history_rows.extend(target_hist)
                stage_eval_rows.extend(target_eval)

                save_results_rows(out_dir, stage_history_rows, base_name="train_history")
                save_results_rows(out_dir, stage_eval_rows, base_name="summary_rows")

                target_checkpoint = save_model_weights(
                    out_dir=out_dir,
                    filename_stem=f"{model_name}_target_finetune",
                    model=target_model,
                )
                exported_weights_path = None
                if save_weights:
                    exported_weights_path = _export_csi_bench_checkpoint(
                        datasets_root=dataset_root,
                        weights_dir_name=dataset_weights_dir_name,
                        task_name=target_task,
                        model_name=model_name,
                        transfer_tag=transfer_tag,
                        source_checkpoint=target_checkpoint,
                        source_out_dir=out_dir,
                        config_path=config_path,
                        run_timestamp=ts,
                    )

                save_training_metadata(
                    out_dir,
                    {
                        "stage": "sl_trial_transfer_csi_bench",
                        "transfer_pair": pair_name,
                        "source_task": source_task,
                        "target_task": target_task,
                        "model_name": model_name,
                        "device": str(device),
                        "base_model_mode": base_model_mode,
                        "finetune_mode": finetune_mode,
                        "source_stage_cfg": source_stage_cfg,
                        "target_stage_cfg": target_stage_cfg,
                        "source_checkpoint_path": source_checkpoint_path,
                        "target_checkpoint_path": target_checkpoint,
                        "source_load_stats": source_load_stats,
                        "freeze_stats": freeze_stats,
                        "dataset_overlap_snapshot": overlap_info,
                        "source_task_metadata": collect_csi_bench_task_metadata(
                            dataset_root=dataset_root,
                            task_name=source_task,
                            splits_dir_name=splits_dir_name,
                            split_names=only_splits,
                        ),
                        "target_task_metadata": collect_csi_bench_task_metadata(
                            dataset_root=dataset_root,
                            task_name=target_task,
                            splits_dir_name=splits_dir_name,
                            split_names=only_splits,
                        ),
                        "dataset_export_weights_path": exported_weights_path,
                        "dataset_export_weights_dir_name": dataset_weights_dir_name,
                    },
                )

                completed = True
                print(f"[RUN] Completed transfer: {out_dir}")
                if exported_weights_path:
                    print(f"[INFO] Exported transfer checkpoint: {exported_weights_path}")

            except BaseException as exc:
                print(f"[ERROR] CSI-Bench transfer failed for {pair_name}/{model_name}: {exc}")
                traceback.print_exc()
                raise
            finally:
                if not completed:
                    target_dir = move_incomplete_run(out_dir)
                    if target_dir:
                        print(f"[INFO] Moved incomplete run to: {target_dir}")


def main(config_path: Optional[str] = None) -> None:
    cfg = load_config(config_path)
    configure_from_env_and_config(cfg)
    configure_torch_performance(cfg)

    common_cfg = cfg.get("common", {})
    experiment_name = str(common_cfg.get("experiment_name", "sl-transfer-trials"))
    model_cfg = cfg.get("model", {})
    model_names = resolve_supervised_models(
        as_list(
            model_cfg.get("architectures") or model_cfg.get("models"),
            default=["ResNet18"],
        )
    )
    save_weights = parse_bool(model_cfg.get("save_weights"), default=True)
    dataset_weights_dir_name = str(
        model_cfg.get("dataset_weights_dir_name")
        or model_cfg.get("weights_dir_name")
        or "model_params_sl_transfer_trials"
    )

    data_cfg = cfg.get("data", {})
    device = _resolve_device(data_cfg.get("device"))
    data_seed = int(data_cfg.get("seed", 42))

    run_cfg = cfg.get("run", {})
    targets_raw = as_list(run_cfg.get("targets"), default=["ntu", "csi_bench"])
    targets = {token.strip().lower().replace("-", "_") for token in targets_raw}

    train_cfg = dict(cfg.get("training", {}) or {})
    transfer_cfg = dict(cfg.get("transfer", {}) or {})

    if "ntu" in targets:
        _run_ntu_transfer(
            cfg=cfg,
            config_path=config_path,
            experiment_name=experiment_name,
            model_names=model_names,
            save_weights=save_weights,
            dataset_weights_dir_name=dataset_weights_dir_name,
            train_cfg=train_cfg,
            transfer_cfg=transfer_cfg,
            device=device,
            data_seed=data_seed,
        )
    if "csi_bench" in targets:
        _run_csi_bench_transfer(
            cfg=cfg,
            config_path=config_path,
            experiment_name=experiment_name,
            model_names=model_names,
            save_weights=save_weights,
            dataset_weights_dir_name=dataset_weights_dir_name,
            train_cfg=train_cfg,
            transfer_cfg=transfer_cfg,
            device=device,
        )


if __name__ == "__main__":
    cfg_path = os.environ.get("EXP_TRAIN_CONFIG")
    if cfg_path is None and len(sys.argv) > 1:
        cfg_path = resolve_config_path_from_arg(sys.argv[1])
    main(cfg_path)
