"""Transfer robustness benchmarking for APGD/APGDT on NTU and CSI-Bench."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from csi.attacks.evasion.attack_factory import generate_X_adv
except Exception:  # pragma: no cover - compatibility path
    from csi.attacks.adv_ml.attack_factory import generate_X_adv
from csi.attacks.utils.attack_utils import _compute_budgets, build_label_mapping
from csi.attacks.utils.metrics import compare_pairwise_adversarial
from csi.attacks.visualization import truncate_loader
from csi.data import get_experiment_loaders
from csi.data.utils import filter_none_collate, load_csi_bench_supervised
from csi.models.metrics import compute_f1_scores
from csi.models.model_factory import build_model
from csi.models.supervised import evaluate_classifier
from csi.models.utils.datasets import DatasetSpec
from csi.models.utils.experiment import load_config, resolve_config_path_from_arg, sanitize_token
from csi.models.utils.gpu_utils import _resolve_device
from csi.models.utils.results import (
    ASR_CI_FIELD_ORDER,
    WB_ATTACK_FIELD_ORDER,
    compute_asr_ci_summary,
    prepare_results_dir,
    save_config,
    save_results_rows,
    save_run_meta,
)
from csi.models.utils.seeding import configure_from_env_and_config

from ..common.shared_utils import (
    DEFAULT_CLEAN_WEIGHTS_DIR,
    normalize_csi_bench_task_name,
    parse_bool,
    parse_splits,
    parse_token_list,
    print_training_start,
)
from ..common.edge_profile import augment_profile_row, build_model_profile, prefix_profile_keys, resolve_profile_cfg
from ..common.optuna_clean_runtime import build_model_for_checkpoint, ensure_optuna_clean_checkpoint
from ..common.train_models_utils import (
    DEFAULT_CSI_BENCH_SUBCARRIERS,
    DEFAULT_CSI_BENCH_TIMESTEPS,
    as_list,
    configure_torch_performance,
    infer_input_shape_from_loader,
    infer_ntu_shape,
    resolve_ntu_num_classes,
    resolve_project_path,
    resolve_supervised_models,
)
from .whitebox_attack_apgd import (
    _apply_attack_kwargs_to_row,
    _as_weight_dir_list,
    _attack_kwargs_from_cfg,
    _call_with_supported_kwargs,
    _default_csi_tasks,
    _default_ntu_datasets,
    _format_attack_plan,
    _format_budget_diag,
    _loader_key_for_csi_split,
    _load_checkpoint_into_model,
    _nested_checkpoint_override,
    _normalise_csi_split_name,
    _normalise_selection_mode,
    _resolve_checkpoint,
    _resolve_csi_bench_checkpoint_dir,
)


def _enforce_har_hid_attack_policy_compat(
    project: str,
    attack_type: str,
    attack_method: str,
    *,
    strict: bool = True,
) -> Tuple[str, str]:
    mode = str(attack_type).strip().lower()
    method = str(attack_method).strip().lower()
    if not strict:
        return mode, method

    key = str(project).strip().lower().replace("_", "").replace("-", "")
    har_names = {"uthar", "ntufihar", "humanactivityrecognition"}
    hid_names = {"ntufihumanid", "humanidentification"}

    if key in har_names and mode != "untargeted":
        raise ValueError(f"{project}: HAR experiments must use attack.type='untargeted' (got {attack_type!r}).")
    if key in hid_names:
        if mode != "targeted":
            raise ValueError(f"{project}: HID experiments must use attack.type='targeted' (got {attack_type!r}).")
        if method not in {"apgd", "apgdt"}:
            raise ValueError(
                f"{project}: HID targeted experiments must use attack.method in {{'apgd','apgdt'}} "
                f"(got {attack_method!r})."
            )
    return mode, method


def _clean_metrics(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    metrics, _ = evaluate_classifier(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        use_tqdm=False,
        collect_predictions=False,
        collect_probabilities=False,
        desc="clean_eval",
    )
    return {
        "clean_acc": float(metrics.get("accuracy", 0.0)),
        "clean_loss": float(metrics.get("loss", 0.0)),
        "clean_f1_macro": float(metrics.get("f1_macro", 0.0)),
        "clean_f1_weighted": float(metrics.get("f1_weighted", 0.0)),
    }


def _safe_slug(value: Any) -> str:
    token = sanitize_token(str(value))
    return token if token else "na"


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool, int, float, np.integer, np.floating))


def _extract_adv_tensors(adv_loader, adv_targets: torch.Tensor, differences: Optional[torch.Tensor]):
    dataset = getattr(adv_loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "tensors"):
        raise RuntimeError("generate_X_adv returned a loader without TensorDataset tensors")
    tensors = dataset.tensors
    if len(tensors) < 2:
        raise RuntimeError("Expected TensorDataset(X_adv, y_true)")

    x_adv = tensors[0].detach().cpu()
    y_true = tensors[1].detach().cpu().long()
    y_target = adv_targets.detach().cpu().long() if adv_targets is not None else torch.empty(0, dtype=torch.long)
    delta = differences.detach().cpu() if isinstance(differences, torch.Tensor) else None
    if delta is not None and delta.shape[0] != x_adv.shape[0]:
        delta = None
    x_clean = (x_adv - delta) if delta is not None else None
    return x_adv, y_true, y_target, x_clean, delta


def _diag_from_adv_tensors(
    *,
    x_adv: torch.Tensor,
    x_clean: Optional[torch.Tensor],
    eps_2: float,
) -> Dict[str, Any]:
    if x_clean is None or x_clean.numel() == 0 or x_adv.numel() == 0:
        return {}

    flat_clean = x_clean.view(x_clean.size(0), -1)
    flat_delta = (x_adv - x_clean).view(x_clean.size(0), -1)
    pre = torch.linalg.norm(flat_clean, ord=2, dim=1)
    post = torch.linalg.norm(flat_delta, ord=2, dim=1)
    budget = float(eps_2) * pre
    ratio = post / pre.clamp_min(1e-12)

    diag: Dict[str, Any] = {
        "pre_l2_mean": float(pre.mean().item()),
        "post_l2_mean": float(post.mean().item()),
        "budget_l2_mean": float(budget.mean().item()),
        "l2_ratio_mean": float(ratio.mean().item()),
        "l2_ratio_std": float(ratio.std(unbiased=False).item()),
    }
    denom = max(diag["budget_l2_mean"], 1e-12)
    diag["utilization_mean"] = float(diag["post_l2_mean"] / denom)
    return diag


def _target_mode_for_success(
    *,
    attack_type: str,
    attack_method: str,
    attack_kwargs: Mapping[str, Any],
) -> str:
    if attack_type != "targeted":
        return "untargeted"
    if attack_method == "apgdt" and bool(attack_kwargs.get("target_class_sweep", False)):
        return "free_targeted"
    return "label_targeted"


def _evaluate_transfer_pair(
    *,
    model: nn.Module,
    x_adv: torch.Tensor,
    x_clean: Optional[torch.Tensor],
    y_true: torch.Tensor,
    y_target: torch.Tensor,
    device: torch.device,
    batch_size: int,
    target_mode: str,
    num_classes: int,
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    n = int(y_true.numel())
    if n == 0:
        return 0.0, {"n_all": 0, "n_clean_correct": 0}, {}

    if x_clean is None:
        x_clean = x_adv.clone()

    clean_preds: List[torch.Tensor] = []
    adv_preds: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    ts: List[torch.Tensor] = []

    total_clean_correct = 0
    successes = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, n, max(int(batch_size), 1)):
            end = min(start + max(int(batch_size), 1), n)
            xb_clean = x_clean[start:end].to(device, non_blocking=True)
            xb_adv = x_adv[start:end].to(device, non_blocking=True)
            yb = y_true[start:end].to(device, non_blocking=True).long()
            tb = y_target[start:end].to(device, non_blocking=True).long() if y_target.numel() == n else yb

            clean_pred = model(xb_clean).argmax(1)
            adv_pred = model(xb_adv).argmax(1)
            clean_mask = clean_pred == yb
            total_clean_correct += int(clean_mask.sum().item())

            if target_mode == "label_targeted":
                succ = (adv_pred == tb) & clean_mask
            else:
                succ = (adv_pred != yb) & clean_mask
            successes += int(succ.sum().item())

            clean_preds.append(clean_pred.detach().cpu())
            adv_preds.append(adv_pred.detach().cpu())
            ys.append(yb.detach().cpu())
            ts.append(tb.detach().cpu())

    y_all = torch.cat(ys, dim=0).numpy() if ys else np.array([], dtype=np.int64)
    clean_all = torch.cat(clean_preds, dim=0).numpy() if clean_preds else np.array([], dtype=np.int64)
    adv_all = torch.cat(adv_preds, dim=0).numpy() if adv_preds else np.array([], dtype=np.int64)
    target_all = torch.cat(ts, dim=0).numpy() if ts else np.array([], dtype=np.int64)

    clean_f1 = compute_f1_scores(y_all, clean_all) if y_all.size else {"macro": 0.0, "weighted": 0.0, "micro": 0.0}
    adv_f1 = compute_f1_scores(y_all, adv_all) if y_all.size else {"macro": 0.0, "weighted": 0.0, "micro": 0.0}

    pairwise: Dict[str, Any] = {}
    if target_mode == "label_targeted" and y_all.size and target_all.size and num_classes >= 2:
        try:
            pairwise = compare_pairwise_adversarial(adv_all, y_all, target_all, num_classes) or {}
        except Exception:
            pairwise = {}

    diag: Dict[str, Any] = {
        "n_all": int(n),
        "n_clean_correct": int(total_clean_correct),
        "clean_f1_macro": float(clean_f1["macro"]),
        "clean_f1_weighted": float(clean_f1["weighted"]),
        "clean_f1_micro": float(clean_f1["micro"]),
        "adv_f1_macro": float(adv_f1["macro"]),
        "adv_f1_weighted": float(adv_f1["weighted"]),
        "adv_f1_micro": float(adv_f1["micro"]),
    }

    asr = float(successes / max(1, total_clean_correct))
    return asr, diag, pairwise


def _save_generated_examples(
    *,
    out_dir: str,
    domain: str,
    dataset_name: str,
    source_model: str,
    budget_label: str,
    x_adv: torch.Tensor,
    y_true: torch.Tensor,
    y_target: torch.Tensor,
    x_clean: Optional[torch.Tensor],
    delta: Optional[torch.Tensor],
    perturbation_summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
    save_delta: bool,
    save_clean: bool,
) -> str:
    examples_dir = Path(out_dir) / "generated_adversarials" / _safe_slug(domain) / _safe_slug(dataset_name) / _safe_slug(source_model)
    examples_dir.mkdir(parents=True, exist_ok=True)
    path = examples_dir / f"{_safe_slug(budget_label)}.pt"
    payload: Dict[str, Any] = {
        "x_adv": x_adv,
        "y_true": y_true,
        "y_target": y_target,
        "perturbation_summary": dict(perturbation_summary or {}),
        "meta": dict(metadata),
    }
    if save_clean and x_clean is not None:
        payload["x_clean"] = x_clean
    if save_delta and delta is not None:
        payload["delta"] = delta
    torch.save(payload, str(path))
    return str(path)


def _model_list_from_transfer_cfg(transfer_cfg: Mapping[str, Any], fallback_models: Sequence[str]) -> Tuple[List[str], List[str], bool]:
    source_models = resolve_supervised_models(
        as_list(transfer_cfg.get("source_models") or transfer_cfg.get("sources"), default=list(fallback_models))
    )
    target_models = resolve_supervised_models(
        as_list(transfer_cfg.get("target_models") or transfer_cfg.get("targets"), default=list(fallback_models))
    )
    include_self = bool(transfer_cfg.get("include_self", False))
    return source_models, target_models, include_self


def _run_transfer_for_case(
    *,
    cfg: Mapping[str, Any],
    domain: str,
    dataset_name: str,
    dataset_key_for_model: str,
    train_loader,
    eval_loader,
    num_classes: int,
    arch_metadata: Mapping[str, Any],
    model_names: Sequence[str],
    source_models: Sequence[str],
    target_models: Sequence[str],
    include_self: bool,
    device: torch.device,
    budgets: List[Tuple[str, float]],
    attack_method: str,
    attack_type: str,
    steps: int,
    alpha: Optional[float],
    alpha_fraction: Optional[float],
    random_start: bool,
    n_restarts: int,
    attack_kwargs: Mapping[str, Any],
    strict_load: bool,
    selection: str,
    checkpoint_glob: Optional[str],
    checkpoint_paths: Any,
    weights_base_dir: Path,
    weights_dir_name: str,
    applied_seed: Optional[int],
    results_rows: List[Dict[str, Any]],
    out_dir: str,
    save_generated_examples: bool,
    save_generated_delta: bool,
    save_generated_clean: bool,
    profile_cfg: Mapping[str, Any],
) -> None:
    transfer_mode = _target_mode_for_success(
        attack_type=attack_type,
        attack_method=attack_method,
        attack_kwargs=attack_kwargs,
    )

    # Load all requested models once for this dataset/task.
    model_context: Dict[str, Dict[str, Any]] = {}
    channels = arch_metadata.get("channels")
    freq_bins = arch_metadata.get("freq_bins")
    time_steps = arch_metadata.get("time_steps")
    if channels is None or freq_bins is None or time_steps is None:
        raise RuntimeError(f"Missing input-shape metadata for transfer attack model build: dataset={dataset_name}")
    spec = DatasetSpec(
        "CSI-Bench" if domain == "csi_bench" else dataset_name,
        channels=int(channels),
        freq_bins=int(freq_bins),
        time_steps=int(time_steps),
    )
    for model_name in model_names:
        explicit_checkpoint = _nested_checkpoint_override(checkpoint_paths, dataset_name, model_name)
        explicit_checkpoint = ensure_optuna_clean_checkpoint(
            domain=domain,
            model_name=model_name,
            weights_dir_name=weights_dir_name,
            weights_cfg=dict(cfg.get("weights", {}) or {}),
            canonical_base_dir=weights_base_dir,
            alias_stems=[f"{dataset_name}_{model_name}", f"CSI-Bench_{model_name}"],
            dataset_name=(dataset_name if domain == "ntu" else None),
            task_name=(dataset_name if domain == "csi_bench" else None),
        ) or explicit_checkpoint
        model, _ = build_model_for_checkpoint(
            domain=domain,
            model_name=model_name,
            weights_dir_name=weights_dir_name,
            weights_cfg=dict(cfg.get("weights", {}) or {}),
            spec=spec,
            dataset_key_for_model=dataset_key_for_model,
            num_classes=num_classes,
            arch_metadata=arch_metadata,
            dataset_name=(dataset_name if domain == "ntu" else None),
            task_name=(dataset_name if domain == "csi_bench" else None),
        )
        model = model.to(device)
        checkpoint_path = _resolve_checkpoint(
            base_dir=weights_base_dir,
            model_name=model_name,
            selection=selection,
            alias_stems=[f"{dataset_name}_{model_name}", f"CSI-Bench_{model_name}"],
            checkpoint_glob=checkpoint_glob,
            explicit_checkpoint=explicit_checkpoint,
        )
        load_info = _load_checkpoint_into_model(
            model=model,
            checkpoint_path=str(checkpoint_path),
            strict=strict_load,
            device=device,
        )
        clean = _clean_metrics(model=model, loader=eval_loader, device=device)
        model_profile = build_model_profile(model=model, loader=eval_loader, device=device, profile_cfg=profile_cfg)
        model_context[model_name] = {
            "model": model,
            "checkpoint_path": str(checkpoint_path),
            "clean": clean,
            "load_info": load_info,
            "profile": model_profile,
        }

    available = set(model_context.keys())
    source_active = [m for m in source_models if m in available]
    target_active = [m for m in target_models if m in available]

    if not source_active:
        raise RuntimeError(f"No valid source models for {dataset_name}; requested={list(source_models)} available={sorted(available)}")
    if not target_active:
        raise RuntimeError(f"No valid target models for {dataset_name}; requested={list(target_models)} available={sorted(available)}")

    label_mapping, targeted_flag = build_label_mapping(dataset_name, attack_type, num_classes=int(num_classes))

    for source_model_name in source_active:
        source_ctx = model_context[source_model_name]
        source_model = source_ctx["model"]
        print(
            f"[TRANSFER][START] domain={domain} dataset={dataset_name} source_model={source_model_name} "
            f"weights_dir={weights_dir_name} "
            f"{_format_attack_plan(steps=steps, alpha=alpha, alpha_fraction=alpha_fraction, random_start=random_start, n_restarts=n_restarts, attack_kwargs=attack_kwargs)}"
        )

        for budget_label, eps_2 in budgets:
            print(
                f"[TRANSFER][START] domain={domain} dataset={dataset_name} source_model={source_model_name} "
                f"budget={budget_label} eps_2={float(eps_2):.6f}"
            )
            attack_kwargs_case = dict(attack_kwargs)
            attack_kwargs_case.setdefault("n_classes", int(num_classes))
            attack_kwargs_case.setdefault("dataset_name", dataset_name)

            adv_loader, adv_labels, perturbation_summary, differences = generate_X_adv(
                source_model,
                eval_loader,
                label_mapping,
                device=device,
                attack_method=attack_method,
                epsilon=float(eps_2),
                target_snr_db=None,
                targeted=bool(targeted_flag),
                verbose=False,
                alpha_fraction=float(alpha_fraction) if alpha_fraction is not None else None,
                alpha=float(alpha) if alpha is not None else None,
                steps=int(steps),
                random_start=bool(random_start),
                n_restarts=int(n_restarts),
                use_l2_ratio_metrics=True,
                **attack_kwargs_case,
            )

            x_adv, y_true, y_target, x_clean, delta = _extract_adv_tensors(adv_loader, adv_labels, differences)
            if x_adv.numel() == 0:
                print(
                    f"[TRANSFER][SKIP] domain={domain} dataset={dataset_name} source_model={source_model_name} "
                    f"budget={budget_label} reason=empty_adversarial_set"
                )
                continue

            adv_diag = _diag_from_adv_tensors(x_adv=x_adv, x_clean=x_clean, eps_2=float(eps_2))
            if isinstance(perturbation_summary, Mapping) and ("attack_success_rate" in perturbation_summary):
                attack_defined_success = float(perturbation_summary["attack_success_rate"])
                adv_diag["attack_defined_success_rate_all_mean"] = attack_defined_success
                adv_diag["attack_success_rate_mean"] = attack_defined_success
            print(
                f"[TRANSFER][GEN] domain={domain} dataset={dataset_name} source_model={source_model_name} "
                f"budget={budget_label} n_examples={int(x_adv.shape[0])} {_format_budget_diag(adv_diag)}"
            )

            saved_adv_path = None
            if save_generated_examples:
                saved_adv_path = _save_generated_examples(
                    out_dir=out_dir,
                    domain=domain,
                    dataset_name=dataset_name,
                    source_model=source_model_name,
                    budget_label=budget_label,
                    x_adv=x_adv,
                    y_true=y_true,
                    y_target=y_target,
                    x_clean=x_clean,
                    delta=delta,
                    perturbation_summary=perturbation_summary or {},
                    metadata={
                        "domain": domain,
                        "dataset": dataset_name,
                        "source_model": source_model_name,
                        "weights_dir_name": weights_dir_name,
                        "budget_label": budget_label,
                        "eps_2": float(eps_2),
                        "attack_method": attack_method,
                        "attack_mode": attack_type,
                        "target_mode": transfer_mode,
                        "steps": int(steps),
                        "alpha": float(alpha) if alpha is not None else None,
                        "alpha_fraction": float(alpha_fraction) if alpha_fraction is not None else None,
                        "random_start": bool(random_start),
                        "n_restarts": int(n_restarts),
                        "attack_kwargs": dict(attack_kwargs_case),
                        "seed": int(applied_seed) if applied_seed is not None else None,
                    },
                    save_delta=bool(save_generated_delta),
                    save_clean=bool(save_generated_clean),
                )

            eval_targets = list(target_active) if include_self else [m for m in target_active if m != source_model_name]
            if not eval_targets:
                print(
                    f"[TRANSFER][SKIP] domain={domain} dataset={dataset_name} source_model={source_model_name} "
                    f"budget={budget_label} reason=no_target_models_after_filter"
                )
                continue

            for target_model_name in eval_targets:
                target_ctx = model_context[target_model_name]
                target_model = target_ctx["model"]

                asr, pair_diag, pairwise = _evaluate_transfer_pair(
                    model=target_model,
                    x_adv=x_adv,
                    x_clean=x_clean,
                    y_true=y_true,
                    y_target=y_target,
                    device=device,
                    batch_size=getattr(eval_loader, "batch_size", 256) or 256,
                    target_mode=transfer_mode,
                    num_classes=int(num_classes),
                )

                row: Dict[str, Any] = {
                    "dataset": dataset_name,
                    "domain": domain,
                    "model": target_model_name,
                    "weights_dir_name": weights_dir_name,
                    "checkpoint_path": str(target_ctx.get("checkpoint_path")),
                    "attack_method": attack_method,
                    "attack_mode": f"transfer_{attack_type}",
                    "targeted": bool(targeted_flag),
                    "budget_label": str(budget_label),
                    "eps_2": float(eps_2),
                    "steps": int(steps),
                    "alpha": float(alpha) if alpha is not None else None,
                    "alpha_fraction": float(alpha_fraction) if alpha_fraction is not None else None,
                    "n_restarts": int(n_restarts),
                    "random_start": bool(random_start),
                    "asr": float(asr),
                    "device": str(device),
                    "seed": int(applied_seed) if applied_seed is not None else None,
                    "transfer_source_model": source_model_name,
                    "transfer_target_model": target_model_name,
                    "transfer_target_mode": transfer_mode,
                    "transfer_source_checkpoint": str(source_ctx.get("checkpoint_path")),
                    "saved_adv_path": str(saved_adv_path) if saved_adv_path else None,
                    **target_ctx.get("clean", {}),
                    **dict(target_ctx.get("profile", {}) or {}),
                    **prefix_profile_keys(dict(source_ctx.get("profile", {}) or {}), "source_"),
                    **adv_diag,
                    **pair_diag,
                    "checkpoint_missing_keys": len(target_ctx.get("load_info", {}).get("missing_keys", [])),
                    "checkpoint_unexpected_keys": len(target_ctx.get("load_info", {}).get("unexpected_keys", [])),
                }
                _apply_attack_kwargs_to_row(row, attack_kwargs_case)
                if isinstance(perturbation_summary, Mapping):
                    for key, value in perturbation_summary.items():
                        if _is_json_scalar(value):
                            row[f"source_{key}"] = value
                augment_profile_row(row, profile_cfg)
                source_attack_ms = row.get("source_attack_generation_time_per_example_ms")
                target_inference_ms = row.get("latency_ms_per_example")
                if isinstance(source_attack_ms, (int, float)) and isinstance(target_inference_ms, (int, float)):
                    row["source_attack_to_target_inference_ratio"] = float(
                        float(source_attack_ms) / max(float(target_inference_ms), 1e-12)
                    )
                source_projection_ms = row.get("source_projection_time_per_example_ms")
                if isinstance(source_projection_ms, (int, float)) and isinstance(target_inference_ms, (int, float)):
                    row["source_projection_to_target_inference_ratio"] = float(
                        float(source_projection_ms) / max(float(target_inference_ms), 1e-12)
                    )

                if pairwise:
                    row.update(
                        {
                            "best_conceal_label": pairwise.get("best_conceal", (None, None))[0],
                            "best_conceal_score": pairwise.get("best_conceal", (None, None))[1],
                            "worst_conceal_label": pairwise.get("worst_conceal", (None, None))[0],
                            "worst_conceal_score": pairwise.get("worst_conceal", (None, None))[1],
                            "best_promote_label": pairwise.get("best_promote", (None, None))[0],
                            "best_promote_score": pairwise.get("best_promote", (None, None))[1],
                            "worst_promote_label": pairwise.get("worst_promote", (None, None))[0],
                            "worst_promote_score": pairwise.get("worst_promote", (None, None))[1],
                        }
                    )

                results_rows.append(row)
                print(
                    f"[TRANSFER][DONE] domain={domain} dataset={dataset_name} "
                    f"source_model={source_model_name} target_model={target_model_name} "
                    f"budget={budget_label} asr={float(asr):.4f} "
                    f"n_clean_correct={int(pair_diag.get('n_clean_correct', 0))} "
                    f"{_format_budget_diag(adv_diag)}"
                )


def _run_ntu_transfer(
    *,
    cfg: Mapping[str, Any],
    out_dir: str,
    model_names: Sequence[str],
    source_models: Sequence[str],
    target_models: Sequence[str],
    include_self: bool,
    device: torch.device,
    attack_method: str,
    attack_type: str,
    budgets: List[Tuple[str, float]],
    steps: int,
    alpha: Optional[float],
    alpha_fraction: Optional[float],
    random_start: bool,
    n_restarts: int,
    attack_kwargs: Mapping[str, Any],
    strict_load: bool,
    selection: str,
    weights_cfg: Mapping[str, Any],
    applied_seed: Optional[int],
    max_batches: Optional[int],
    results_rows: List[Dict[str, Any]],
    save_generated_examples: bool,
    save_generated_delta: bool,
    save_generated_clean: bool,
    profile_cfg: Mapping[str, Any],
) -> None:
    ntu_cfg = dict(cfg.get("ntu", {}) or {})
    weights_ntu = dict(weights_cfg.get("ntu", {}) or {})

    datasets = as_list(
        ntu_cfg.get("datasets") or ntu_cfg.get("dataset"),
        default=_default_ntu_datasets("hid" if attack_type == "targeted" else "har"),
    )
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
    eval_split = str(ntu_cfg.get("eval_split", "test")).strip().lower()
    if eval_split not in {"test", "val"}:
        raise ValueError(f"Unsupported ntu.eval_split={eval_split!r}. Use 'test' or 'val'.")

    weights_root = str(weights_ntu.get("datasets_root", root))
    weights_dir_names = _as_weight_dir_list(
        weights_ntu.get("weights_dir_names", weights_ntu.get("weights_dir_name")),
        default=str(weights_cfg.get("weights_dir_name") or DEFAULT_CLEAN_WEIGHTS_DIR),
    )
    checkpoint_glob = weights_ntu.get("checkpoint_glob")
    checkpoint_paths = weights_ntu.get("checkpoint_paths")

    for dataset_name in datasets:
        for weights_dir_name in weights_dir_names:
            mode_fixed, method_fixed = _enforce_har_hid_attack_policy_compat(
                dataset_name,
                attack_type,
                attack_method,
                strict=True,
            )
            print_training_start(
                f"Transfer NTU ({dataset_name})",
                {
                    "attack_method": method_fixed,
                    "attack_type": mode_fixed,
                    "data_seed": data_seed,
                    "weights_dir_name": weights_dir_name,
                    "eval_split": eval_split,
                    "device": str(device),
                },
            )

            bundle = _call_with_supported_kwargs(
                get_experiment_loaders,
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
            eval_loader = val_loader if eval_split == "val" else test_loader
            if eval_loader is None:
                raise RuntimeError(f"Requested NTU eval split '{eval_split}' unavailable for {dataset_name}.")
            if train_loader is None:
                raise RuntimeError(f"NTU train loader missing for {dataset_name}.")

            if max_batches is not None:
                eval_loader = truncate_loader(eval_loader, int(max_batches))

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

            weights_base_dir = resolve_project_path(weights_root) / dataset_name / weights_dir_name
            _run_transfer_for_case(
                cfg=cfg,
                domain="ntu",
                dataset_name=dataset_name,
                dataset_key_for_model=dataset_name,
                train_loader=train_loader,
                eval_loader=eval_loader,
                num_classes=int(num_classes),
                arch_metadata=arch_metadata,
                model_names=model_names,
                source_models=source_models,
                target_models=target_models,
                include_self=include_self,
                device=device,
                budgets=budgets,
                attack_method=method_fixed,
                attack_type=mode_fixed,
                steps=steps,
                alpha=alpha,
                alpha_fraction=alpha_fraction,
                random_start=random_start,
                n_restarts=n_restarts,
                attack_kwargs=attack_kwargs,
                strict_load=strict_load,
                selection=selection,
                checkpoint_glob=str(checkpoint_glob) if checkpoint_glob else None,
                checkpoint_paths=checkpoint_paths,
                weights_base_dir=weights_base_dir,
                weights_dir_name=weights_dir_name,
                applied_seed=applied_seed,
                results_rows=results_rows,
                out_dir=out_dir,
                save_generated_examples=save_generated_examples,
                save_generated_delta=save_generated_delta,
                save_generated_clean=save_generated_clean,
                profile_cfg=profile_cfg,
            )


def _run_csi_bench_transfer(
    *,
    cfg: Mapping[str, Any],
    out_dir: str,
    model_names: Sequence[str],
    source_models: Sequence[str],
    target_models: Sequence[str],
    include_self: bool,
    device: torch.device,
    attack_method: str,
    attack_type: str,
    budgets: List[Tuple[str, float]],
    steps: int,
    alpha: Optional[float],
    alpha_fraction: Optional[float],
    random_start: bool,
    n_restarts: int,
    attack_kwargs: Mapping[str, Any],
    strict_load: bool,
    selection: str,
    weights_cfg: Mapping[str, Any],
    applied_seed: Optional[int],
    max_batches: Optional[int],
    results_rows: List[Dict[str, Any]],
    save_generated_examples: bool,
    save_generated_delta: bool,
    save_generated_clean: bool,
    profile_cfg: Mapping[str, Any],
) -> None:
    csi_cfg = dict(cfg.get("csi_bench", {}) or {})
    weights_csi = dict(weights_cfg.get("csi_bench", {}) or {})

    task_names = parse_token_list(
        csi_cfg.get("task_names")
        or csi_cfg.get("tasks")
        or csi_cfg.get("task_name")
        or ",".join(_default_csi_tasks("hid" if attack_type == "targeted" else "har"))
    )
    task_names = [normalize_csi_bench_task_name(name) for name in task_names]
    if not task_names:
        task_names = _default_csi_tasks("hid" if attack_type == "targeted" else "har")

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
    eval_split_name = str(csi_cfg.get("eval_split_name", "test_id"))
    canonical_split = _normalise_csi_split_name(eval_split_name)
    loader_key = _loader_key_for_csi_split(eval_split_name)

    weights_dataset_root = str(weights_csi.get("dataset_root", dataset_root))
    weights_dir_names = _as_weight_dir_list(
        weights_csi.get("weights_dir_names", weights_csi.get("weights_dir_name")),
        default=str(weights_cfg.get("weights_dir_name") or DEFAULT_CLEAN_WEIGHTS_DIR),
    )
    checkpoint_glob = weights_csi.get("checkpoint_glob")
    checkpoint_paths = weights_csi.get("checkpoint_paths")

    for task_name in task_names:
        for weights_dir_name in weights_dir_names:
            mode_fixed, method_fixed = _enforce_har_hid_attack_policy_compat(
                task_name,
                attack_type,
                attack_method,
                strict=True,
            )
            print_training_start(
                f"Transfer CSI-Bench ({task_name})",
                {
                    "attack_method": method_fixed,
                    "attack_type": mode_fixed,
                    "weights_dir_name": weights_dir_name,
                    "eval_split_name": eval_split_name,
                    "device": str(device),
                },
            )

            bundle = _call_with_supported_kwargs(
                load_csi_bench_supervised,
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
                only_splits=parse_splits([canonical_split], default=["test_id"]),
                splits_dir_name=splits_dir_name,
                collate_fn=filter_none_collate,
            )
            if not bundle:
                raise RuntimeError(f"Failed to load CSI-Bench task {task_name}")
            loaders = bundle["loaders"]
            eval_loader = loaders.get(loader_key)
            if eval_loader is None:
                raise RuntimeError(
                    f"Requested CSI-Bench split '{eval_split_name}' unavailable for {task_name}. "
                    f"Available={sorted(loaders.keys())}"
                )
            if max_batches is not None:
                eval_loader = truncate_loader(eval_loader, int(max_batches))

            num_classes = int(bundle["num_classes"])
            shape = infer_input_shape_from_loader(eval_loader)
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

            weights_base_dir = _resolve_csi_bench_checkpoint_dir(
                weights_dataset_root=weights_dataset_root,
                weights_dir_name=weights_dir_name,
                task_name=task_name,
            )
            _run_transfer_for_case(
                cfg=cfg,
                domain="csi_bench",
                dataset_name=task_name,
                dataset_key_for_model="csi-bench",
                train_loader=None,
                eval_loader=eval_loader,
                num_classes=int(num_classes),
                arch_metadata=arch_metadata,
                model_names=model_names,
                source_models=source_models,
                target_models=target_models,
                include_self=include_self,
                device=device,
                budgets=budgets,
                attack_method=method_fixed,
                attack_type=mode_fixed,
                steps=steps,
                alpha=alpha,
                alpha_fraction=alpha_fraction,
                random_start=random_start,
                n_restarts=n_restarts,
                attack_kwargs=attack_kwargs,
                strict_load=strict_load,
                selection=selection,
                checkpoint_glob=str(checkpoint_glob) if checkpoint_glob else None,
                checkpoint_paths=checkpoint_paths,
                weights_base_dir=weights_base_dir,
                weights_dir_name=weights_dir_name,
                applied_seed=applied_seed,
                results_rows=results_rows,
                out_dir=out_dir,
                save_generated_examples=save_generated_examples,
                save_generated_delta=save_generated_delta,
                save_generated_clean=save_generated_clean,
                profile_cfg=profile_cfg,
            )


def main(
    config_name: Optional[str] = None,
    *,
    attack_method_override: Optional[str] = None,
    default_attack_type: str = "untargeted",
    default_task_family: str = "har",
    default_free_targeted: bool = False,
    default_config_relpath: Optional[str] = None,
) -> None:
    cfg_arg = config_name or os.environ.get("EXP_ATTACK_CONFIG")
    cfg_path = resolve_config_path_from_arg(cfg_arg)
    if cfg_path is None and default_config_relpath:
        cfg_path = resolve_config_path_from_arg(default_config_relpath)
    cfg = load_config(cfg_path)

    applied_seed = configure_from_env_and_config(cfg)
    configure_torch_performance(cfg)

    common_cfg = cfg.get("common", {})
    experiment_name = str(common_cfg.get("experiment_name", "transfer_apgd"))

    data_cfg = cfg.get("data", {})
    device = _resolve_device(data_cfg.get("device"))
    profile_cfg = resolve_profile_cfg(cfg)

    model_cfg = cfg.get("model", {})
    model_names = resolve_supervised_models(
        as_list(model_cfg.get("architectures") or model_cfg.get("models"), default=["ResNet18", "LeNet", "BiLSTM"])
    )

    transfer_cfg = dict(cfg.get("transfer", {}) or {})
    source_models, target_models, include_self = _model_list_from_transfer_cfg(transfer_cfg, model_names)
    max_batches_raw = transfer_cfg.get("max_batches", (cfg.get("eval", {}) or {}).get("max_batches"))
    try:
        max_batches = int(max_batches_raw) if max_batches_raw is not None else None
        if max_batches is not None and max_batches <= 0:
            max_batches = None
    except Exception:
        max_batches = None

    save_generated_examples = parse_bool(transfer_cfg.get("save_generated_examples"), default=True)
    save_generated_delta = parse_bool(transfer_cfg.get("save_generated_delta"), default=True)
    save_generated_clean = parse_bool(transfer_cfg.get("save_generated_clean"), default=False)

    run_cfg = cfg.get("run", {})
    targets_raw = as_list(run_cfg.get("targets"), default=["ntu", "csi_bench"])
    targets = {token.strip().lower().replace("-", "_") for token in targets_raw}

    attack_cfg = dict(cfg.get("attack", {}) or {})
    attack_method = str(
        attack_method_override
        or attack_cfg.get("method")
        or (attack_cfg.get("methods") or ["apgd"])[0]
    ).strip().lower()
    if attack_method not in {"apgd", "apgdt"}:
        raise ValueError("transfer attacks currently support only attack.method in {'apgd', 'apgdt'}")

    attack_type = str(attack_cfg.get("type", default_attack_type)).strip().lower()
    if attack_type not in {"untargeted", "targeted"}:
        raise ValueError("attack.type must be one of {'untargeted', 'targeted'}")

    task_family = str(attack_cfg.get("task_family", default_task_family)).strip().lower()
    if task_family not in {"har", "hid"}:
        raise ValueError("attack.task_family must be 'har' or 'hid'.")

    steps = int(attack_cfg.get("steps", 100))
    alpha = attack_cfg.get("alpha")
    alpha = float(alpha) if alpha is not None else None
    alpha_fraction = attack_cfg.get("alpha_fraction")
    alpha_fraction = float(alpha_fraction) if alpha_fraction is not None else None
    random_start = bool(attack_cfg.get("random_start", True))
    n_restarts = int(attack_cfg.get("n_restarts", 1))

    budgets = _compute_budgets(attack_cfg)
    attack_kwargs = _attack_kwargs_from_cfg(
        attack_cfg,
        attack_method=attack_method,
        attack_type=attack_type,
        default_free_targeted=default_free_targeted,
    )

    weights_cfg = dict(cfg.get("weights", {}) or {})
    selection = _normalise_selection_mode(weights_cfg.get("selection"))
    strict_load = parse_bool(weights_cfg.get("strict_load"), default=True)

    results_cfg = dict(cfg.get("results", {}) or {})
    results_job_dir = results_cfg.get("job_dir")
    results_job_name = results_cfg.get("job_name")
    results_job_timestamp = results_cfg.get("job_timestamp")

    simple_paths = parse_bool(results_cfg.get("simple_paths"), default=True)
    if simple_paths:
        relative_parts: List[str] = []
    else:
        relative_parts = [attack_method, attack_type, task_family]
        if applied_seed is not None:
            relative_parts.append(f"seed_{int(applied_seed)}")

    out_dir, ts, cfg_name = prepare_results_dir(
        experiment_name=experiment_name,
        cfg_path=cfg_path,
        mode=f"transfer_{attack_type}_{attack_method}",
        job_dir=str(results_job_dir) if results_job_dir else None,
        job_name=str(results_job_name) if results_job_name else None,
        job_timestamp=str(results_job_timestamp) if results_job_timestamp else None,
        relative_parts=relative_parts,
    )
    save_config(out_dir, dict(cfg))

    print_training_start(
        "Transfer robustness run",
        {
            "config_path": cfg_path,
            "output_dir": out_dir,
            "attack_method": attack_method,
            "attack_type": attack_type,
            "task_family": task_family,
            "budgets": [label for (label, _) in budgets],
            "steps": steps,
            "alpha": alpha,
            "alpha_fraction": alpha_fraction,
            "random_start": random_start,
            "n_restarts": n_restarts,
            "attack_kwargs": dict(attack_kwargs),
            "device": str(device),
            "seed": applied_seed,
            "source_models": list(source_models),
            "target_models": list(target_models),
            "include_self": bool(include_self),
            "save_generated_examples": bool(save_generated_examples),
        },
    )

    results_rows: List[Dict[str, Any]] = []
    if "ntu" in targets:
        _run_ntu_transfer(
            cfg=cfg,
            out_dir=out_dir,
            model_names=model_names,
            source_models=source_models,
            target_models=target_models,
            include_self=include_self,
            device=device,
            attack_method=attack_method,
            attack_type=attack_type,
            budgets=budgets,
            steps=steps,
            alpha=alpha,
            alpha_fraction=alpha_fraction,
            random_start=random_start,
            n_restarts=n_restarts,
            attack_kwargs=attack_kwargs,
            strict_load=strict_load,
            selection=selection,
            weights_cfg=weights_cfg,
            applied_seed=applied_seed,
            max_batches=max_batches,
            results_rows=results_rows,
            save_generated_examples=save_generated_examples,
            save_generated_delta=save_generated_delta,
            save_generated_clean=save_generated_clean,
            profile_cfg=profile_cfg,
        )
    if "csi_bench" in targets:
        _run_csi_bench_transfer(
            cfg=cfg,
            out_dir=out_dir,
            model_names=model_names,
            source_models=source_models,
            target_models=target_models,
            include_self=include_self,
            device=device,
            attack_method=attack_method,
            attack_type=attack_type,
            budgets=budgets,
            steps=steps,
            alpha=alpha,
            alpha_fraction=alpha_fraction,
            random_start=random_start,
            n_restarts=n_restarts,
            attack_kwargs=attack_kwargs,
            strict_load=strict_load,
            selection=selection,
            weights_cfg=weights_cfg,
            applied_seed=applied_seed,
            max_batches=max_batches,
            results_rows=results_rows,
            save_generated_examples=save_generated_examples,
            save_generated_delta=save_generated_delta,
            save_generated_clean=save_generated_clean,
            profile_cfg=profile_cfg,
        )

    save_results_rows(out_dir, results_rows, base_name="summary_rows", field_order=WB_ATTACK_FIELD_ORDER)
    ci_rows = compute_asr_ci_summary(results_rows, config_name=cfg_name)
    if ci_rows:
        save_results_rows(out_dir, ci_rows, base_name="asr_ci95", field_order=ASR_CI_FIELD_ORDER)

    save_run_meta(
        out_dir,
        {
            "timestamp": ts,
            "config_name": cfg_name,
            "attack": {
                "method": attack_method,
                "type": attack_type,
                "task_family": task_family,
                "steps": steps,
                "alpha": alpha,
                "alpha_fraction": alpha_fraction,
                "random_start": random_start,
                "n_restarts": n_restarts,
                "kwargs": dict(attack_kwargs),
                "budgets": [label for (label, _) in budgets],
            },
            "transfer": {
                "source_models": list(source_models),
                "target_models": list(target_models),
                "include_self": bool(include_self),
                "max_batches": max_batches,
                "save_generated_examples": bool(save_generated_examples),
                "save_generated_delta": bool(save_generated_delta),
                "save_generated_clean": bool(save_generated_clean),
            },
            "weights": {
                "selection": selection,
                "strict_load": bool(strict_load),
            },
            "models": list(model_names),
            "targets": sorted(targets),
            "seed": int(applied_seed) if applied_seed is not None else None,
            "num_rows": int(len(results_rows)),
        },
    )
    print(f"[INFO] Saved transfer results: {out_dir}")


if __name__ == "__main__":
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_name=arg)
