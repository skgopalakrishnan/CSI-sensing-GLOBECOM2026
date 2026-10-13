"""Universal perturbation benchmarking for APGD/APGDT-style objectives on NTU and CSI-Bench."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from csi.attacks.utils.attack_utils import (
    _compute_budgets,
    _mean_l2_norm,
    _project_l2,
    build_label_mapping,
)
from csi.attacks.utils.correlation_utils import process_csi_tensor_for_correlation
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
from ..common.edge_profile import augment_profile_row, build_model_profile, resolve_profile_cfg
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
    _as_weight_dir_list,
    _call_with_supported_kwargs,
    _default_csi_tasks,
    _default_ntu_datasets,
    _format_budget_diag,
    _format_log_value,
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


def _dlr_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    x_sorted, ind_sorted = logits.sort(dim=1)
    ind = (ind_sorted[:, -1] == labels).float()
    if x_sorted.size(1) >= 3:
        denom = (x_sorted[:, -1] - x_sorted[:, -3]).clamp_min(1e-12)
    elif x_sorted.size(1) == 2:
        denom = (x_sorted[:, -1] - x_sorted[:, -2]).clamp_min(1e-12)
    else:
        denom = torch.ones_like(labels, dtype=logits.dtype, device=logits.device)
    second_best = x_sorted[:, -2] if x_sorted.size(1) >= 2 else x_sorted[:, -1]
    row_idx = torch.arange(logits.size(0), device=logits.device)
    return -(
        logits[row_idx, labels]
        - second_best * ind
        - x_sorted[:, -1] * (1.0 - ind)
    ) / denom


def _dlr_loss_targeted(logits: torch.Tensor, labels: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
    row_idx = torch.arange(logits.size(0), device=logits.device)
    z_y = logits[row_idx, labels]
    z_t = logits[row_idx, target_labels]
    top2 = logits.topk(k=min(2, logits.size(1)), dim=1).values
    if top2.size(1) >= 2:
        denom = (top2[:, 0] - top2[:, 1]).clamp_min(1e-12)
    else:
        denom = torch.ones_like(z_y)
    return -((z_y - z_t) / denom)


def _labelwise_targets(labels: torch.Tensor, mapping: Mapping[int, List[int]], device: torch.device) -> torch.Tensor:
    mapped = [int(mapping[int(lbl.item())][0]) for lbl in labels.detach().cpu()]
    return torch.tensor(mapped, dtype=torch.long, device=device)


def _free_target_candidates_from_clean_logits(
    logits_clean: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_target_classes: int,
) -> torch.Tensor:
    classes_desc = logits_clean.argsort(dim=1, descending=True)
    wrong_mask = classes_desc != labels.unsqueeze(1)
    pool = classes_desc[wrong_mask].reshape(logits_clean.size(0), -1)
    k = max(1, min(int(n_target_classes), pool.size(1)))
    return pool[:, :k]


def _apply_rnn_safe_mode(model: nn.Module) -> Tuple[bool, bool]:
    prev_training = bool(model.training)
    has_rnn = False
    try:
        has_rnn = any(isinstance(m, (nn.RNN, nn.GRU, nn.LSTM)) for m in model.modules())
    except Exception:
        has_rnn = False

    if has_rnn:
        model.train()
        for module in model.modules():
            if "BatchNorm" in module.__class__.__name__:
                module.eval()
            if "Dropout" in module.__class__.__name__:
                module.eval()
    else:
        model.eval()

    return prev_training, has_rnn


def _restore_model_mode(model: nn.Module, prev_training: bool) -> None:
    if prev_training:
        model.train()
    else:
        model.eval()


def _freeze_model_params(model: nn.Module) -> List[bool]:
    prev: List[bool] = []
    for param in model.parameters():
        prev.append(bool(param.requires_grad))
        param.requires_grad_(False)
    return prev


def _restore_model_params(model: nn.Module, prev: Sequence[bool]) -> None:
    for param, req in zip(model.parameters(), prev):
        param.requires_grad_(bool(req))


def _batch_objective_for_uap(
    *,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    delta: torch.Tensor,
    attack_type: str,
    attack_method: str,
    loss_name: str,
    label_mapping: Mapping[int, List[int]],
    target_class_sweep: bool,
    n_target_classes: int,
) -> torch.Tensor:
    x_adv = x + delta.unsqueeze(0)
    logits_adv = model(x_adv)
    ce_none = nn.CrossEntropyLoss(reduction="none")

    if attack_type == "untargeted":
        if loss_name == "dlr":
            return _dlr_loss(logits_adv, y).mean()
        return ce_none(logits_adv, y).mean()

    # Targeted modes
    if attack_method == "apgdt" and target_class_sweep:
        with torch.no_grad():
            logits_clean = model(x)
        target_pool = _free_target_candidates_from_clean_logits(
            logits_clean,
            y,
            n_target_classes=n_target_classes,
        )
        per_target_losses = []
        for idx in range(target_pool.size(1)):
            t = target_pool[:, idx]
            per_target_losses.append(_dlr_loss_targeted(logits_adv, y, t))
        stacked = torch.stack(per_target_losses, dim=1)
        # Approximate APGD-T sweep by optimizing the best target objective per sample.
        return stacked.max(dim=1).values.mean()

    targets = _labelwise_targets(y, label_mapping, device=y.device)
    if loss_name == "dlr":
        return _dlr_loss_targeted(logits_adv, y, targets).mean()
    return (-ce_none(logits_adv, targets)).mean()


def _estimate_uap_fooling(
    *,
    model: nn.Module,
    loader,
    delta: torch.Tensor,
    device: torch.device,
    attack_type: str,
    attack_method: str,
    label_mapping: Mapping[int, List[int]],
    target_class_sweep: bool,
    max_batches: Optional[int],
) -> Tuple[float, int, int]:
    total = 0
    clean_correct = 0
    successes = 0

    prev_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for bidx, batch in enumerate(loader):
                if max_batches is not None and bidx >= int(max_batches):
                    break
                xb, yb = batch
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).long()

                adv_pred = model(xb + delta.unsqueeze(0)).argmax(1)
                clean_pred = model(xb).argmax(1)
                clean_mask = clean_pred == yb

                total += int(yb.numel())
                clean_correct += int(clean_mask.sum().item())

                if attack_type == "targeted" and not (attack_method == "apgdt" and target_class_sweep):
                    tb = _labelwise_targets(yb, label_mapping, device=device)
                    succ = (adv_pred == tb) & clean_mask
                else:
                    succ = (adv_pred != yb) & clean_mask
                successes += int(succ.sum().item())
    finally:
        _restore_model_mode(model, prev_training)

    return float(successes / max(1, clean_correct)), int(clean_correct), int(total)


def _train_uap_delta(
    *,
    model: nn.Module,
    train_loader,
    dataset_name: str,
    device: torch.device,
    eps_2: float,
    attack_type: str,
    attack_method: str,
    loss_name: str,
    label_mapping: Mapping[int, List[int]],
    target_class_sweep: bool,
    n_target_classes: int,
    max_passes: int,
    step_size: Optional[float],
    step_fraction: float,
    target_fooling: Optional[float],
    mean_norm_batches: Optional[int],
    max_train_batches: Optional[int],
    max_eval_batches: Optional[int],
    preserve_correlation: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    first_batch = next(iter(train_loader))
    x0 = first_batch[0].to(device)
    if x0.ndim < 2:
        raise RuntimeError("Expected batched tensor from train loader for UAP training")

    mean_norm = float(_mean_l2_norm(train_loader, device, max_batches=mean_norm_batches))
    radius = float(eps_2) * float(mean_norm)
    delta = torch.zeros_like(x0[0], device=device)

    prev_training, _ = _apply_rnn_safe_mode(model)
    prev_req = _freeze_model_params(model)

    last_train_fooling = 0.0
    total_updates = 0
    total_examples = 0
    projection_time_s_total = 0.0
    projection_calls_total = 0
    correlation_projection_time_s_total = 0.0
    effective_step = float(step_size) if step_size is not None else float(step_fraction) * float(radius)
    effective_step = max(effective_step, 1e-8)
    max_passes_eff = max(1, int(max_passes))
    attack_started_at = time.perf_counter()

    try:
        for p in range(max_passes_eff):
            for bidx, (xb, yb) in enumerate(train_loader):
                if max_train_batches is not None and bidx >= int(max_train_batches):
                    break

                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).long()
                total_examples += int(yb.numel())
                delta_var = delta.detach().clone().requires_grad_(True)

                objective = _batch_objective_for_uap(
                    model=model,
                    x=xb,
                    y=yb,
                    delta=delta_var,
                    attack_type=attack_type,
                    attack_method=attack_method,
                    loss_name=loss_name,
                    label_mapping=label_mapping,
                    target_class_sweep=target_class_sweep,
                    n_target_classes=n_target_classes,
                )
                grad = torch.autograd.grad(objective, delta_var, retain_graph=False, create_graph=False)[0]
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                grad_norm = grad.reshape(-1).norm(p=2).clamp_min(1e-12)

                delta = (delta + effective_step * (grad / grad_norm)).detach()
                project_started_at = time.perf_counter()
                if preserve_correlation:
                    clean_ref = xb[:1].detach()
                    corr_started_at = time.perf_counter()
                    delta = process_csi_tensor_for_correlation(
                        delta.unsqueeze(0),
                        dataset_name=str(dataset_name),
                        clean_tensor=clean_ref,
                    ).squeeze(0)
                    delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                    correlation_projection_time_s_total += float(time.perf_counter() - corr_started_at)

                delta = _project_l2(delta, float(radius)).detach()
                projection_time_s_total += float(time.perf_counter() - project_started_at)
                projection_calls_total += 1
                total_updates += 1

            if target_fooling is not None:
                pass_fooling, _, _ = _estimate_uap_fooling(
                    model=model,
                    loader=train_loader,
                    delta=delta,
                    device=device,
                    attack_type=attack_type,
                    attack_method=attack_method,
                    label_mapping=label_mapping,
                    target_class_sweep=target_class_sweep,
                    max_batches=max_eval_batches,
                )
                last_train_fooling = float(pass_fooling)
                if pass_fooling >= float(target_fooling):
                    delta_norm = float(delta.reshape(-1).norm(p=2).item())
                    util = float(delta_norm / max(float(radius), 1e-12))
                    print(
                        f"[UAP][PASS] dataset={dataset_name} pass={p + 1}/{max_passes_eff} "
                        f"delta_l2={delta_norm:.6f} radius_abs={float(radius):.6f} "
                        f"utilization={util:.4f} train_fooling={float(pass_fooling):.4f} "
                        f"target_fooling={float(target_fooling):.4f} total_updates={int(total_updates)}"
                    )
                    break
            delta_norm = float(delta.reshape(-1).norm(p=2).item())
            util = float(delta_norm / max(float(radius), 1e-12))
            msg = (
                f"[UAP][PASS] dataset={dataset_name} pass={p + 1}/{max_passes_eff} "
                f"delta_l2={delta_norm:.6f} radius_abs={float(radius):.6f} "
                f"utilization={util:.4f} total_updates={int(total_updates)}"
            )
            if target_fooling is not None:
                msg += (
                    f" train_fooling={float(last_train_fooling):.4f} "
                    f"target_fooling={float(target_fooling):.4f}"
                )
            print(msg)
    finally:
        _restore_model_params(model, prev_req)
        _restore_model_mode(model, prev_training)

    attack_generation_time_s_total = float(time.perf_counter() - attack_started_at)
    stats: Dict[str, Any] = {
        "mean_norm_ref": float(mean_norm),
        "radius_abs": float(radius),
        "step_size": float(effective_step),
        "train_fooling_rate": float(last_train_fooling),
        "total_updates": int(total_updates),
        "attack_num_examples": int(total_examples),
        "attack_generation_time_s_total": attack_generation_time_s_total,
        "uap_utilization_mean": float(delta.reshape(-1).norm(p=2).item() / max(float(radius), 1e-12)),
    }
    if total_updates > 0:
        stats["attack_generation_time_per_update_ms"] = float((attack_generation_time_s_total * 1000.0) / total_updates)
    if total_examples > 0:
        stats["attack_generation_time_per_example_ms"] = float((attack_generation_time_s_total * 1000.0) / total_examples)
    stats["projection_time_s_total"] = float(projection_time_s_total)
    stats["projection_calls_total"] = int(projection_calls_total)
    if projection_calls_total > 0:
        stats["projection_time_per_call_ms"] = float((projection_time_s_total * 1000.0) / projection_calls_total)
    if total_examples > 0:
        stats["projection_time_per_example_ms"] = float((projection_time_s_total * 1000.0) / total_examples)
    if attack_generation_time_s_total > 0.0:
        stats["projection_time_share"] = float(projection_time_s_total / attack_generation_time_s_total)
        stats["correlation_projection_time_share"] = float(
            correlation_projection_time_s_total / attack_generation_time_s_total
        )
    stats["correlation_projection_time_s_total"] = float(correlation_projection_time_s_total)
    return delta.detach().cpu(), stats


def _evaluate_uap(
    *,
    model: nn.Module,
    eval_loader,
    delta: torch.Tensor,
    device: torch.device,
    attack_type: str,
    attack_method: str,
    label_mapping: Mapping[int, List[int]],
    target_class_sweep: bool,
    eps_2: float,
) -> Tuple[float, Dict[str, Any], Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_adv_list: List[torch.Tensor] = []
    x_clean_list: List[torch.Tensor] = []
    y_true_list: List[torch.Tensor] = []
    y_target_list: List[torch.Tensor] = []

    clean_preds_all: List[np.ndarray] = []
    adv_preds_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []

    total = 0
    clean_correct = 0
    successes = 0

    model.eval()
    with torch.no_grad():
        for xb, yb in eval_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()
            db = delta.to(device).reshape(1, *delta.shape).expand(xb.size(0), *delta.shape)
            xa = xb + db

            clean_pred = model(xb).argmax(1)
            adv_pred = model(xa).argmax(1)
            clean_mask = clean_pred == yb

            if attack_type == "targeted" and not (attack_method == "apgdt" and target_class_sweep):
                tb = _labelwise_targets(yb, label_mapping, device=device)
                succ = (adv_pred == tb) & clean_mask
            else:
                tb = yb.clone()
                succ = (adv_pred != yb) & clean_mask

            total += int(yb.numel())
            clean_correct += int(clean_mask.sum().item())
            successes += int(succ.sum().item())

            clean_preds_all.append(clean_pred.detach().cpu().numpy())
            adv_preds_all.append(adv_pred.detach().cpu().numpy())
            labels_all.append(yb.detach().cpu().numpy())

            x_adv_list.append(xa.detach().cpu())
            x_clean_list.append(xb.detach().cpu())
            y_true_list.append(yb.detach().cpu())
            y_target_list.append(tb.detach().cpu())

    x_adv = torch.cat(x_adv_list, dim=0) if x_adv_list else torch.empty(0)
    x_clean = torch.cat(x_clean_list, dim=0) if x_clean_list else torch.empty(0)
    y_true = torch.cat(y_true_list, dim=0).long() if y_true_list else torch.empty(0, dtype=torch.long)
    y_target = torch.cat(y_target_list, dim=0).long() if y_target_list else torch.empty(0, dtype=torch.long)

    if labels_all and clean_preds_all:
        clean_f1 = compute_f1_scores(np.concatenate(labels_all), np.concatenate(clean_preds_all))
        adv_f1 = compute_f1_scores(np.concatenate(labels_all), np.concatenate(adv_preds_all))
    else:
        clean_f1 = {"macro": 0.0, "weighted": 0.0, "micro": 0.0}
        adv_f1 = {"macro": 0.0, "weighted": 0.0, "micro": 0.0}

    diag: Dict[str, Any] = {
        "n_all": int(total),
        "n_clean_correct": int(clean_correct),
        "clean_f1_macro": float(clean_f1["macro"]),
        "clean_f1_weighted": float(clean_f1["weighted"]),
        "clean_f1_micro": float(clean_f1["micro"]),
        "adv_f1_macro": float(adv_f1["macro"]),
        "adv_f1_weighted": float(adv_f1["weighted"]),
        "adv_f1_micro": float(adv_f1["micro"]),
    }

    if x_clean.numel() > 0 and x_adv.numel() > 0:
        flat_clean = x_clean.reshape(x_clean.size(0), -1)
        flat_delta = (x_adv - x_clean).reshape(x_clean.size(0), -1)
        pre = torch.linalg.norm(flat_clean, ord=2, dim=1)
        post = torch.linalg.norm(flat_delta, ord=2, dim=1)
        budget = float(eps_2) * pre
        ratio = post / pre.clamp_min(1e-12)

        diag.update(
            {
                "pre_l2_mean": float(pre.mean().item()),
                "post_l2_mean": float(post.mean().item()),
                "budget_l2_mean": float(budget.mean().item()),
                "l2_ratio_mean": float(ratio.mean().item()),
                "l2_ratio_std": float(ratio.std(unbiased=False).item()),
            }
        )
        denom = max(diag["budget_l2_mean"], 1e-12)
        diag["utilization_mean"] = float(diag["post_l2_mean"] / denom)

    asr = float(successes / max(1, clean_correct))
    return asr, diag, {"clean_f1": clean_f1, "adv_f1": adv_f1}, x_adv, x_clean, y_true, y_target


def _save_uap_artifacts(
    *,
    out_dir: str,
    domain: str,
    dataset_name: str,
    model_name: str,
    budget_label: str,
    delta: torch.Tensor,
    train_stats: Mapping[str, Any],
    eval_diag: Mapping[str, Any],
    x_adv: Optional[torch.Tensor],
    x_clean: Optional[torch.Tensor],
    y_true: Optional[torch.Tensor],
    y_target: Optional[torch.Tensor],
    save_examples: bool,
    save_clean: bool,
) -> Tuple[str, Optional[str]]:
    base_dir = Path(out_dir) / "uap_artifacts" / _safe_slug(domain) / _safe_slug(dataset_name) / _safe_slug(model_name)
    base_dir.mkdir(parents=True, exist_ok=True)

    uap_path = base_dir / f"uap_{_safe_slug(budget_label)}.pt"
    torch.save(
        {
            "uap": delta,
            "meta": {
                "domain": domain,
                "dataset": dataset_name,
                "model": model_name,
                "budget_label": budget_label,
                "train_stats": dict(train_stats),
                "eval_diag": dict(eval_diag),
            },
        },
        str(uap_path),
    )

    adv_path: Optional[str] = None
    if save_examples and x_adv is not None and y_true is not None and y_target is not None:
        adv_file = base_dir / f"adv_examples_{_safe_slug(budget_label)}.pt"
        payload: Dict[str, Any] = {
            "x_adv": x_adv,
            "y_true": y_true,
            "y_target": y_target,
            "delta": (x_adv - x_clean) if x_clean is not None and x_clean.numel() == x_adv.numel() else None,
            "meta": {
                "domain": domain,
                "dataset": dataset_name,
                "model": model_name,
                "budget_label": budget_label,
            },
        }
        if save_clean and x_clean is not None:
            payload["x_clean"] = x_clean
        torch.save(payload, str(adv_file))
        adv_path = str(adv_file)

    return str(uap_path), adv_path


def _run_uap_for_case(
    *,
    out_dir: str,
    weights_cfg: Mapping[str, Any],
    domain: str,
    dataset_name: str,
    dataset_key_for_model: str,
    train_loader,
    eval_loader,
    num_classes: int,
    arch_metadata: Mapping[str, Any],
    model_names: Sequence[str],
    device: torch.device,
    budgets: List[Tuple[str, float]],
    attack_method: str,
    attack_type: str,
    loss_name: str,
    target_class_sweep: bool,
    n_target_classes: int,
    max_passes: int,
    step_size: Optional[float],
    step_fraction: float,
    target_fooling: Optional[float],
    mean_norm_batches: Optional[int],
    max_train_batches: Optional[int],
    max_eval_batches: Optional[int],
    preserve_correlation: bool,
    strict_load: bool,
    selection: str,
    checkpoint_glob: Optional[str],
    checkpoint_paths: Any,
    weights_base_dir: Path,
    weights_dir_name: str,
    applied_seed: Optional[int],
    results_rows: List[Dict[str, Any]],
    save_generated_examples: bool,
    save_generated_clean: bool,
    profile_cfg: Mapping[str, Any],
) -> None:
    label_mapping, _ = build_label_mapping(dataset_name, attack_type, num_classes=int(num_classes))
    channels = arch_metadata.get("channels")
    freq_bins = arch_metadata.get("freq_bins")
    time_steps = arch_metadata.get("time_steps")
    if channels is None or freq_bins is None or time_steps is None:
        raise RuntimeError(f"Missing input-shape metadata for UAP model build: dataset={dataset_name}")
    spec = DatasetSpec(
        "CSI-Bench" if domain == "csi_bench" else dataset_name,
        channels=int(channels),
        freq_bins=int(freq_bins),
        time_steps=int(time_steps),
    )

    for model_name in model_names:
        print(
            f"[UAP][START] domain={domain} dataset={dataset_name} model={model_name} "
            f"weights_dir={weights_dir_name} loss={loss_name} "
            f"target_class_sweep={_format_log_value(bool(target_class_sweep))} "
            f"n_target_classes={_format_log_value(int(n_target_classes))} "
            f"max_passes={_format_log_value(int(max_passes))} "
            f"step_size={_format_log_value(step_size, digits=6)} "
            f"step_fraction={_format_log_value(step_fraction, digits=6)} "
            f"target_fooling={_format_log_value(target_fooling, digits=6)} "
            f"preserve_correlation={_format_log_value(bool(preserve_correlation))}"
        )
        explicit_checkpoint = _nested_checkpoint_override(checkpoint_paths, dataset_name, model_name)
        explicit_checkpoint = ensure_optuna_clean_checkpoint(
            domain=domain,
            model_name=model_name,
            weights_dir_name=weights_dir_name,
            weights_cfg=dict(weights_cfg or {}),
            canonical_base_dir=weights_base_dir,
            alias_stems=[f"{dataset_name}_{model_name}", f"CSI-Bench_{model_name}"],
            dataset_name=(dataset_name if domain == "ntu" else None),
            task_name=(dataset_name if domain == "csi_bench" else None),
        ) or explicit_checkpoint
        model, _ = build_model_for_checkpoint(
            domain=domain,
            model_name=model_name,
            weights_dir_name=weights_dir_name,
            weights_cfg=dict(weights_cfg or {}),
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

        for budget_label, eps_2 in budgets:
            print(
                f"[UAP][START] domain={domain} dataset={dataset_name} model={model_name} "
                f"budget={budget_label} eps_2={float(eps_2):.6f} "
                f"loss={loss_name} target_class_sweep={_format_log_value(bool(target_class_sweep))} "
                f"n_target_classes={_format_log_value(int(n_target_classes))}"
            )
            delta, train_stats = _train_uap_delta(
                model=model,
                train_loader=train_loader,
                dataset_name=dataset_name,
                device=device,
                eps_2=float(eps_2),
                attack_type=attack_type,
                attack_method=attack_method,
                loss_name=loss_name,
                label_mapping=label_mapping,
                target_class_sweep=bool(target_class_sweep),
                n_target_classes=int(n_target_classes),
                max_passes=int(max_passes),
                step_size=step_size,
                step_fraction=float(step_fraction),
                target_fooling=target_fooling,
                mean_norm_batches=mean_norm_batches,
                max_train_batches=max_train_batches,
                max_eval_batches=max_eval_batches,
                preserve_correlation=bool(preserve_correlation),
            )

            asr, eval_diag, _, x_adv, x_clean, y_true, y_target = _evaluate_uap(
                model=model,
                eval_loader=eval_loader,
                delta=delta,
                device=device,
                attack_type=attack_type,
                attack_method=attack_method,
                label_mapping=label_mapping,
                target_class_sweep=bool(target_class_sweep),
                eps_2=float(eps_2),
            )

            uap_path, adv_path = _save_uap_artifacts(
                out_dir=out_dir,
                domain=domain,
                dataset_name=dataset_name,
                model_name=model_name,
                budget_label=budget_label,
                delta=delta,
                train_stats=train_stats,
                eval_diag=eval_diag,
                x_adv=x_adv,
                x_clean=x_clean,
                y_true=y_true,
                y_target=y_target,
                save_examples=bool(save_generated_examples),
                save_clean=bool(save_generated_clean),
            )

            row: Dict[str, Any] = {
                "dataset": dataset_name,
                "domain": domain,
                "model": model_name,
                "weights_dir_name": weights_dir_name,
                "checkpoint_path": str(checkpoint_path),
                "attack_method": f"uap_{attack_method}",
                "attack_mode": attack_type,
                "targeted": bool(attack_type == "targeted"),
                "budget_label": str(budget_label),
                "eps_2": float(eps_2),
                "steps": int(max_passes),
                "alpha": float(step_size) if step_size is not None else None,
                "alpha_fraction": float(step_fraction),
                "n_restarts": 1,
                "random_start": False,
                "asr": float(asr),
                "device": str(device),
                "seed": int(applied_seed) if applied_seed is not None else None,
                "uap_path": str(uap_path),
                "saved_adv_path": str(adv_path) if adv_path else None,
                "uap_base_method": attack_method,
                "uap_loss": loss_name,
                "uap_target_class_sweep": bool(target_class_sweep),
                "uap_n_target_classes": int(n_target_classes),
                "uap_preserve_correlation": bool(preserve_correlation),
                "checkpoint_missing_keys": len(load_info.get("missing_keys", [])),
                "checkpoint_unexpected_keys": len(load_info.get("unexpected_keys", [])),
                **model_profile,
                **clean,
                **train_stats,
                **eval_diag,
            }
            augment_profile_row(row, profile_cfg)
            results_rows.append(row)
            print(
                f"[UAP][DONE] domain={domain} dataset={dataset_name} model={model_name} "
                f"budget={budget_label} asr={float(asr):.4f} "
                f"n_clean_correct={int(eval_diag.get('n_clean_correct', 0))} "
                f"train_fooling={_format_log_value(train_stats.get('train_fooling_rate'))} "
                f"total_updates={_format_log_value(train_stats.get('total_updates'))} "
                f"{_format_budget_diag(eval_diag)}"
            )


def _run_ntu_uap(
    *,
    cfg: Mapping[str, Any],
    out_dir: str,
    model_names: Sequence[str],
    device: torch.device,
    attack_method: str,
    attack_type: str,
    budgets: List[Tuple[str, float]],
    loss_name: str,
    target_class_sweep: bool,
    n_target_classes: int,
    max_passes: int,
    step_size: Optional[float],
    step_fraction: float,
    target_fooling: Optional[float],
    mean_norm_batches: Optional[int],
    max_train_batches: Optional[int],
    max_eval_batches: Optional[int],
    preserve_correlation: bool,
    strict_load: bool,
    selection: str,
    weights_cfg: Mapping[str, Any],
    applied_seed: Optional[int],
    max_batches: Optional[int],
    results_rows: List[Dict[str, Any]],
    save_generated_examples: bool,
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
                f"UAP NTU ({dataset_name})",
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
                from csi.attacks.visualization import truncate_loader

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
            _run_uap_for_case(
                out_dir=out_dir,
                weights_cfg=weights_cfg,
                domain="ntu",
                dataset_name=dataset_name,
                dataset_key_for_model=dataset_name,
                train_loader=train_loader,
                eval_loader=eval_loader,
                num_classes=int(num_classes),
                arch_metadata=arch_metadata,
                model_names=model_names,
                device=device,
                budgets=budgets,
                attack_method=method_fixed,
                attack_type=mode_fixed,
                loss_name=loss_name,
                target_class_sweep=target_class_sweep,
                n_target_classes=n_target_classes,
                max_passes=max_passes,
                step_size=step_size,
                step_fraction=step_fraction,
                target_fooling=target_fooling,
                mean_norm_batches=mean_norm_batches,
                max_train_batches=max_train_batches,
                max_eval_batches=max_eval_batches,
                preserve_correlation=preserve_correlation,
                strict_load=strict_load,
                selection=selection,
                checkpoint_glob=str(checkpoint_glob) if checkpoint_glob else None,
                checkpoint_paths=checkpoint_paths,
                weights_base_dir=weights_base_dir,
                weights_dir_name=weights_dir_name,
                applied_seed=applied_seed,
                results_rows=results_rows,
                save_generated_examples=save_generated_examples,
                save_generated_clean=save_generated_clean,
                profile_cfg=profile_cfg,
            )


def _run_csi_bench_uap(
    *,
    cfg: Mapping[str, Any],
    out_dir: str,
    model_names: Sequence[str],
    device: torch.device,
    attack_method: str,
    attack_type: str,
    budgets: List[Tuple[str, float]],
    loss_name: str,
    target_class_sweep: bool,
    n_target_classes: int,
    max_passes: int,
    step_size: Optional[float],
    step_fraction: float,
    target_fooling: Optional[float],
    mean_norm_batches: Optional[int],
    max_train_batches: Optional[int],
    max_eval_batches: Optional[int],
    preserve_correlation: bool,
    strict_load: bool,
    selection: str,
    weights_cfg: Mapping[str, Any],
    applied_seed: Optional[int],
    max_batches: Optional[int],
    results_rows: List[Dict[str, Any]],
    save_generated_examples: bool,
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
                f"UAP CSI-Bench ({task_name})",
                {
                    "attack_method": method_fixed,
                    "attack_type": mode_fixed,
                    "weights_dir_name": weights_dir_name,
                    "eval_split_name": eval_split_name,
                    "device": str(device),
                },
            )

            # Need both train and eval splits for UAP training/evaluation.
            only_splits = parse_splits(["train_id", canonical_split], default=["train_id", "test_id"])
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
                only_splits=only_splits,
                splits_dir_name=splits_dir_name,
                collate_fn=filter_none_collate,
            )
            if not bundle:
                raise RuntimeError(f"Failed to load CSI-Bench task {task_name}")
            loaders = bundle["loaders"]
            train_loader = loaders.get("train")
            eval_loader = loaders.get(loader_key)
            if train_loader is None:
                raise RuntimeError(f"CSI-Bench train split missing for {task_name}")
            if eval_loader is None:
                raise RuntimeError(
                    f"Requested CSI-Bench split '{eval_split_name}' unavailable for {task_name}. "
                    f"Available={sorted(loaders.keys())}"
                )

            if max_batches is not None:
                from csi.attacks.visualization import truncate_loader

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
            _run_uap_for_case(
                out_dir=out_dir,
                weights_cfg=weights_cfg,
                domain="csi_bench",
                dataset_name=task_name,
                dataset_key_for_model="csi-bench",
                train_loader=train_loader,
                eval_loader=eval_loader,
                num_classes=int(num_classes),
                arch_metadata=arch_metadata,
                model_names=model_names,
                device=device,
                budgets=budgets,
                attack_method=method_fixed,
                attack_type=mode_fixed,
                loss_name=loss_name,
                target_class_sweep=target_class_sweep,
                n_target_classes=n_target_classes,
                max_passes=max_passes,
                step_size=step_size,
                step_fraction=step_fraction,
                target_fooling=target_fooling,
                mean_norm_batches=mean_norm_batches,
                max_train_batches=max_train_batches,
                max_eval_batches=max_eval_batches,
                preserve_correlation=preserve_correlation,
                strict_load=strict_load,
                selection=selection,
                checkpoint_glob=str(checkpoint_glob) if checkpoint_glob else None,
                checkpoint_paths=checkpoint_paths,
                weights_base_dir=weights_base_dir,
                weights_dir_name=weights_dir_name,
                applied_seed=applied_seed,
                results_rows=results_rows,
                save_generated_examples=save_generated_examples,
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
    experiment_name = str(common_cfg.get("experiment_name", "uap_apgd"))

    data_cfg = cfg.get("data", {})
    device = _resolve_device(data_cfg.get("device"))
    profile_cfg = resolve_profile_cfg(cfg)

    model_cfg = cfg.get("model", {})
    model_names = resolve_supervised_models(
        as_list(model_cfg.get("architectures") or model_cfg.get("models"), default=["ResNet18", "LeNet", "BiLSTM"])
    )

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
        raise ValueError("UAP runner currently supports attack.method in {'apgd', 'apgdt'}")

    attack_type = str(attack_cfg.get("type", default_attack_type)).strip().lower()
    if attack_type not in {"untargeted", "targeted"}:
        raise ValueError("attack.type must be one of {'untargeted', 'targeted'}")

    task_family = str(attack_cfg.get("task_family", default_task_family)).strip().lower()
    if task_family not in {"har", "hid"}:
        raise ValueError("attack.task_family must be 'har' or 'hid'.")

    loss_name_default = "dlr" if attack_type == "targeted" else "ce"
    loss_name = str(attack_cfg.get("loss", loss_name_default)).strip().lower()
    if loss_name not in {"ce", "dlr"}:
        raise ValueError("attack.loss must be one of {'ce', 'dlr'}")

    budgets = _compute_budgets(attack_cfg)

    uap_cfg = dict(cfg.get("uap", {}) or {})
    target_class_sweep = bool(uap_cfg.get("target_class_sweep", default_free_targeted if attack_method == "apgdt" else False))
    n_target_classes = int(uap_cfg.get("n_target_classes", 2 if target_class_sweep else 1))
    max_passes = int(uap_cfg.get("max_passes", 1))
    step_size_raw = uap_cfg.get("step_size")
    step_size = float(step_size_raw) if step_size_raw is not None else None
    step_fraction = float(uap_cfg.get("step_fraction", 0.1))
    target_fooling_raw = uap_cfg.get("target_fooling")
    target_fooling = float(target_fooling_raw) if target_fooling_raw is not None else None
    mean_norm_batches_raw = uap_cfg.get("mean_norm_batches", 50)
    mean_norm_batches = int(mean_norm_batches_raw) if mean_norm_batches_raw is not None else None
    max_train_batches_raw = uap_cfg.get("max_train_batches")
    max_train_batches = int(max_train_batches_raw) if max_train_batches_raw is not None else None
    max_eval_batches_raw = uap_cfg.get("max_eval_batches")
    max_eval_batches = int(max_eval_batches_raw) if max_eval_batches_raw is not None else None
    preserve_correlation = bool(uap_cfg.get("preserve_correlation", False))
    save_generated_examples = parse_bool(uap_cfg.get("save_generated_examples"), default=False)
    save_generated_clean = parse_bool(uap_cfg.get("save_generated_clean"), default=False)

    eval_cfg = dict(cfg.get("eval", {}) or {})
    max_batches_raw = eval_cfg.get("max_batches")
    try:
        max_batches = int(max_batches_raw) if max_batches_raw is not None else None
        if max_batches is not None and max_batches <= 0:
            max_batches = None
    except Exception:
        max_batches = None

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
        mode=f"uap_{attack_type}_{attack_method}",
        job_dir=str(results_job_dir) if results_job_dir else None,
        job_name=str(results_job_name) if results_job_name else None,
        job_timestamp=str(results_job_timestamp) if results_job_timestamp else None,
        relative_parts=relative_parts,
    )
    save_config(out_dir, dict(cfg))

    print_training_start(
        "UAP robustness run",
        {
            "config_path": cfg_path,
            "output_dir": out_dir,
            "attack_method": attack_method,
            "attack_type": attack_type,
            "task_family": task_family,
            "loss": loss_name,
            "target_class_sweep": target_class_sweep,
            "n_target_classes": n_target_classes,
            "budgets": [label for (label, _) in budgets],
            "max_passes": max_passes,
            "step_size": step_size,
            "step_fraction": step_fraction,
            "target_fooling": target_fooling,
            "preserve_correlation": preserve_correlation,
            "eval_n_restarts": 1,
            "eval_random_start": False,
            "device": str(device),
            "seed": applied_seed,
        },
    )

    results_rows: List[Dict[str, Any]] = []
    if "ntu" in targets:
        _run_ntu_uap(
            cfg=cfg,
            out_dir=out_dir,
            model_names=model_names,
            device=device,
            attack_method=attack_method,
            attack_type=attack_type,
            budgets=budgets,
            loss_name=loss_name,
            target_class_sweep=target_class_sweep,
            n_target_classes=n_target_classes,
            max_passes=max_passes,
            step_size=step_size,
            step_fraction=step_fraction,
            target_fooling=target_fooling,
            mean_norm_batches=mean_norm_batches,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
            preserve_correlation=preserve_correlation,
            strict_load=strict_load,
            selection=selection,
            weights_cfg=weights_cfg,
            applied_seed=applied_seed,
            max_batches=max_batches,
            results_rows=results_rows,
            save_generated_examples=save_generated_examples,
            save_generated_clean=save_generated_clean,
            profile_cfg=profile_cfg,
        )
    if "csi_bench" in targets:
        _run_csi_bench_uap(
            cfg=cfg,
            out_dir=out_dir,
            model_names=model_names,
            device=device,
            attack_method=attack_method,
            attack_type=attack_type,
            budgets=budgets,
            loss_name=loss_name,
            target_class_sweep=target_class_sweep,
            n_target_classes=n_target_classes,
            max_passes=max_passes,
            step_size=step_size,
            step_fraction=step_fraction,
            target_fooling=target_fooling,
            mean_norm_batches=mean_norm_batches,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
            preserve_correlation=preserve_correlation,
            strict_load=strict_load,
            selection=selection,
            weights_cfg=weights_cfg,
            applied_seed=applied_seed,
            max_batches=max_batches,
            results_rows=results_rows,
            save_generated_examples=save_generated_examples,
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
                "loss": loss_name,
                "budgets": [label for (label, _) in budgets],
            },
            "uap": {
                "target_class_sweep": bool(target_class_sweep),
                "n_target_classes": int(n_target_classes),
                "max_passes": int(max_passes),
                "step_size": float(step_size) if step_size is not None else None,
                "step_fraction": float(step_fraction),
                "target_fooling": float(target_fooling) if target_fooling is not None else None,
                "mean_norm_batches": int(mean_norm_batches) if mean_norm_batches is not None else None,
                "max_train_batches": int(max_train_batches) if max_train_batches is not None else None,
                "max_eval_batches": int(max_eval_batches) if max_eval_batches is not None else None,
                "eval_n_restarts": 1,
                "eval_random_start": False,
                "preserve_correlation": bool(preserve_correlation),
                "save_generated_examples": bool(save_generated_examples),
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
    print(f"[INFO] Saved UAP results: {out_dir}")


if __name__ == "__main__":
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_name=arg)
