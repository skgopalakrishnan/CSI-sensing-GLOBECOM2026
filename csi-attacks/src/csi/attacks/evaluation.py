from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import time
import numpy as np
import torch

from csi.attacks.evasion.attack_factory import generate_X_adv
from csi.attacks.utils.attack_utils import (
    CLASSES,
    build_label_mapping,
    canonical_project_name,
    enforce_har_hid_attack_policy,
)
from csi.models.metrics import compute_f1_scores


def _extract_tensor_from_batch(batch: Any) -> Optional[torch.Tensor]:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, dict):
        for value in batch.values():
            tensor = _extract_tensor_from_batch(value)
            if tensor is not None:
                return tensor
        return None
    if isinstance(batch, (list, tuple)):
        for item in batch:
            tensor = _extract_tensor_from_batch(item)
            if tensor is not None:
                return tensor
        return None
    return None


def _sample_input_tensor(loader: Any) -> Optional[torch.Tensor]:
    if loader is None:
        return None
    try:
        first_batch = next(iter(loader))
    except Exception:
        return None
    tensor = _extract_tensor_from_batch(first_batch)
    if tensor is None:
        return None
    return tensor.detach()


def _normalise_device(device: Any) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _format_log_value(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _collect_model_stats(
    model: torch.nn.Module,
    sample_tensor: Optional[torch.Tensor],
    device: Any,
    *,
    latency_warmup: int = 3,
    latency_repeats: int = 10,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}

    try:
        stats["model_param_count"] = int(sum(p.numel() for p in model.parameters()))
    except Exception:
        pass
    try:
        stats["model_param_bytes"] = int(sum(int(p.numel()) * int(p.element_size()) for p in model.parameters()))
    except Exception:
        pass

    if sample_tensor is None:
        return stats

    stats["latency_batch_size"] = int(sample_tensor.size(0)) if sample_tensor.dim() > 0 else 1

    input_shape = tuple(int(dim) for dim in sample_tensor.shape[1:])
    if input_shape:
        try:
            from ptflops import get_model_complexity_info  # type: ignore

            model_for_profile = model
            params_device = None
            try:
                params_device = next(model.parameters()).device
            except StopIteration:
                params_device = torch.device("cpu")
            if params_device.type != "cpu":
                model_for_profile = deepcopy(model).cpu()
            model_for_profile.eval()
            with torch.no_grad():
                flops_val, params_val = get_model_complexity_info(
                    model_for_profile,
                    input_shape,
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False,
                )
            try:
                if flops_val is not None:
                    stats["model_flops"] = float(flops_val)
            except Exception:
                pass
            if "model_param_count" not in stats:
                try:
                    if params_val is not None:
                        stats["model_param_count"] = int(params_val)
                except Exception:
                    pass
        except Exception:
            pass

    model_device = _normalise_device(device)
    try:
        params_device = next(model.parameters()).device
        model_device = params_device
    except StopIteration:
        pass

    sample_for_latency = sample_tensor.to(model_device, non_blocking=True)
    prev_mode = model.training
    latencies: List[float] = []
    try:
        model.eval()
        with torch.no_grad():
            for _ in range(max(0, latency_warmup)):
                _ = model(sample_for_latency)
            repeats = max(1, latency_repeats)
            for _ in range(repeats):
                if model_device.type == "cuda":
                    torch.cuda.synchronize(model_device)
                start = time.perf_counter()
                _ = model(sample_for_latency)
                if model_device.type == "cuda":
                    torch.cuda.synchronize(model_device)
                latencies.append(time.perf_counter() - start)
    except Exception:
        latencies = []
    finally:
        if prev_mode:
            model.train()

    if latencies:
        stats["latency_ms_mean"] = float(np.mean(latencies) * 1000.0)
        if len(latencies) > 1:
            stats["latency_ms_std"] = float(np.std(latencies, ddof=1) * 1000.0)
        else:
            stats["latency_ms_std"] = 0.0
        stats["latency_num_repeats"] = len(latencies)

    return stats


def _infer_num_classes(
    model: torch.nn.Module,
    sample_tensor: Optional[torch.Tensor],
    *,
    src_loader: Any,
    test_loader: Any,
    dataset_name: str,
    device: Any,
) -> int:
    """Infer number of classes from model logits, then dataset metadata/maps."""
    model_device = _normalise_device(device)
    try:
        params_device = next(model.parameters()).device
        model_device = params_device
    except StopIteration:
        pass

    if sample_tensor is not None:
        prev_mode = model.training
        try:
            model.eval()
            with torch.no_grad():
                logits = model(sample_tensor.to(model_device, non_blocking=True))
            if isinstance(logits, torch.Tensor) and logits.ndim >= 2 and int(logits.size(1)) >= 2:
                return int(logits.size(1))
        except Exception:
            pass
        finally:
            if prev_mode:
                model.train()

    for loader in (src_loader, test_loader):
        dataset = getattr(loader, "dataset", None) if loader is not None else None
        if dataset is None:
            continue
        for attr in ("num_classes", "n_classes"):
            value = getattr(dataset, attr, None)
            if isinstance(value, (int, np.integer)) and int(value) >= 2:
                return int(value)
            if isinstance(value, str):
                try:
                    parsed = int(value)
                except Exception:
                    parsed = 0
                if parsed >= 2:
                    return parsed

    canonical_name = canonical_project_name(dataset_name)
    fallback = CLASSES.get(canonical_name)
    if fallback is not None and int(fallback) >= 2:
        return int(fallback)

    raise ValueError(
        f"Could not infer num_classes for dataset '{dataset_name}'. "
        "Provide a loader with dataset.num_classes or ensure model logits are class scores."
    )


def evaluate_pgdl2_over_budgets(
    model,
    test_loader,
    device,
    *,
    dataset_name: str,
    attack_type: str,
    budgets: List[Tuple[str, float]],
    steps: int,
    alpha_fraction: Optional[float],
    alpha: Optional[float],
    random_start: bool,
    dataloader_workers: Optional[int] = None,
    n_restarts: Optional[int],
    max_batches: Optional[int],
    attack_method: str = "apgd",
    attack_kwargs: Optional[Dict[str, Any]] = None,
    profile_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    results: List[Dict[str, Any]] = []
    confusions: List[Dict[str, Any]] = []

    src_loader = test_loader
    if (max_batches is not None) and (int(max_batches) > 0):
        try:
            base_bs = getattr(test_loader, "batch_size", None) or 256
            max_n = int(max_batches) * int(base_bs)
            dataset = getattr(test_loader, "dataset", None)
            if dataset is not None:
                from torch.utils.data import DataLoader, Subset

                n_total = len(dataset)
                n_use = min(max_n, n_total)
                if n_use < n_total:
                    indices = list(range(n_use))
                    pin_memory_val: bool = (
                        device.type == "cuda"
                        if isinstance(device, torch.device)
                        else (str(device) == "cuda")
                    )
                    workers: int = (
                        int(dataloader_workers)
                        if dataloader_workers is not None
                        else int(getattr(test_loader, "num_workers", 0) or 0)
                    )
                    dl_kwargs = {
                        "batch_size": int(base_bs),
                        "shuffle": False,
                        "num_workers": workers,
                        "pin_memory": pin_memory_val,
                    }
                    if workers > 0:
                        dl_kwargs["persistent_workers"] = True
                        dl_kwargs["prefetch_factor"] = 2
                    src_loader = DataLoader(Subset(dataset, indices), **dl_kwargs)
        except Exception:
            src_loader = test_loader

    sample_tensor = _sample_input_tensor(src_loader)
    if sample_tensor is None:
        sample_tensor = _sample_input_tensor(test_loader)
    profile_sample_tensor = sample_tensor
    if isinstance(profile_sample_tensor, torch.Tensor) and profile_sample_tensor.ndim > 0 and profile_sample_tensor.size(0) > 1:
        profile_sample_tensor = profile_sample_tensor[:1].detach()
    profile_opts = dict(profile_kwargs or {})
    profile_enabled = bool(profile_opts.pop("enabled", True))
    try:
        model_stats = _collect_model_stats(
            model,
            profile_sample_tensor if profile_enabled else None,
            device,
            latency_warmup=int(profile_opts.get("latency_warmup", 3)),
            latency_repeats=int(profile_opts.get("latency_repeats", 10)),
        )
    except Exception:
        model_stats = {}

    n_classes_eval = _infer_num_classes(
        model,
        sample_tensor,
        src_loader=src_loader,
        test_loader=test_loader,
        dataset_name=dataset_name,
        device=device,
    )

    extra_attack_kwargs: Dict[str, Any] = dict(attack_kwargs or {})
    policy_enforced = bool(extra_attack_kwargs.pop("enforce_har_hid_policy", True))
    attack_type, attack_method = enforce_har_hid_attack_policy(
        dataset_name,
        attack_type,
        attack_method,
        strict=policy_enforced,
    )

    extra_attack_kwargs.setdefault("dataset_name", dataset_name)
    extra_attack_kwargs.setdefault("n_classes", int(n_classes_eval))
    for key in ("alpha_fraction", "alpha"):
        extra_attack_kwargs.pop(key, None)

    mapping, targeted = build_label_mapping(dataset_name, attack_type, num_classes=int(n_classes_eval))
    iters = len(mapping[0]) if mapping and 0 in mapping else 1

    for budget_label, eps2_val in budgets:
        budget_started_at = time.time()
        print(
            f"[EVAL][BUDGET][START] dataset={dataset_name} budget={budget_label} "
            f"eps_2={float(eps2_val):.6f} attack={attack_method}/{attack_type} "
            f"steps={int(steps)} n_restarts={int(n_restarts) if n_restarts is not None else 1} "
            f"random_start={_format_log_value(bool(random_start))} "
            f"alpha={_format_log_value(alpha, digits=6)} "
            f"alpha_fraction={_format_log_value(alpha_fraction, digits=6)} "
            f"mapping_iters={int(iters)} "
            f"target_class_sweep={_format_log_value(extra_attack_kwargs.get('target_class_sweep'))} "
            f"n_target_classes={_format_log_value(extra_attack_kwargs.get('n_target_classes'))} "
            f"loss={_format_log_value(extra_attack_kwargs.get('loss'))} "
            f"constraint_mode={_format_log_value(extra_attack_kwargs.get('constraint_mode'))}"
        )
        total_all = 0
        total_clean_correct = 0
        successes = 0
        l2_ratio_sum = 0.0
        l2_ratio_sumsq = 0.0
        l2_ratio_count = 0
        attack_success_rate_sum = 0.0
        attack_success_rate_count = 0
        attack_generation_time_sum = 0.0
        attack_generation_batches = 0
        attack_generation_examples = 0
        projection_time_sum = 0.0
        projection_calls_sum = 0
        correlation_projection_time_sum = 0.0
        pre_sum = 0.0
        post_sum = 0.0
        budget_sum = 0.0
        diag_count = 0
        diag_error: Optional[str] = None

        clean_f1_scores = {"macro": 0.0, "weighted": 0.0, "micro": 0.0}
        adv_f1_scores = dict(clean_f1_scores)

        for ctr in range(iters):
            if targeted:
                label_mapping_iter = {i: mapping[i][ctr] for i in range(len(mapping))}
            else:
                label_mapping_iter = {i: i for i in range(int(n_classes_eval))}

            adv_loader, adv_labels_test, perturbation_summary, differences = generate_X_adv(
                model,
                src_loader,
                device=device,
                attack_method=attack_method,
                epsilon=float(eps2_val),
                target_snr_db=None,
                targeted=targeted,
                verbose=False,
                label_mapping=label_mapping_iter,
                alpha_fraction=float(alpha_fraction) if alpha_fraction is not None else None,
                alpha=float(alpha) if alpha is not None else None,
                steps=steps,
                random_start=bool(random_start),
                n_restarts=int(n_restarts) if n_restarts is not None else 1,
                use_l2_ratio_metrics=True,
                **extra_attack_kwargs,
            )

            n_generated = 0
            try:
                n_generated = len(getattr(adv_loader, "dataset", []))
            except Exception:
                pass
            if differences is None or n_generated == 0:
                print(
                    f"[EVAL][WARN] dataset={dataset_name} budget={budget_label} "
                    f"iter={ctr + 1}/{iters} reason=no_adversarial_examples_generated"
                )
                continue

            x_adv_all = []
            y_true_list = []
            y_target_all = adv_labels_test.to(
                device if isinstance(device, torch.device) else torch.device(device)
            ).long()
            for xb, yb in adv_loader:
                x_adv_all.append(xb)
                y_true_list.append(yb)
            if len(x_adv_all) == 0:
                continue
            x_adv_all = torch.cat(x_adv_all, dim=0)
            y_true_all = torch.cat(y_true_list, dim=0).to(y_target_all.device).long()
            delta_all = differences.to(x_adv_all.device) if isinstance(differences, torch.Tensor) else differences
            x_clean_all_for_diag = x_adv_all - delta_all
            x_clean_all = x_clean_all_for_diag.to(y_target_all.device)
            y_clean_all = y_true_all

            with torch.no_grad():
                clean_logits = model(x_clean_all)
                adv_logits = model(x_adv_all.to(y_target_all.device))
                clean_pred = clean_logits.argmax(1)
                adv_pred = adv_logits.argmax(1)

            clean_pred_cpu = clean_pred.detach().cpu()
            adv_pred_cpu = adv_pred.detach().cpu()
            y_cpu = y_clean_all.detach().cpu()

            clean_f1_scores = compute_f1_scores(y_cpu.numpy(), clean_pred_cpu.numpy())
            adv_f1_scores = compute_f1_scores(y_cpu.numpy(), adv_pred_cpu.numpy())

            clean_mask = clean_pred == y_clean_all
            n_all = int(y_clean_all.numel())
            n_clean = int(clean_mask.sum().item())
            total_all += n_all
            total_clean_correct += n_clean

            used_attack_defined_success = False
            if targeted and isinstance(perturbation_summary, dict) and ("attack_success_mask" in perturbation_summary):
                try:
                    attack_success = np.asarray(perturbation_summary["attack_success_mask"], dtype=bool)
                    if attack_success.shape[0] == n_all:
                        success_tensor = torch.from_numpy(attack_success).to(clean_mask.device)
                        successes += int((success_tensor & clean_mask).sum().item())
                        used_attack_defined_success = True
                except Exception:
                    used_attack_defined_success = False

            if not used_attack_defined_success:
                if targeted:
                    successes += int(((adv_pred == y_target_all) & clean_mask).sum().item())
                else:
                    successes += int(((adv_pred != y_clean_all) & clean_mask).sum().item())

            if perturbation_summary and "l2_ratio_mean" in perturbation_summary:
                mean = float(perturbation_summary["l2_ratio_mean"])
                std = float(perturbation_summary.get("l2_ratio_std", 0.0))
                l2_ratio_sum += mean
                l2_ratio_sumsq += (std * std + mean * mean)
                l2_ratio_count += 1
            if perturbation_summary and (
                ("attack_defined_success_rate_all" in perturbation_summary) or ("attack_success_rate" in perturbation_summary)
            ):
                attack_success_rate_sum += float(
                    perturbation_summary.get("attack_defined_success_rate_all", perturbation_summary.get("attack_success_rate"))
                )
                attack_success_rate_count += 1
            if perturbation_summary:
                try:
                    attack_generation_time_sum += float(perturbation_summary.get("attack_generation_time_s_total", 0.0))
                except Exception:
                    pass
                try:
                    attack_generation_batches += int(perturbation_summary.get("attack_num_batches", 0))
                except Exception:
                    pass
                try:
                    attack_generation_examples += int(perturbation_summary.get("attack_num_examples", 0))
                except Exception:
                    pass
                try:
                    projection_time_sum += float(perturbation_summary.get("projection_time_s_total", 0.0))
                except Exception:
                    pass
                try:
                    projection_calls_sum += int(perturbation_summary.get("projection_calls_total", 0))
                except Exception:
                    pass
                try:
                    correlation_projection_time_sum += float(
                        perturbation_summary.get("correlation_projection_time_s_total", 0.0)
                    )
                except Exception:
                    pass

            try:
                flat_clean = x_clean_all_for_diag.view(x_clean_all_for_diag.size(0), -1)
                flat_delta = (x_adv_all - x_clean_all_for_diag).view(x_clean_all_for_diag.size(0), -1)
                pre = torch.linalg.norm(flat_clean, ord=2, dim=1).mean().item()
                post = torch.linalg.norm(flat_delta, ord=2, dim=1).mean().item()
                budget = float(eps2_val) * torch.linalg.norm(flat_clean, ord=2, dim=1).mean().item()
                pre_sum += float(pre)
                post_sum += float(post)
                budget_sum += float(budget)
                diag_count += 1
            except Exception as exc:
                if diag_error is None:
                    diag_error = f"{type(exc).__name__}: {exc}"

        asr = successes / max(1, total_clean_correct)
        diag: Dict[str, Any] = {
            "n_all": int(total_all),
            "n_clean_correct": int(total_clean_correct),
        }
        if l2_ratio_count > 0:
            mean = l2_ratio_sum / l2_ratio_count
            var = max(l2_ratio_sumsq / l2_ratio_count - mean * mean, 0.0)
            diag["l2_ratio_mean"] = mean
            diag["l2_ratio_std"] = var**0.5
        if attack_success_rate_count > 0:
            attack_defined_success = attack_success_rate_sum / attack_success_rate_count
            diag["attack_defined_success_rate_all_mean"] = attack_defined_success
            diag["attack_success_rate_mean"] = attack_defined_success
        if diag_count > 0:
            diag["pre_l2_mean"] = pre_sum / diag_count
            diag["post_l2_mean"] = post_sum / diag_count
            diag["budget_l2_mean"] = budget_sum / diag_count
            denom = max(diag["budget_l2_mean"], 1e-12)
            diag["utilization_mean"] = diag["post_l2_mean"] / denom
        elif diag_error is not None:
            diag["budget_diag_error"] = diag_error
        if attack_generation_batches > 0 or attack_generation_examples > 0 or attack_generation_time_sum > 0.0:
            diag["attack_generation_time_s_total"] = float(attack_generation_time_sum)
            diag["attack_num_batches"] = int(attack_generation_batches)
            diag["attack_num_examples"] = int(attack_generation_examples)
            if attack_generation_batches > 0:
                diag["attack_generation_time_s_per_batch_mean"] = float(attack_generation_time_sum / attack_generation_batches)
            if attack_generation_examples > 0:
                diag["attack_generation_time_per_example_ms"] = float(
                    (attack_generation_time_sum * 1000.0) / attack_generation_examples
                )
        if projection_calls_sum > 0 or projection_time_sum > 0.0:
            diag["projection_time_s_total"] = float(projection_time_sum)
            diag["projection_calls_total"] = int(projection_calls_sum)
            if projection_calls_sum > 0:
                diag["projection_time_per_call_ms"] = float((projection_time_sum * 1000.0) / projection_calls_sum)
            if attack_generation_examples > 0:
                diag["projection_time_per_example_ms"] = float((projection_time_sum * 1000.0) / attack_generation_examples)
            if attack_generation_time_sum > 0.0:
                diag["projection_time_share"] = float(projection_time_sum / attack_generation_time_sum)
        if correlation_projection_time_sum > 0.0:
            diag["correlation_projection_time_s_total"] = float(correlation_projection_time_sum)
            if attack_generation_time_sum > 0.0:
                diag["correlation_projection_time_share"] = float(
                    correlation_projection_time_sum / attack_generation_time_sum
                )

        if model_stats:
            diag.update(model_stats)

        res: Dict[str, Any] = {
            "budget_label": budget_label,
            "eps_2": float(eps2_val),
            "asr": float(asr),
            "diag": diag,
            "clean_f1_macro": float(clean_f1_scores["macro"]),
            "clean_f1_weighted": float(clean_f1_scores["weighted"]),
            "clean_f1_micro": float(clean_f1_scores["micro"]),
            "adv_f1_macro": float(adv_f1_scores["macro"]),
            "adv_f1_weighted": float(adv_f1_scores["weighted"]),
            "adv_f1_micro": float(adv_f1_scores["micro"]),
        }
        if model_stats:
            res.update(model_stats)

        results.append(res)
        elapsed_s = time.time() - budget_started_at
        print(
            f"[EVAL][BUDGET][DONE] dataset={dataset_name} budget={budget_label} "
            f"eps_2={float(eps2_val):.6f} asr={float(asr):.4f} "
            f"n_clean_correct={int(diag.get('n_clean_correct', 0))} "
            f"attack_defined_success_all={_format_log_value(diag.get('attack_defined_success_rate_all_mean'))} "
            f"l2_ratio={_format_log_value(diag.get('l2_ratio_mean'))} "
            f"utilization={_format_log_value(diag.get('utilization_mean'))} "
            f"post_l2={_format_log_value(diag.get('post_l2_mean'))} "
            f"budget_l2={_format_log_value(diag.get('budget_l2_mean'))} "
            f"proj_share={_format_log_value(diag.get('projection_time_share'))} "
            f"corr_proj_share={_format_log_value(diag.get('correlation_projection_time_share'))} "
            f"elapsed_s={elapsed_s:.1f}"
        )
        if diag_error is not None and diag_count == 0:
            print(
                f"[EVAL][WARN] dataset={dataset_name} budget={budget_label} "
                f"reason=budget_diagnostics_failed error={diag_error}"
            )

    return results, confusions


def evaluate_asr_pgdl2_untargeted(
    model,
    test_loader,
    device,
    *,
    attack_method: str,
    steps: int,
    eps_2: float,
    alpha_fraction: Optional[float],
    alpha: Optional[float],
    random_start: bool,
    dataloader_workers: Optional[int],
    n_restarts: int,
        max_batches: Optional[int],
        dataset_name: str,
        attack_kwargs: Optional[Dict[str, Any]] = None,
        profile_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    res_list, _ = evaluate_pgdl2_over_budgets(
        model,
        test_loader,
        device,
        dataset_name=dataset_name,
        attack_type="untargeted",
        budgets=[("eps2", float(eps_2))],
        steps=steps,
        alpha_fraction=alpha_fraction,
        alpha=alpha,
        random_start=random_start,
        dataloader_workers=dataloader_workers,
        n_restarts=n_restarts,
        max_batches=max_batches,
        attack_method=attack_method,
        attack_kwargs=attack_kwargs,
        profile_kwargs=profile_kwargs,
    )
    if not res_list:
        return 0.0, {"n_all": 0, "n_clean_correct": 0}

    res = res_list[0]
    diag = res.get("diag", {})
    if not isinstance(diag, dict):
        diag = {}

    for key in (
        "model_param_count",
        "model_flops",
        "latency_ms_mean",
        "latency_ms_std",
        "latency_batch_size",
        "latency_num_repeats",
        "clean_f1_macro",
        "clean_f1_weighted",
        "clean_f1_micro",
        "adv_f1_macro",
        "adv_f1_weighted",
        "adv_f1_micro",
    ):
        if key in res and key not in diag:
            diag[key] = res[key]

    asr_value = res.get("asr", 0.0)
    if not isinstance(asr_value, (int, float)):
        asr_value = 0.0

    def _normalise_diag_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return float(value)
        return value

    return float(asr_value), {k: _normalise_diag_value(v) for k, v in diag.items()}


__all__ = [
    "evaluate_pgdl2_over_budgets",
    "evaluate_asr_pgdl2_untargeted",
    "_collect_model_stats",
    "_sample_input_tensor",
]
