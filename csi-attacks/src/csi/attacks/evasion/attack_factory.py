import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..utils.attack_utils import eps_l2_from_snr_db
from .apgd import APGD
from .apgdt import APGDT


def _normalize_attack_method_name(attack_method: str) -> str:
    return str(attack_method).strip().lower()


def _restart_objective(
    atk: Any,
    attack_method_norm: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_labels: torch.Tensor,
    effective_targeted: bool,
    ce_none: nn.CrossEntropyLoss,
) -> torch.Tensor:
    if attack_method_norm in {"apgd", "apgdt"}:
        loss_name = str(getattr(atk, "loss", "ce")).strip().lower()
        if loss_name == "dlr":
            if effective_targeted and hasattr(atk, "_dlr_loss_targeted"):
                return atk._dlr_loss_targeted(logits, labels, target_labels)
            if (not effective_targeted) and hasattr(atk, "_dlr_loss"):
                return atk._dlr_loss(logits, labels)

    if effective_targeted:
        return -ce_none(logits, target_labels)
    return ce_none(logits, labels)


def _resolve_physical_mode(kwargs: Dict[str, Any]) -> Tuple[bool, Optional[str], bool, float]:
    physical_mode = str(kwargs.pop("physical_mode", kwargs.pop("constraint_mode", "none"))).strip().lower()
    preserve_correlation = bool(
        kwargs.pop(
            "preserve_correlation",
            physical_mode in {"corr", "correlation", "physical", "constrained"},
        )
    )
    dataset_name = kwargs.pop("dataset_name", None)
    dataset_name = None if dataset_name is None else str(dataset_name)
    use_mmd_loss = bool(kwargs.pop("use_mmd_loss", False))
    mmd_weight = float(kwargs.pop("mmd_weight", 0.05))
    return preserve_correlation, dataset_name, use_mmd_loss, mmd_weight


def _reset_attack_runtime_metrics(atk: Any) -> None:
    resetter = getattr(atk, "reset_runtime_metrics", None)
    if callable(resetter):
        try:
            resetter()
            return
        except Exception:
            pass
    if hasattr(atk, "_runtime_metrics"):
        try:
            setattr(atk, "_runtime_metrics", {})
        except Exception:
            pass


def _get_attack_runtime_metrics(atk: Any) -> Dict[str, Any]:
    getter = getattr(atk, "get_runtime_metrics", None)
    if callable(getter):
        try:
            metrics = getter()
            if isinstance(metrics, dict):
                return dict(metrics)
        except Exception:
            pass
    raw_metrics = getattr(atk, "_runtime_metrics", None)
    if isinstance(raw_metrics, dict):
        return dict(raw_metrics)
    return {}


def create_attack(
    model,
    tensor_loader,
    device: Union[str, torch.device] = "cuda",
    attack_method="apgd",
    epsilon=0.01,
    target_snr_db=None,
    targeted=False,
    **kwargs,
):
    verbose = kwargs.pop("verbose", False)
    attack_method_norm = _normalize_attack_method_name(attack_method)

    if isinstance(device, torch.device):
        device_obj = device
    else:
        device_obj = torch.device(device)

    if target_snr_db is not None:
        _ = next(iter(tensor_loader))
        epsilon = eps_l2_from_snr_db(target_snr_db)
        if verbose:
            print(f"  Calculated epsilon (L2 ratio)={epsilon:.6f} for target SNR={target_snr_db}dB")
    elif epsilon is None:
        raise ValueError("Either epsilon or target_snr_db must be provided")

    alpha_provided = "alpha" in kwargs
    alpha_ = kwargs.pop("alpha", 0.05)
    alpha_fraction_ = kwargs.pop("alpha_fraction", None)
    steps_ = int(kwargs.pop("steps", 40))
    random_start_ = bool(kwargs.pop("random_start", False))
    norm_ = kwargs.pop("norm", "L2")
    n_restarts_inner = int(
        kwargs.pop(
            "attack_n_restarts",
            kwargs.pop("n_restarts_inner", kwargs.pop("internal_n_restarts", 1)),
        )
        or 1
    )
    seed_ = int(kwargs.pop("seed", 0))
    n_classes_ = int(kwargs.pop("n_classes", kwargs.pop("num_classes", 10)))
    eot_iter_ = int(kwargs.pop("eot_iter", 1))
    rho_ = float(kwargs.pop("rho", 0.75))

    use_l2_ratio_metrics = bool(kwargs.pop("use_l2_ratio_metrics", False))
    report_budget_utilization = bool(kwargs.pop("report_budget_utilization", False))
    if use_l2_ratio_metrics:
        report_budget_utilization = True

    preserve_correlation, dataset_name, use_mmd_loss, mmd_weight = _resolve_physical_mode(kwargs)

    targeted_capable_methods = {"apgd", "apgdt"}
    uses_target_labels = bool(targeted and attack_method_norm in targeted_capable_methods)

    if targeted and attack_method_norm not in targeted_capable_methods:
        raise ValueError(f"Attack '{attack_method_norm}' does not support targeted mode.")
    if attack_method_norm == "apgd":
        loss_name = kwargs.pop("loss", "ce")
        apgd_alpha = alpha_ if alpha_provided else None
        atk = APGD(
            model,
            norm=norm_,
            eps=epsilon,
            steps=steps_,
            n_restarts=n_restarts_inner,
            seed=seed_,
            loss=loss_name,
            eot_iter=eot_iter_,
            rho=rho_,
            verbose=verbose,
            random_start=random_start_,
            alpha=apgd_alpha,
            alpha_fraction=alpha_fraction_,
            dataset_name=dataset_name,
            preserve_correlation=preserve_correlation,
            use_mmd_loss=use_mmd_loss,
            mmd_weight=mmd_weight,
        )

    elif attack_method_norm == "apgdt":
        loss_name = kwargs.pop("loss", "dlr")
        apgdt_alpha = alpha_ if alpha_provided else None
        target_class_sweep_ = bool(kwargs.pop("target_class_sweep", kwargs.pop("sweep_targets", True)))
        n_target_classes_default = 2 if target_class_sweep_ else max(int(n_classes_) - 1, 1)
        n_target_classes_ = int(
            kwargs.pop(
                "n_target_classes",
                kwargs.pop("target_top_k", kwargs.pop("sweep_top_k", n_target_classes_default)),
            )
            or n_target_classes_default
        )
        atk = APGDT(
            model,
            norm=norm_,
            eps=epsilon,
            steps=steps_,
            n_restarts=n_restarts_inner,
            seed=seed_,
            eot_iter=eot_iter_,
            rho=rho_,
            verbose=verbose,
            n_classes=n_classes_,
            n_target_classes=n_target_classes_,
            random_start=random_start_,
            alpha=apgdt_alpha,
            alpha_fraction=alpha_fraction_,
            loss=loss_name,
            target_class_sweep=target_class_sweep_,
            dataset_name=dataset_name,
            preserve_correlation=preserve_correlation,
            use_mmd_loss=use_mmd_loss,
            mmd_weight=mmd_weight,
        )
        if target_class_sweep_:
            uses_target_labels = False

    else:
        raise NotImplementedError(
            f"Attack type '{attack_method_norm}' is not implemented. "
            "Use one of: apgd, apgdt."
        )

    try:
        has_rnn = any(isinstance(m, (nn.RNN, nn.GRU, nn.LSTM)) for m in model.modules())
    except Exception:
        has_rnn = False
    if has_rnn:
        atk.set_model_training_mode(model_training=True, batchnorm_training=False, dropout_training=False)
        if verbose:
            print("  Note: Enabled training mode during attack for RNN compatibility (BN/Dropout kept eval).")

    if targeted:
        try:
            atk.set_mode_targeted_by_label(quiet=True)
        except TypeError:
            atk.set_mode_targeted_by_label()

    if report_budget_utilization:
        setattr(atk, "report_budget_utilization", True)

    setattr(atk, "_expects_target_labels", uses_target_labels)
    setattr(atk, "_targeted_success_is_attack_defined", bool(attack_method_norm == "apgdt" and not uses_target_labels))

    if kwargs and verbose:
        print(f"  Ignored unused attack kwargs for {attack_method_norm}: {sorted(kwargs.keys())}")

    _ = device_obj
    return atk


def generate_X_adv(
    model,
    tensor_loader,
    label_mapping: dict,
    device: Union[str, torch.device] = "cuda",
    attack_method="apgd",
    epsilon=0.01,
    target_snr_db=None,
    targeted=False,
    verbose=False,
    **kwargs,
):
    use_l2_ratio_metrics = bool(kwargs.get("use_l2_ratio_metrics", False))
    attack_method_norm = _normalize_attack_method_name(attack_method)

    n_restarts = int(kwargs.pop("n_restarts", 1) or 1)
    local_kwargs = dict(kwargs)
    inner_restarts_explicit = (
        ("attack_n_restarts" in local_kwargs)
        or ("n_restarts_inner" in local_kwargs)
        or ("internal_n_restarts" in local_kwargs)
    )
    if attack_method_norm in {"apgd", "apgdt"} and (n_restarts > 1) and (not inner_restarts_explicit):
        local_kwargs["n_restarts_inner"] = int(n_restarts)
        n_restarts = 1
        if verbose:
            print(
                f"  Using attack-internal restarts for {attack_method_norm}: "
                f"n_restarts_inner={local_kwargs['n_restarts_inner']}"
            )

    inner_restarts_hint = int(
        local_kwargs.get(
            "attack_n_restarts",
            local_kwargs.get("n_restarts_inner", local_kwargs.get("internal_n_restarts", 1)),
        )
        or 1
    )
    if n_restarts > 1 or inner_restarts_hint > 1:
        local_kwargs["random_start"] = True

    atk = create_attack(
        model,
        tensor_loader,
        device=device,
        attack_method=attack_method,
        epsilon=epsilon,
        target_snr_db=target_snr_db,
        targeted=targeted,
        verbose=verbose,
        **local_kwargs,
    )
    effective_targeted = bool(targeted and getattr(atk, "_expects_target_labels", False))
    attack_defined_targeted_success = bool(targeted and getattr(atk, "_targeted_success_is_attack_defined", False))

    if isinstance(device, torch.device):
        device_obj = device
        device_str = device.type
    else:
        device_str = device
        device_obj = torch.device(device)

    X_adv = []
    labels_test = []
    adv_labels_test = []
    perturbations = []
    differences = []

    label_mapping_dict: Dict[int, Any] = {}
    target_indices: Dict[int, int] = {}
    attack_generation_times: List[float] = []
    projection_times: List[float] = []
    projection_calls: List[int] = []
    correlation_projection_times: List[float] = []
    attack_batch_sizes: List[int] = []
    if label_mapping is not None:
        for key, value in label_mapping.items():
            key_int = int(key)
            label_mapping_dict[key_int] = value
            if effective_targeted and isinstance(value, (list, tuple)):
                target_indices[key_int] = 0

    for i, data in enumerate(tensor_loader):
        if verbose and (i % 20 == 0):
            print(f"  Processing batch {i + 1}/{len(tensor_loader)}")

        inputs, labels = data
        mapped_list = []
        labels_cpu = labels.cpu()
        for l in labels_cpu:
            class_id = int(l.item())
            entry = label_mapping_dict.get(class_id, class_id)
            mapped_value = class_id
            if isinstance(entry, (list, tuple)):
                if entry:
                    if effective_targeted:
                        idx = target_indices.get(class_id, 0)
                        mapped_value = int(entry[idx % len(entry)])
                        target_indices[class_id] = idx + 1
                    else:
                        mapped_value = int(entry[0])
            else:
                mapped_value = int(entry)
            mapped_list.append(mapped_value)

        inputs = inputs.to(device_obj, non_blocking=True)
        labels = labels.to(device_obj, non_blocking=True).long()
        adv_labels = torch.tensor(mapped_list, dtype=torch.long, device=device_obj)
        labels_for_attack = adv_labels if bool(getattr(atk, "_expects_target_labels", False)) else labels

        ce_none = nn.CrossEntropyLoss(reduction="none")
        best_adv = None
        best_obj = None
        success_any = torch.zeros(labels.size(0), dtype=torch.bool, device=labels.device)
        _reset_attack_runtime_metrics(atk)
        batch_attack_started_at = time.perf_counter()

        for _ in range(n_restarts):
            if attack_method_norm in {"apgd", "apgdt"} and hasattr(atk, "perturb"):
                robust_r, x_adv_r = atk.perturb(inputs, labels_for_attack, cheap=True)
                success_r = ~robust_r
            else:
                x_adv_r = atk(inputs, labels_for_attack)
                with torch.no_grad():
                    preds_r = model(x_adv_r).argmax(1)
                if effective_targeted:
                    success_r = preds_r == adv_labels
                else:
                    success_r = preds_r != labels

            with torch.no_grad():
                logits_r = model(x_adv_r)
                obj_r = _restart_objective(
                    atk=atk,
                    attack_method_norm=attack_method_norm,
                    logits=logits_r,
                    labels=labels,
                    target_labels=adv_labels,
                    effective_targeted=effective_targeted,
                    ce_none=ce_none,
                )

            if best_adv is None:
                best_adv = x_adv_r.detach()
                best_obj = obj_r.detach()
            else:
                mask_obj = obj_r > best_obj
                mask_obj = mask_obj & (~success_any)
                if mask_obj.any():
                    best_adv = best_adv.clone()
                    best_obj = best_obj.clone()
                    best_adv[mask_obj] = x_adv_r.detach()[mask_obj]
                    best_obj[mask_obj] = obj_r.detach()[mask_obj]

            if success_r.any():
                best_adv = best_adv.clone()
                best_adv[success_r] = x_adv_r.detach()[success_r]
                success_any = success_any | success_r

        batch_attack_elapsed = time.perf_counter() - batch_attack_started_at
        batch_runtime = _get_attack_runtime_metrics(atk)
        attack_generation_times.append(float(batch_attack_elapsed))
        attack_batch_sizes.append(int(labels.size(0)))
        projection_times.append(float(batch_runtime.get("projection_time_s_total", 0.0) or 0.0))
        projection_calls.append(int(batch_runtime.get("projection_calls_total", 0) or 0))
        correlation_projection_times.append(
            float(batch_runtime.get("correlation_projection_time_s_total", 0.0) or 0.0)
        )

        x_adv = best_adv if best_adv is not None else atk(inputs, labels_for_attack)

        with torch.no_grad():
            delta = x_adv - inputs
            flat_delta = delta.view(delta.size(0), -1)
            l2_perturbation = torch.linalg.norm(flat_delta, ord=2, dim=1).detach().cpu().numpy()
            linf_perturbation = torch.linalg.norm(flat_delta, ord=float("inf"), dim=1).detach().cpu().numpy()

            perturbation_metrics: Dict[str, Any] = {
                "l2": l2_perturbation,
                "linf": linf_perturbation,
            }
            if use_l2_ratio_metrics:
                sig_norm = torch.linalg.norm(inputs.view(inputs.size(0), -1), ord=2, dim=1)
                denom = (sig_norm + 1e-12).detach().cpu().numpy()
                perturbation_metrics["l2_ratio"] = l2_perturbation / denom

            if targeted and (attack_defined_targeted_success or effective_targeted):
                perturbation_metrics["attack_success_mask"] = success_any.detach().cpu().numpy().astype(bool)

            X_adv.append(x_adv.detach().cpu())
            labels_test.append(labels.cpu())
            adv_labels_test.append(adv_labels.cpu())
            perturbations.append(perturbation_metrics)
            differences.append(delta.detach().cpu())

        if i % 50 == 0 and device_str == "cuda":
            torch.cuda.empty_cache()

    if len(X_adv) == 0:
        X_adv = torch.empty(0)
        labels_test = torch.empty(0, dtype=torch.long)
        adv_labels_test = torch.empty(0, dtype=torch.long)
        perturbation_summary: Dict[str, Any] = {
            "l2_mean": 0.0,
            "l2_std": 0.0,
            "linf_mean": 0.0,
            "linf_std": 0.0,
            "attack_generation_time_s_total": 0.0,
            "attack_generation_time_s_per_batch_mean": 0.0,
            "attack_generation_time_per_example_ms": 0.0,
            "attack_num_examples": 0,
            "attack_num_batches": 0,
            "projection_time_s_total": 0.0,
            "projection_calls_total": 0,
            "projection_time_per_call_ms": 0.0,
            "projection_time_per_example_ms": 0.0,
            "projection_time_share": 0.0,
            "correlation_projection_time_s_total": 0.0,
            "correlation_projection_time_share": 0.0,
        }
        differences = None
        if verbose:
            print("  No batches processed; returning empty adversarial set.")
        adv_test_set = torch.utils.data.TensorDataset(X_adv, labels_test)
        adv_test_loader = torch.utils.data.DataLoader(adv_test_set, batch_size=256, shuffle=False)
        return adv_test_loader, adv_labels_test, perturbation_summary, differences

    X_adv = torch.cat(X_adv, dim=0)
    labels_test = torch.cat(labels_test, dim=0)
    adv_labels_test = torch.cat(adv_labels_test, dim=0)

    l2_perturbations = np.concatenate([p["l2"] for p in perturbations if p.get("l2") is not None])
    linf_perturbations = np.concatenate([p["linf"] for p in perturbations if p.get("linf") is not None])

    perturbation_summary = {
        "l2_mean": float(np.mean(l2_perturbations)) if l2_perturbations.size > 0 else 0.0,
        "l2_std": float(np.std(l2_perturbations)) if l2_perturbations.size > 0 else 0.0,
        "linf_mean": float(np.mean(linf_perturbations)) if linf_perturbations.size > 0 else 0.0,
        "linf_std": float(np.std(linf_perturbations)) if linf_perturbations.size > 0 else 0.0,
    }

    if use_l2_ratio_metrics and len(perturbations) > 0 and ("l2_ratio" in perturbations[0]):
        l2_ratios = np.concatenate([p["l2_ratio"] for p in perturbations])
        perturbation_summary["l2_ratio_mean"] = float(np.mean(l2_ratios)) if l2_ratios.size > 0 else 0.0
        perturbation_summary["l2_ratio_std"] = float(np.std(l2_ratios)) if l2_ratios.size > 0 else 0.0

    if len(perturbations) > 0 and ("attack_success_mask" in perturbations[0]):
        success_mask = np.concatenate([p["attack_success_mask"] for p in perturbations]).astype(bool)
        attack_defined_success = float(success_mask.mean()) if success_mask.size > 0 else 0.0
        perturbation_summary["attack_success_mask"] = success_mask
        perturbation_summary["attack_defined_success_rate_all"] = attack_defined_success
        perturbation_summary["attack_success_rate"] = attack_defined_success

    attack_generation_time_total = float(sum(attack_generation_times))
    attack_num_batches = int(len(attack_batch_sizes))
    attack_num_examples = int(sum(attack_batch_sizes))
    projection_time_total = float(sum(projection_times))
    projection_calls_total = int(sum(projection_calls))
    correlation_projection_time_total = float(sum(correlation_projection_times))

    perturbation_summary["attack_generation_time_s_total"] = attack_generation_time_total
    perturbation_summary["attack_num_examples"] = attack_num_examples
    perturbation_summary["attack_num_batches"] = attack_num_batches
    if attack_num_batches > 0:
        perturbation_summary["attack_generation_time_s_per_batch_mean"] = float(
            attack_generation_time_total / attack_num_batches
        )
    if attack_num_examples > 0:
        perturbation_summary["attack_generation_time_per_example_ms"] = float(
            (attack_generation_time_total * 1000.0) / attack_num_examples
        )

    perturbation_summary["projection_time_s_total"] = projection_time_total
    perturbation_summary["projection_calls_total"] = projection_calls_total
    if projection_calls_total > 0:
        perturbation_summary["projection_time_per_call_ms"] = float(
            (projection_time_total * 1000.0) / projection_calls_total
        )
    if attack_num_examples > 0:
        perturbation_summary["projection_time_per_example_ms"] = float(
            (projection_time_total * 1000.0) / attack_num_examples
        )
    if attack_generation_time_total > 0.0:
        perturbation_summary["projection_time_share"] = float(projection_time_total / attack_generation_time_total)
        perturbation_summary["correlation_projection_time_share"] = float(
            correlation_projection_time_total / attack_generation_time_total
        )
    perturbation_summary["correlation_projection_time_s_total"] = correlation_projection_time_total

    differences = torch.cat(differences, dim=0) if len(differences) > 0 else None

    if verbose:
        print(f"  Generated {X_adv.shape[0]} adversarial examples")

    try:
        base_bs = getattr(tensor_loader, "batch_size", None) or 256
    except Exception:
        base_bs = 256
    try:
        workers = int(getattr(tensor_loader, "num_workers", 0) or 0)
    except Exception:
        workers = 0

    pin_memory = device_str == "cuda"
    dl_kwargs = {"batch_size": base_bs, "shuffle": False, "pin_memory": pin_memory, "num_workers": workers}
    if workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2
    adv_test_set = torch.utils.data.TensorDataset(X_adv, labels_test)
    adv_test_loader = torch.utils.data.DataLoader(adv_test_set, **dl_kwargs)

    return adv_test_loader, adv_labels_test, perturbation_summary, differences
