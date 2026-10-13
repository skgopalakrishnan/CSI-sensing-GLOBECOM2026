import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import Attack
from ..utils.correlation_utils import process_csi_tensor_for_correlation
from ..utils.attack_utils import (
    eps_abs_from_l2_ratio,
    project_l2_ratio,
    sample_l2_ratio_ball,
    view_as_broadcast,
)


class APGD(Attack):
    r"""
    APGD-L2 adapted for CSI tensors with per-sample ratio budgets.

    eps semantics:
        - `eps` is treated as eps_2 (dimensionless L2 ratio).
        - Absolute per-sample budget: eps_abs[b] = eps_2 * ||x_b||_2.
        - Compatible with SNR parameterization via eps_2 = 10^(-SNR_dB/20).

    Notes:
        - Supports arbitrary CSI tensor rank `(B, ...)` (not only images).
        - No `[0, 1]` clamping is applied.
        - Projection is always onto the per-sample L2-ratio ball around clean inputs.
    """

    def __init__(
        self,
        model,
        norm: str = "L2",
        eps: float = 8 / 255,
        steps: int = 10,
        n_restarts: int = 1,
        seed: int = 0,
        loss: str = "ce",
        eot_iter: int = 1,
        rho: float = 0.75,
        verbose: bool = False,
        random_start: bool = True,
        alpha: Optional[float] = 0.05,
        alpha_fraction: Optional[float] = None,
        eps_for_division: float = 1e-10,
        dataset_name: Optional[str] = None,
        preserve_correlation: bool = False,
        use_mmd_loss: bool = False,
        mmd_weight: float = 0.0,
    ):
        super().__init__("APGD", model)

        norm_norm = str(norm).strip().upper()
        self.norm = "L2" if norm_norm in {"L2", "2"} else norm_norm
        if self.norm != "L2":
            raise ValueError(f"APGD for CSI currently supports only norm='L2', got {norm!r}.")

        self.eps_2 = float(eps)
        self.steps = int(steps)
        self.n_restarts = int(n_restarts)
        self.seed = int(seed)
        self.loss = str(loss).strip().lower()
        if self.loss not in {"ce", "dlr"}:
            raise ValueError(f"Unsupported APGD loss {loss!r}; choose 'ce' or 'dlr'.")

        self.eot_iter = int(eot_iter)
        self.thr_decr = float(rho)
        self.verbose = bool(verbose)
        self.random_start = bool(random_start)
        self.alpha = None if alpha is None else float(alpha)
        self.alpha_fraction = None if alpha_fraction is None else float(alpha_fraction)
        self.eps_for_division = float(eps_for_division)
        self.dataset_name = None if dataset_name is None else str(dataset_name)
        self.preserve_correlation = bool(preserve_correlation)
        self.use_mmd_loss = bool(use_mmd_loss)
        self.mmd_weight = float(mmd_weight)
        if self.preserve_correlation and not self.dataset_name:
            raise ValueError("APGD with preserve_correlation=True requires dataset_name.")

        self.supported_mode = ["default", "targeted"]
        self._debug_checks = os.environ.get("CSI_VALIDATE_LOGITS", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._runtime_metrics: Dict[str, float] = {
            "projection_time_s_total": 0.0,
            "projection_calls_total": 0,
            "correlation_projection_time_s_total": 0.0,
        }

    def reset_runtime_metrics(self) -> None:
        self._runtime_metrics = {
            "projection_time_s_total": 0.0,
            "projection_calls_total": 0,
            "correlation_projection_time_s_total": 0.0,
        }

    def get_runtime_metrics(self) -> dict:
        return dict(self._runtime_metrics)

    @staticmethod
    def _view_as_broadcast(vec: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return view_as_broadcast(vec, like)

    def _eps_abs(self, x: torch.Tensor) -> torch.Tensor:
        return eps_abs_from_l2_ratio(x, self.eps_2)

    def _project_delta(self, delta: torch.Tensor, x_original: torch.Tensor) -> torch.Tensor:
        project_started_at = time.perf_counter()
        corr_elapsed_s = 0.0
        if self.preserve_correlation:
            corr_started_at = time.perf_counter()
            delta = process_csi_tensor_for_correlation(
                delta,
                dataset_name=str(self.dataset_name),
                clean_tensor=x_original,
            )
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            corr_elapsed_s = time.perf_counter() - corr_started_at

        delta = project_l2_ratio(
            delta,
            x_original,
            eps_2=self.eps_2,
            eps_for_division=self.eps_for_division,
        )
        # Numerical guard: enforce strict in-budget perturbations.
        b = delta.size(0)
        l2_post = torch.norm(delta.reshape(b, -1), p=2, dim=1)
        budgets = self.eps_2 * torch.norm(x_original.reshape(b, -1), p=2, dim=1)
        ratio = (budgets.double() / (l2_post.double() + self.eps_for_division)).clamp(max=1.0)
        ratio = ratio * (1.0 - 1e-6)
        delta = delta * ratio.to(delta.dtype).view(ratio.size(0), *([1] * (delta.dim() - 1)))
        project_elapsed_s = time.perf_counter() - project_started_at
        self._runtime_metrics["projection_time_s_total"] += float(project_elapsed_s)
        self._runtime_metrics["projection_calls_total"] += 1
        if corr_elapsed_s > 0.0:
            self._runtime_metrics["correlation_projection_time_s_total"] += float(corr_elapsed_s)
        return torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def calculate_mmd_loss(X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        batch_size = X.size(0)
        X_flat = X.reshape(batch_size, -1)
        Y_flat = Y.reshape(batch_size, -1)
        XX = torch.cdist(X_flat, X_flat, p=2) ** 2
        YY = torch.cdist(Y_flat, Y_flat, p=2) ** 2
        XY = torch.cdist(X_flat, Y_flat, p=2) ** 2
        gamma = 1.0 / (2 * sigma ** 2)
        K_XX = torch.exp(-gamma * XX)
        K_YY = torch.exp(-gamma * YY)
        K_XY = torch.exp(-gamma * XY)
        diagonal_mask = torch.eye(batch_size, device=K_XX.device, dtype=torch.bool)
        K_XX_masked = K_XX.masked_fill(diagonal_mask, 0)
        K_YY_masked = K_YY.masked_fill(diagonal_mask, 0)
        m = float(batch_size)
        return (
            K_XX_masked.sum() / (m * (m - 1))
            + K_YY_masked.sum() / (m * (m - 1))
            - 2 * K_XY.sum() / (m * m)
        )

    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        norm = torch.norm(x.reshape(b, -1), p=2, dim=1).clamp_min(self.eps_for_division)
        return x / self._view_as_broadcast(norm, x)

    def _dlr_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x_sorted, ind_sorted = logits.sort(dim=1)
        ind = (ind_sorted[:, -1] == labels).float()
        if x_sorted.size(1) >= 3:
            denom = (x_sorted[:, -1] - x_sorted[:, -3]).clamp_min(1e-12)
        elif x_sorted.size(1) == 2:
            denom = (x_sorted[:, -1] - x_sorted[:, -2]).clamp_min(1e-12)
        else:
            denom = torch.ones_like(labels, dtype=logits.dtype, device=logits.device)
        second_best = x_sorted[:, -2] if x_sorted.size(1) >= 2 else x_sorted[:, -1]
        return -(
            logits[torch.arange(logits.size(0), device=logits.device), labels]
            - second_best * ind
            - x_sorted[:, -1] * (1.0 - ind)
        ) / denom

    def _dlr_loss_targeted(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Canonical targeted DLR objective for APGD-T:
            (z_y - z_t) / (z_max - z_second_max)
        We return its negative because APGD maximizes per-sample objective.
        """
        row_idx = torch.arange(logits.size(0), device=logits.device)
        z_y = logits[row_idx, labels]
        z_t = logits[row_idx, target_labels]
        top2 = logits.topk(k=min(2, logits.size(1)), dim=1).values
        if top2.size(1) >= 2:
            denom = (top2[:, 0] - top2[:, 1]).clamp_min(1e-12)
        else:
            denom = torch.ones_like(z_y)
        return -((z_y - z_t) / denom)

    def _loss_per_sample(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.loss == "ce":
            ce = nn.CrossEntropyLoss(reduction="none")
            if self.targeted:
                if target_labels is None:
                    raise RuntimeError("Target labels are required for targeted APGD.")
                return -ce(logits, target_labels)
            return ce(logits, labels)

        if self.targeted:
            if target_labels is None:
                raise RuntimeError("Target labels are required for targeted APGD.")
            return self._dlr_loss_targeted(logits, labels, target_labels)

        return self._dlr_loss(logits, labels)

    def _success_mask(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        preds = logits.detach().argmax(dim=1)
        if self.targeted:
            if target_labels is None:
                raise RuntimeError("Target labels are required for targeted APGD.")
            return preds == target_labels
        return preds != labels

    def _validate_logits(self, logits: torch.Tensor, labels: torch.Tensor, *, context: str) -> None:
        if not self._debug_checks:
            return
        if logits.ndim != 2:
            raise RuntimeError(f"[{context}] Expected logits with shape (B, C); got {tuple(logits.shape)}")
        num_classes = logits.size(1)
        labels_flat = labels.reshape(-1)
        l_min = int(labels_flat.min().item())
        l_max = int(labels_flat.max().item())
        if l_min < 0 or l_max >= num_classes:
            raise RuntimeError(
                f"[{context}] Label range [{l_min}, {l_max}] outside model output (num_classes={num_classes}). "
                "Check dataset/model pairing or label preprocessing."
            )
        if torch.isnan(logits).any():
            raise RuntimeError(f"[{context}] Logits contain NaNs; inspect model outputs or gradients.")
        if torch.isinf(logits).any():
            raise RuntimeError(f"[{context}] Logits contain Infs; inspect model outputs or gradients.")

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]
        return t <= k * k3 * np.ones(t.shape)

    def attack_single_run(
        self,
        x_in: torch.Tensor,
        y_in: torch.Tensor,
        *,
        target_labels: Optional[torch.Tensor] = None,
    ):
        x = x_in.clone()
        y = y_in.clone() if y_in.ndim > 0 else y_in.clone().unsqueeze(0)
        if target_labels is not None and target_labels.ndim == 0:
            target_labels = target_labels.unsqueeze(0)

        batch_size = x.size(0)
        self.steps_2 = max(int(0.22 * self.steps), 1)
        self.steps_min = max(int(0.06 * self.steps), 1)
        self.size_decr = max(int(0.03 * self.steps), 1)

        if self.verbose:
            print("parameters:", self.steps, self.steps_2, self.steps_min, self.size_decr)

        x_adv = x.clone().detach()
        eps_abs = self._eps_abs(x)

        if self.random_start:
            x_adv = (x_adv + sample_l2_ratio_ball(
                x,
                eps_2=self.eps_2,
                eps_for_division=self.eps_for_division,
            )).detach()

        if self.alpha_fraction is not None:
            step_size = (self.alpha_fraction * eps_abs / max(self.steps, 1)).detach()
        elif self.alpha is not None:
            step_size = torch.full((batch_size,), float(self.alpha), device=x.device)
        else:
            step_size = (2.0 * eps_abs / max(self.steps, 1)).detach()

        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        success_best = torch.zeros(batch_size, device=x.device, dtype=torch.bool)

        loss_steps = torch.zeros([self.steps, batch_size], device=x.device)
        loss_best_steps = torch.zeros([self.steps + 1, batch_size], device=x.device)

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(max(1, self.eot_iter)):
            with torch.enable_grad():
                logits = self.get_logits(x_adv)
                labels_for_check = target_labels if self.targeted else y
                if labels_for_check is not None:
                    self._validate_logits(logits, labels_for_check, context="apgd/inner")
                loss_indiv = self._loss_per_sample(logits, y, target_labels)
                loss = loss_indiv.sum()
                if self.use_mmd_loss:
                    loss = loss + self.mmd_weight * self.calculate_mmd_loss(x_adv, x)
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(max(1, self.eot_iter))
        grad_best = grad.clone()
        loss_best = loss_indiv.detach().clone()
        loss_best_steps[0] = loss_best

        x_adv_old = x_adv.clone().detach()
        k = int(self.steps_2)
        u = np.arange(batch_size)
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape, dtype=bool)

        for i in range(self.steps):
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                a = 0.75 if i > 0 else 1.0

                grad_n = self._l2_normalize(grad)
                step_br = self._view_as_broadcast(step_size, x_adv)

                x_adv_1 = x_adv + step_br * grad_n
                delta_1 = self._project_delta(x_adv_1 - x, x)
                x_adv_1 = x + delta_1

                x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1.0 - a)
                delta_2 = self._project_delta(x_adv_1 - x, x)
                x_adv = x + delta_2

            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(max(1, self.eot_iter)):
                with torch.enable_grad():
                    logits = self.get_logits(x_adv)
                    labels_for_check = target_labels if self.targeted else y
                    if labels_for_check is not None:
                        self._validate_logits(logits, labels_for_check, context="apgd/loop")
                    loss_indiv = self._loss_per_sample(logits, y, target_labels)
                    loss = loss_indiv.sum()
                    if self.use_mmd_loss:
                        loss = loss + self.mmd_weight * self.calculate_mmd_loss(x_adv, x)
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            grad /= float(max(1, self.eot_iter))

            success = self._success_mask(logits, y, target_labels)
            success_best = success_best | success
            if success.any():
                x_best_adv[success] = x_adv.detach()[success]

            if self.verbose:
                robust_now = (~success_best).float().mean().item()
                print(f"iteration: {i} - robust: {robust_now:.2%} - best loss: {loss_best.sum().item():.6f}")

            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1
                ind = torch.where(y1 > loss_best)[0]
                if ind.numel() > 0:
                    x_best[ind] = x_adv[ind].clone()
                    grad_best[ind] = grad[ind].clone()
                    loss_best[ind] = y1[ind]
                loss_best_steps[i + 1] = loss_best

                counter3 += 1
                if counter3 == k:
                    fl_oscillation = self.check_oscillation(
                        loss_steps.detach().cpu().numpy(),
                        i,
                        k,
                        loss_best.detach().cpu().numpy(),
                        k3=self.thr_decr,
                    )
                    fl_reduce_no_impr = (~reduced_last_check) * (
                        loss_best_last_check.detach().cpu().numpy() >= loss_best.detach().cpu().numpy()
                    )
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()

                    osc_idx = np.where(fl_oscillation)[0]
                    if osc_idx.size > 0:
                        idx_t = torch.as_tensor(osc_idx, device=self.device, dtype=torch.long)
                        step_size[idx_t] = step_size[idx_t] / 2.0
                        x_adv[idx_t] = x_best[idx_t].clone()
                        grad[idx_t] = grad_best[idx_t].clone()

                    counter3 = 0
                    k = max(k - self.size_decr, self.steps_min)

        robust_mask = ~success_best
        return x_best, robust_mask, loss_best, x_best_adv

    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        given_training = self.model.training
        self._change_model_mode(given_training)
        try:
            if self.norm != "L2":
                raise ValueError(f"Unsupported norm {self.norm!r}; APGD for CSI currently supports only 'L2'.")

            x = x_in.clone() if x_in.ndim > 1 else x_in.clone().unsqueeze(0)
            y = y_in.clone() if y_in.ndim > 0 else y_in.clone().unsqueeze(0)

            if self.seed is not None:
                torch.random.manual_seed(int(self.seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(self.seed))

            target_labels = self.get_target_label(x, y) if self.targeted else None

            with torch.no_grad():
                logits_clean = self.get_logits(x)
                labels_for_check = target_labels if self.targeted else y
                if labels_for_check is not None:
                    self._validate_logits(logits_clean, labels_for_check, context="apgd/clean")
                if self.targeted:
                    success = logits_clean.argmax(dim=1) == target_labels
                else:
                    success = logits_clean.argmax(dim=1) != y

            robust = ~success
            adv_best = x.clone().detach()
            obj_best = torch.full((x.size(0),), -float("inf"), device=self.device)

            if self.verbose:
                mode = "targeted" if self.targeted else "untargeted"
                print(
                    f"---------------- running APGD-L2 ({mode}) eps_2={self.eps_2:.6f} | restarts={self.n_restarts} ----------------"
                )
                print(f"initial robust fraction: {robust.float().mean().item():.2%}")

            if not cheap:
                raise ValueError("not implemented yet")

            startt = time.time()
            for counter in range(max(1, self.n_restarts)):
                ind_to_fool = torch.where(robust)[0]
                if ind_to_fool.numel() == 0:
                    break

                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()
                t_to_fool = target_labels[ind_to_fool].clone() if target_labels is not None else None

                best_curr, robust_curr, loss_curr, adv_curr = self.attack_single_run(
                    x_to_fool,
                    y_to_fool,
                    target_labels=t_to_fool,
                )

                better = loss_curr > obj_best[ind_to_fool]
                if better.any():
                    better_idx = ind_to_fool[better]
                    adv_best[better_idx] = best_curr[better].clone()
                    obj_best[better_idx] = loss_curr[better].clone()

                success_curr = ~robust_curr
                if success_curr.any():
                    succ_idx = ind_to_fool[success_curr]
                    adv_best[succ_idx] = adv_curr[success_curr].clone()
                    robust[succ_idx] = False

                if self.verbose:
                    print(
                        "restart {} - robust fraction: {:.2%} - elapsed: {:.1f}s".format(
                            counter,
                            robust.float().mean().item(),
                            time.time() - startt,
                        )
                    )

            if best_loss:
                return obj_best, adv_best
            return robust, adv_best
        finally:
            self._recover_model_mode(given_training)

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        _, adv_images = self.perturb(images, labels, cheap=True)
        return adv_images
