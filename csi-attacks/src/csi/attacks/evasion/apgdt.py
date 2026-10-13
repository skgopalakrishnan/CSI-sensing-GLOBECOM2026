import time
from typing import Optional

import torch

from .apgd import APGD


class APGDT(APGD):
    r"""
    Targeted Auto-PGD (APGD-T) variant for CSI with L2-ratio constraints.

    By default (`target_class_sweep=True`), this follows APGD-T style target
    sweep:
    - per sample, select top-K non-true classes from clean logits
    - run targeted APGD for each class and each restart
    - aggregate success with logical OR across target classes and restarts

    Set `target_class_sweep=False` to fall back to APGD behavior with externally
    supplied target labels (e.g., targeted-by-label mapping).

    eps semantics:
        - `eps` is eps_2 (dimensionless L2 ratio), i.e. ||delta||_2 <= eps_2 * ||x||_2.
    """

    def __init__(
        self,
        model,
        norm: str = "L2",
        eps: float = 8 / 255,
        steps: int = 10,
        n_restarts: int = 1,
        seed: int = 0,
        eot_iter: int = 1,
        rho: float = 0.75,
        verbose: bool = False,
        n_classes: int = 10,
        n_target_classes: Optional[int] = None,
        random_start: bool = True,
        alpha: Optional[float] = 0.05,
        alpha_fraction: Optional[float] = None,
        eps_for_division: float = 1e-10,
        loss: str = "dlr",
        target_class_sweep: bool = True,
        dataset_name: Optional[str] = None,
        preserve_correlation: bool = False,
        use_mmd_loss: bool = False,
        mmd_weight: float = 0.0,
    ):
        super().__init__(
            model,
            norm=norm,
            eps=eps,
            steps=steps,
            n_restarts=n_restarts,
            seed=seed,
            loss=loss,
            eot_iter=eot_iter,
            rho=rho,
            verbose=verbose,
            random_start=random_start,
            alpha=alpha,
            alpha_fraction=alpha_fraction,
            eps_for_division=eps_for_division,
            dataset_name=dataset_name,
            preserve_correlation=preserve_correlation,
            use_mmd_loss=use_mmd_loss,
            mmd_weight=mmd_weight,
        )
        self.attack = "APGDT"
        max_targets = max(int(n_classes) - 1, 1)
        if n_target_classes is None:
            self.n_target_classes = max_targets
        else:
            self.n_target_classes = max(1, min(int(n_target_classes), max_targets))
        self.target_class_sweep = bool(target_class_sweep)

    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        if (not self.target_class_sweep) or self.loss != "dlr":
            return super().perturb(x_in, y_in, best_loss=best_loss, cheap=cheap)

        given_training = self.model.training
        self._change_model_mode(given_training)
        try:
            if self.norm != "L2":
                raise ValueError(f"Unsupported norm {self.norm!r}; APGDT for CSI currently supports only 'L2'.")

            x = x_in.clone() if x_in.ndim > 1 else x_in.clone().unsqueeze(0)
            y = y_in.clone() if y_in.ndim > 0 else y_in.clone().unsqueeze(0)
            y = y.long()

            if self.seed is not None:
                torch.random.manual_seed(int(self.seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(self.seed))

            with torch.no_grad():
                logits_clean = self.get_logits(x)
                self._validate_logits(logits_clean, y, context="apgdt/clean")
                preds_clean = logits_clean.argmax(dim=1)

            # Standard robust-eval convention: attack only clean-correct samples.
            robust = preds_clean == y
            adv_best = x.clone().detach()
            obj_best = torch.full((x.size(0),), -float("inf"), device=self.device)

            if not cheap:
                raise ValueError("not implemented yet")

            # Build per-sample ordered target candidates from clean logits.
            # Remove the true class and keep the top-K most likely wrong classes.
            classes_desc = logits_clean.argsort(dim=1, descending=True)
            wrong_mask = classes_desc != y.unsqueeze(1)
            target_pool = classes_desc[wrong_mask].view(x.size(0), -1)
            n_targets = min(self.n_target_classes, target_pool.size(1))
            target_pool = target_pool[:, :n_targets]

            if self.verbose:
                print(
                    f"---------------- running APGD-T sweep eps_2={self.eps_2:.6f} | "
                    f"targets={n_targets} | restarts={self.n_restarts} ----------------"
                )
                print(f"initial robust fraction: {robust.float().mean().item():.2%}")

            prev_targeted = bool(self.targeted)
            self.targeted = True
            try:
                startt = time.time()
                for t_rank in range(n_targets):
                    ind_to_fool_global = torch.where(robust)[0]
                    if ind_to_fool_global.numel() == 0:
                        break

                    target_labels_full = target_pool[:, t_rank]
                    robust_this_target = robust.clone()

                    for restart in range(max(1, self.n_restarts)):
                        ind_to_fool = torch.where(robust_this_target)[0]
                        if ind_to_fool.numel() == 0:
                            break

                        x_to_fool = x[ind_to_fool].clone()
                        y_to_fool = y[ind_to_fool].clone()
                        t_to_fool = target_labels_full[ind_to_fool].clone()

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
                            robust_this_target[succ_idx] = False

                        if self.verbose:
                            print(
                                "target_rank {} - restart {} - robust fraction: {:.2%} - elapsed: {:.1f}s".format(
                                    t_rank,
                                    restart,
                                    robust.float().mean().item(),
                                    time.time() - startt,
                                )
                            )
            finally:
                self.targeted = prev_targeted

            if best_loss:
                return obj_best, adv_best
            return robust, adv_best
        finally:
            self._recover_model_mode(given_training)
