from typing import Optional, Callable
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from .losses import trades_outer

try:
    from csi.models.utils.training import _maybe_tqdm
except (ModuleNotFoundError, ImportError):  # pragma: no cover - optional dependency fallback
    try:
        from csi.models.core import _maybe_tqdm
    except (ModuleNotFoundError, ImportError):  # pragma: no cover - last-resort local shim
        def _maybe_tqdm(loader, use_tqdm, desc=None):
            _ = (use_tqdm, desc)
            return loader


def _l2_normalize(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    b = t.size(0)
    flat = t.reshape(b, -1)
    n = flat.norm(p=2, dim=1).clamp_min(eps)
    return (flat / n[:, None]).reshape_as(t)


def _project_l2(x: torch.Tensor, x0: torch.Tensor, eps_abs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Project per-sample onto L2 ball of radius eps_abs[b] around x0[b]
    b = x.size(0)
    delta = (x - x0).reshape(b, -1)
    norms = delta.norm(p=2, dim=1).clamp_min(eps)
    scale = torch.minimum(eps_abs / norms, torch.ones_like(norms))
    return (x0 + (delta * scale[:, None]).reshape_as(x)).detach()


class TRADESTrainer:
    """TRADES with L2 inner maximization (KL).

    Configuration is read from a dict `cfg` mirroring PGD-AT conventions:
      - attack: steps, alpha_fraction or alpha (optional), random_start, epsilon or snr_db
      - adv_training.trades.beta: outer loss beta
    """

    def __init__(self, model, optimizer, cfg: dict, amp: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self._debug_checks = os.environ.get("CSI_VALIDATE_LOGITS", "1").strip().lower() in {"1", "true", "yes", "on"}

        atk = cfg.get('attack', {})
        self.steps = int(atk.get('steps', 10))
        self.alpha_fraction = atk.get('alpha_fraction')
        self.alpha = atk.get('alpha')  # optional absolute step
        self.random_start = bool(atk.get('random_start', True))
        self.snr_db = atk.get('snr_db', [20])
        if isinstance(self.snr_db, list):
            self.snr_db = self.snr_db[0] if self.snr_db else 20
        self.epsilon = atk.get('epsilon')

        adv_training_cfg = cfg.get('adv_training', {})
        trades_cfg = adv_training_cfg.get('trades', {})
        self.beta = float(trades_cfg.get('beta', 2.0))

    def _eps_abs(self, x: torch.Tensor) -> torch.Tensor:
        eps2 = self.epsilon
        if eps2 is None:
            from csi.attacks.utils.attack_utils import eps_l2_from_snr_db
            eps2 = eps_l2_from_snr_db(self.snr_db)
        b = x.size(0)
        flat = x.reshape(b, -1)
        xnorm = flat.norm(p=2, dim=1)
        return float(eps2) * xnorm

    def _per_sample_step(self, eps_abs: torch.Tensor) -> torch.Tensor:
        if self.alpha_fraction is not None:
            step_vec = (float(self.alpha_fraction) * eps_abs) / max(1, self.steps)
        elif self.alpha is not None:
            step_vec = torch.full_like(eps_abs, float(self.alpha))
        else:
            # Reasonable default: 2*eps/steps (common PGD heuristic)
            step_vec = (2.0 * eps_abs) / max(1, self.steps)
        return step_vec

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
            raise RuntimeError(f"[{context}] Logits contain NaNs; inspect model outputs or KL attack steps.")
        if torch.isinf(logits).any():
            raise RuntimeError(f"[{context}] Logits contain Infs; inspect model outputs or KL attack steps.")

    def _pgd_l2_kl(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x0 = x.detach()
        eps_abs = self._eps_abs(x0)
        step_vec = self._per_sample_step(eps_abs)

        if self.random_start:
            g = torch.randn_like(x0)
            g = _l2_normalize(g)
            # Random radius in [0, 1]^(1/d) for uniform in ball
            dim = g.reshape(g.size(0), -1).size(1)
            u = torch.rand(x0.size(0), device=x0.device).pow(1.0 / dim)
            r = u.reshape(-1, *([1] * (x0.dim() - 1))) * eps_abs.reshape(-1, *([1] * (x0.dim() - 1)))
            x_adv = (x0 + r * g).detach()
            x_adv = _project_l2(x_adv, x0, eps_abs)
        else:
            x_adv = x0.clone().detach()

        for _ in range(self.steps):
            x_adv.requires_grad_(True)

            # Clean probabilities for KL target (detached)
            with torch.no_grad():
                logits_clean = self.model(x0)
                self._validate_logits(logits_clean, y, context="trades/clean-inner")
                p_clean = F.log_softmax(logits_clean, dim=1).exp()

            logits_adv = self.model(x_adv)
            self._validate_logits(logits_adv, y, context="trades/adv-inner")
            kl_mean = F.kl_div(F.log_softmax(logits_adv, dim=1), p_clean, reduction="batchmean")

            grad = torch.autograd.grad(kl_mean, x_adv, retain_graph=False, create_graph=False)[0]
            g = _l2_normalize(grad)

            step_br = step_vec.reshape(-1, *([1] * (x0.dim() - 1)))
            x_adv = (x_adv + step_br * g).detach()
            x_adv = _project_l2(x_adv, x0, eps_abs)

        return x_adv.detach()

    def train_one_epoch(self, train_loader, device, log_fn: Optional[Callable] = None, *, use_tqdm: bool = False):
        self.model.train()
        total_loss = n = 0
        iterator = _maybe_tqdm(train_loader, use_tqdm, desc="TRADES training")

        for i, (x, y) in enumerate(iterator):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            # Craft adversarial examples (inner maximization)
            x_adv = self._pgd_l2_kl(x, y)

            # Outer TRADES objective
            self.optimizer.zero_grad(set_to_none=True)
            if self.amp:
                with autocast(device_type=device.type):
                    logits_clean = self.model(x)
                    logits_adv = self.model(x_adv)
                    self._validate_logits(logits_clean, y, context="trades/clean/amp")
                    self._validate_logits(logits_adv, y, context="trades/adv/amp")
                    loss = trades_outer(logits_clean, logits_adv, y, beta=self.beta)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits_clean = self.model(x)
                logits_adv = self.model(x_adv)
                self._validate_logits(logits_clean, y, context="trades/clean")
                self._validate_logits(logits_adv, y, context="trades/adv")
                loss = trades_outer(logits_clean, logits_adv, y, beta=self.beta)
                loss.backward()
                self.optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            n += bs

            if log_fn and (i % 50 == 0):
                log_fn({"step": i, "loss": total_loss / n})

        return {"loss": total_loss / n}
