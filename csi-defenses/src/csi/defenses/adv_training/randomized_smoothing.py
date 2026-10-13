from __future__ import annotations

import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from csi.defenses.certified_robustness.noise import add_noise

try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover - torch<2.0 compatibility
    from torch.cuda.amp import GradScaler, autocast

try:
    from csi.models.utils.training import _maybe_tqdm
except (ModuleNotFoundError, ImportError):  # pragma: no cover - optional dependency fallback
    try:
        from csi.models.core import _maybe_tqdm
    except (ModuleNotFoundError, ImportError):  # pragma: no cover - last-resort local shim
        def _maybe_tqdm(loader, use_tqdm, desc=None):
            _ = (use_tqdm, desc)
            return loader


def _autocast_context_for_device(device: torch.device):
    try:
        return autocast(device_type=device.type)
    except TypeError:  # pragma: no cover - torch.cuda.amp.autocast signature
        return autocast()


class RandomizedSmoothingTrainer:
    """Noise-augmentation trainer for randomized smoothing.

    Supports:
    - `noise_type='gaussian'` (default)
    - `noise_type='amplitude'` for amplitude-only CSI tensors
    """

    def __init__(self, model, optimizer, cfg: dict, amp: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self._debug_checks = os.environ.get("CSI_VALIDATE_LOGITS", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        cert_cfg = cfg.get("certified_robustness", {}) or {}
        rs_cfg = cert_cfg.get("randomized_smoothing", cert_cfg) or {}

        self.sigma = float(rs_cfg.get("sigma", 0.25))
        self.num_noise_vec = int(rs_cfg.get("num_noise_vec", rs_cfg.get("m_train", 1)) or 1)
        self.mix_clean = float(rs_cfg.get("mix_clean", 0.0))
        self.noise_type = str(rs_cfg.get("noise_type", "gaussian")).strip().lower()
        raw_clamp_min = rs_cfg.get("amplitude_clamp_min", 0.0)
        self.amplitude_clamp_min = None if raw_clamp_min is None else float(raw_clamp_min)
        self.clip_grad_norm = rs_cfg.get("clip_grad_norm")
        self.clip_grad_norm = None if self.clip_grad_norm is None else float(self.clip_grad_norm)

        if self.sigma <= 0.0:
            raise ValueError("randomized smoothing sigma must be > 0.")
        if self.num_noise_vec <= 0:
            raise ValueError("randomized smoothing num_noise_vec must be >= 1.")
        if not (0.0 <= self.mix_clean <= 1.0):
            raise ValueError("randomized smoothing mix_clean must be in [0, 1].")

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
                f"[{context}] Label range [{l_min}, {l_max}] outside model output (num_classes={num_classes})."
            )
        if torch.isnan(logits).any():
            raise RuntimeError(f"[{context}] Logits contain NaNs; inspect model outputs/noise settings.")
        if torch.isinf(logits).any():
            raise RuntimeError(f"[{context}] Logits contain Infs; inspect model outputs/noise settings.")

    def _build_noisy_batch(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_noise_vec == 1:
            x_rep = x
            y_rep = y
        else:
            x_rep = x.repeat_interleave(self.num_noise_vec, dim=0)
            y_rep = y.repeat_interleave(self.num_noise_vec, dim=0)
        x_noisy = add_noise(
            x_rep,
            self.sigma,
            noise_type=self.noise_type,
            amplitude_clamp_min=self.amplitude_clamp_min,
        )
        return x_noisy, y_rep

    def train_one_epoch(self, train_loader, device, log_fn: Optional[Callable] = None, *, use_tqdm: bool = False):
        self.model.train()
        total_loss = total_noisy = total_clean = n = 0

        iterator = _maybe_tqdm(train_loader, use_tqdm, desc="RS training")
        for i, (x, y) in enumerate(iterator):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            x_noisy, y_noisy = self._build_noisy_batch(x, y)
            self.optimizer.zero_grad(set_to_none=True)

            if self.amp:
                with _autocast_context_for_device(device):
                    logits_noisy = self.model(x_noisy)
                    self._validate_logits(logits_noisy, y_noisy, context="rs/noisy/amp")
                    loss_noisy = F.cross_entropy(logits_noisy, y_noisy)
                    if self.mix_clean > 0:
                        logits_clean = self.model(x)
                        self._validate_logits(logits_clean, y, context="rs/clean/amp")
                        loss_clean = F.cross_entropy(logits_clean, y)
                        loss = (1.0 - self.mix_clean) * loss_noisy + self.mix_clean * loss_clean
                    else:
                        loss_clean = torch.tensor(0.0, device=x.device, dtype=loss_noisy.dtype)
                        loss = loss_noisy
                self.scaler.scale(loss).backward()
                if self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits_noisy = self.model(x_noisy)
                self._validate_logits(logits_noisy, y_noisy, context="rs/noisy")
                loss_noisy = F.cross_entropy(logits_noisy, y_noisy)
                if self.mix_clean > 0:
                    logits_clean = self.model(x)
                    self._validate_logits(logits_clean, y, context="rs/clean")
                    loss_clean = F.cross_entropy(logits_clean, y)
                    loss = (1.0 - self.mix_clean) * loss_noisy + self.mix_clean * loss_clean
                else:
                    loss_clean = torch.tensor(0.0, device=x.device, dtype=loss_noisy.dtype)
                    loss = loss_noisy
                loss.backward()
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_noisy += float(loss_noisy.item()) * bs
            total_clean += float(loss_clean.item()) * bs
            n += bs

            if log_fn and (i % 50 == 0):
                log_fn(
                    {
                        "step": i,
                        "loss": total_loss / n,
                        "loss_noisy": total_noisy / n,
                        "loss_clean": total_clean / n,
                    }
                )

        if n == 0:
            return {"loss": 0.0, "loss_noisy": 0.0, "loss_clean": 0.0}
        return {
            "loss": total_loss / n,
            "loss_noisy": total_noisy / n,
            "loss_clean": total_clean / n,
        }
