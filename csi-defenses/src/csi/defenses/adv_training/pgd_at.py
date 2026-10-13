import importlib
import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F

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


_APGD_METHODS = {"apgd", "apgdt"}
_CONSUMED_ATTACK_KEYS = {
    "method",
    "methods",
    "steps",
    "alpha",
    "alpha_fraction",
    "random_start",
    "loss",
    "epsilon",
    "snr_db",
    "n_restarts",
    "attack_n_restarts",
    "n_restarts_inner",
    "internal_n_restarts",
    "wrapper_n_restarts",
    "n_restarts_outer",
}


def _resolve_create_attack() -> Callable:
    candidates = (
        "csi.attacks.evasion.attack_factory",
        "csi.attacks.adv_ml.attack_factory",
    )
    errors = []
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        create_attack = getattr(module, "create_attack", None)
        if callable(create_attack):
            return create_attack
        errors.append(f"{module_name}: missing create_attack")

    msg = (
        "Could not import attack factory from csi-attacks. "
        "Tried: " + ", ".join(candidates) + ". "
        "Install/configure csi-attacks on PYTHONPATH."
    )
    if errors:
        msg = msg + " Details: " + " | ".join(errors)
    raise ModuleNotFoundError(msg)


def _resolve_eps_l2_from_snr_db() -> Callable:
    candidates = (
        "csi.attacks.utils.attack_utils",
        "csi.attacks.adv_ml.attack_utils",
    )
    errors = []
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        fn = getattr(module, "eps_l2_from_snr_db", None)
        if callable(fn):
            return fn
        errors.append(f"{module_name}: missing eps_l2_from_snr_db")

    msg = (
        "Could not resolve eps_l2_from_snr_db from csi-attacks. "
        "Tried: " + ", ".join(candidates) + "."
    )
    if errors:
        msg = msg + " Details: " + " | ".join(errors)
    raise ModuleNotFoundError(msg)


def _coerce_attack_method(raw_attack_cfg: dict) -> str:
    if "method" in raw_attack_cfg:
        raw_method = raw_attack_cfg.get("method")
    else:
        raw_method = raw_attack_cfg.get("methods", "apgd")
    if isinstance(raw_method, (list, tuple)):
        if not raw_method:
            return "apgd"
        raw_method = raw_method[0]
    return str(raw_method).strip().lower()


def _autocast_context_for_device(device: torch.device):
    try:
        return autocast(device_type=device.type)
    except TypeError:  # pragma: no cover - torch.cuda.amp.autocast signature
        return autocast()


class PGDATrainer:
    """Adversarial training with configurable inner maximization attack.

    APGD/APGD-T conventions:
    - `attack.n_restarts` defaults to attack-internal restarts for APGD/APGD-T.
    - `attack.attack_n_restarts` (or `n_restarts_inner`) explicitly sets
      APGD/APGD-T internal restarts.
    - `attack.wrapper_n_restarts` (or `n_restarts_outer`) controls external
      best-of-N wrapper restarts regardless of method.

    Args:
        model: the model to train
        optimizer: the optimizer to use
        cfg: configuration dictionary
        amp: whether to use automatic mixed precision (AMP) for training
    """
    def __init__(self, model, optimizer, cfg: dict, amp: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self._attack = None  # lazily created
        self._debug_checks = os.environ.get("CSI_VALIDATE_LOGITS", "1").strip().lower() in {"1", "true", "yes", "on"}
        
        attack_cfg = cfg.get("attack", {})
        self.attack_method = _coerce_attack_method(attack_cfg)
        self.steps = int(attack_cfg.get("steps", 20))
        self.alpha_fraction = attack_cfg.get("alpha_fraction")
        self.alpha = attack_cfg.get("alpha")
        self.random_start = bool(attack_cfg.get("random_start", True))

        attack_loss_default = "dlr" if self.attack_method == "apgdt" else "ce"
        self.attack_loss = str(attack_cfg.get("loss", attack_loss_default)).strip().lower()

        n_restarts_cfg = int(attack_cfg.get("n_restarts", 1) or 1)
        n_restarts_inner_cfg = attack_cfg.get(
            "attack_n_restarts",
            attack_cfg.get("n_restarts_inner", attack_cfg.get("internal_n_restarts")),
        )
        n_restarts_outer_cfg = attack_cfg.get(
            "wrapper_n_restarts",
            attack_cfg.get("n_restarts_outer"),
        )

        if n_restarts_inner_cfg is None:
            self.n_restarts_inner = n_restarts_cfg if self.attack_method in _APGD_METHODS else 1
        else:
            self.n_restarts_inner = int(n_restarts_inner_cfg or 1)

        if n_restarts_outer_cfg is None:
            self.n_restarts_outer = 1 if self.attack_method in _APGD_METHODS else n_restarts_cfg
        else:
            self.n_restarts_outer = int(n_restarts_outer_cfg or 1)

        adv_training_cfg = cfg.get("adv_training", {})
        self.mix_clean = float(adv_training_cfg.get("mix_clean", 0.0))

        self.snr_db = attack_cfg.get("snr_db", [20])
        if isinstance(self.snr_db, list):
            self.snr_db = self.snr_db[0] if self.snr_db else 20
        self.epsilon = attack_cfg.get("epsilon")
        self.attack_extra_kwargs = {
            k: v for k, v in attack_cfg.items() if k not in _CONSUMED_ATTACK_KEYS
        }

    def _build_attack_kwargs(self) -> dict:
        kwargs = dict(self.attack_extra_kwargs)
        kwargs["steps"] = self.steps
        force_random_start = (self.n_restarts_outer > 1) or (self.n_restarts_inner > 1)
        kwargs["random_start"] = bool(self.random_start or force_random_start)

        if self.alpha is not None:
            kwargs["alpha"] = self.alpha
        if self.alpha_fraction is not None:
            kwargs["alpha_fraction"] = self.alpha_fraction

        if self.attack_method in _APGD_METHODS:
            kwargs["attack_n_restarts"] = int(max(1, self.n_restarts_inner))
            kwargs["loss"] = self.attack_loss

        return kwargs

    def _ensure_attack(self, device, example_loader):
        if self._attack is not None:
            return self._attack

        create_attack = _resolve_create_attack()
        attack_kwargs = self._build_attack_kwargs()

        epsilon = self.epsilon
        if epsilon is None:
            eps_l2_from_snr_db = _resolve_eps_l2_from_snr_db()
            epsilon = eps_l2_from_snr_db(self.snr_db)

        self._attack = create_attack(
            self.model,
            example_loader,
            device=device,
            attack_method=self.attack_method,
            epsilon=epsilon,
            targeted=False,
            **attack_kwargs,
        )
        return self._attack


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
            raise RuntimeError(f"[{context}] Logits contain NaNs; inspect model outputs or attack perturbations.")
        if torch.isinf(logits).any():
            raise RuntimeError(f"[{context}] Logits contain Infs; inspect model outputs or attack perturbations.")

    def train_one_epoch(self, train_loader, device, log_fn: Optional[Callable] = None, *, use_tqdm: bool = False):
        self.model.train()
        attack = self._ensure_attack(device, train_loader)
        total_loss = total_adv = n = 0

        iterator = _maybe_tqdm(train_loader, use_tqdm, desc="PGD-AT training")
        for i, (x, y) in enumerate(iterator):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            if int(self.n_restarts_outer) > 1:
                best_loss = None
                best_x_adv = None
                for _ in range(int(self.n_restarts_outer)):
                    x_adv_try = attack(x, y)
                    with torch.no_grad():
                        logits_try = self.model(x_adv_try)
                        loss_try = F.cross_entropy(logits_try, y)
                    if best_loss is None or float(loss_try.item()) > float(best_loss):
                        best_loss = float(loss_try.item())
                        best_x_adv = x_adv_try.detach()
                x_adv = best_x_adv  # type: ignore
            else:
                x_adv = attack(x, y).detach()

            # Outer update: single step on adversarial batch
            self.optimizer.zero_grad(set_to_none=True)
            if self.amp:  # use AMP for outer step if enabled
                with _autocast_context_for_device(device):
                    logits_adv = self.model(x_adv)
                    self._validate_logits(logits_adv, y, context="pgdat/adv/amp")
                    loss_adv = F.cross_entropy(logits_adv, y)
                    if self.mix_clean > 0:  # Mix clean and adv losses
                        logits_clean = self.model(x)  # `x` is clean batch
                        self._validate_logits(logits_clean, y, context="pgdat/clean/amp")
                        loss_clean = F.cross_entropy(logits_clean, y)
                        loss = (1 - self.mix_clean) * loss_adv + self.mix_clean * loss_clean
                    else:
                        loss = loss_adv
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits_adv = self.model(x_adv)
                self._validate_logits(logits_adv, y, context="pgdat/adv")
                loss_adv = F.cross_entropy(logits_adv, y)
                if self.mix_clean > 0:
                    logits_clean = self.model(x)
                    self._validate_logits(logits_clean, y, context="pgdat/clean")
                    loss_clean = F.cross_entropy(logits_clean, y)
                    loss = (1 - self.mix_clean) * loss_adv + self.mix_clean * loss_clean
                else:
                    loss = loss_adv
                loss.backward()
                self.optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_adv += loss_adv.item() * bs
            n += bs
            if log_fn and (i % 50 == 0):
                log_fn({
                    "step": i,
                    "loss": total_loss / n,
                    "loss_adv": total_adv / n,
                })
        return {"loss": total_loss / n, "loss_adv": total_adv / n}
