"""Minimal smoke tests for csi-defenses trainers and losses."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from csi.defenses.adv_training.losses import ce_inner, trades_outer
import csi.defenses.adv_training.pgd_at as pgd_at_mod
from csi.defenses.adv_training.pgd_at import PGDATrainer


class _TinyClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 3) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(x))


class _DummyAttack:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _ = y
        self.calls += 1
        return x + (0.01 * self.calls)


def _make_loader(shape = (4, 1, 4, 4)) -> DataLoader:
    x = torch.randn(*shape)
    y = torch.randint(0, 3, (shape[0],), dtype=torch.long)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=shape[0], shuffle=False)


def test_losses_basic_properties() -> None:
    logits = torch.randn(5, 3)
    labels = torch.tensor([0, 1, 2, 0, 1])
    inner = ce_inner(logits, labels)
    assert inner.shape == (5,)
    adv_logits = torch.randn(5, 3)
    loss = trades_outer(logits, adv_logits, labels, beta=1.0)
    assert torch.isfinite(loss)


def test_pgdatrainer_apgd_maps_n_restarts_to_internal(monkeypatch) -> None:
    called = {}
    attack_obj = _DummyAttack()

    def fake_create_attack(model, tensor_loader, **kwargs):
        _ = (model, tensor_loader)
        called.update(kwargs)
        return attack_obj

    monkeypatch.setattr(pgd_at_mod, "_resolve_create_attack", lambda: fake_create_attack)

    cfg = {
        "attack": {
            "methods": "apgd",
            "steps": 3,
            "random_start": False,
            "epsilon": 0.1,
            "n_restarts": 5,
        },
        "adv_training": {
            "mix_clean": 0.0,
        },
    }

    model = _TinyClassifier(16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = PGDATrainer(model, optimizer, cfg, amp=False)
    _ = trainer._ensure_attack(torch.device("cpu"), _make_loader())

    assert trainer.n_restarts_inner == 5
    assert trainer.n_restarts_outer == 1
    assert called["attack_method"] == "apgd"
    assert called["attack_n_restarts"] == 5
    assert called["loss"] == "ce"
    assert called["random_start"] is True


def test_pgdatrainer_pgdl2_keeps_external_wrapper_restarts(monkeypatch) -> None:
    attack_obj = _DummyAttack()

    def fake_create_attack(model, tensor_loader, **kwargs):
        _ = (model, tensor_loader, kwargs)
        return attack_obj

    monkeypatch.setattr(pgd_at_mod, "_resolve_create_attack", lambda: fake_create_attack)

    loader = _make_loader()
    sample = next(iter(loader))[0]
    in_features = sample[0].numel()
    model = _TinyClassifier(in_features)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = {
        "attack": {
            "methods": "pgdl2",
            "steps": 1,
            "random_start": False,
            "epsilon": 0.1,
            "n_restarts": 3,
        },
        "adv_training": {
            "mix_clean": 0.0,
        },
    }
    trainer = PGDATrainer(model, optimizer, cfg, amp=False)
    stats = trainer.train_one_epoch(loader, device=torch.device("cpu"), use_tqdm=False)
    assert trainer.n_restarts_inner == 1
    assert trainer.n_restarts_outer == 3
    assert attack_obj.calls == 3
    assert "loss" in stats and stats["loss"] >= 0.0


def test_pgdatrainer_uses_snr_when_epsilon_missing(monkeypatch) -> None:
    called = {}

    def fake_create_attack(model, tensor_loader, **kwargs):
        _ = (model, tensor_loader)
        called.update(kwargs)
        return _DummyAttack()

    monkeypatch.setattr(pgd_at_mod, "_resolve_create_attack", lambda: fake_create_attack)
    monkeypatch.setattr(pgd_at_mod, "_resolve_eps_l2_from_snr_db", lambda: (lambda snr: 0.123 if snr == 20 else 0.0))

    loader = _make_loader()
    model = _TinyClassifier(16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = {
        "attack": {
            "methods": "apgd",
            "steps": 1,
            "random_start": False,
            "snr_db": [20],
        },
        "adv_training": {
            "mix_clean": 0.0,
        },
    }

    trainer = PGDATrainer(model, optimizer, cfg, amp=False)
    _ = trainer._ensure_attack(torch.device("cpu"), loader)
    assert called["epsilon"] == 0.123


def test_pgdatrainer_passes_extra_attack_kwargs(monkeypatch) -> None:
    called = {}

    def fake_create_attack(model, tensor_loader, **kwargs):
        _ = (model, tensor_loader)
        called.update(kwargs)
        return _DummyAttack()

    monkeypatch.setattr(pgd_at_mod, "_resolve_create_attack", lambda: fake_create_attack)

    loader = _make_loader()
    model = _TinyClassifier(16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = {
        "attack": {
            "methods": "apgdt",
            "steps": 1,
            "epsilon": 0.1,
            "n_restarts": 2,
            "n_classes": 3,
            "target_class_sweep": True,
            "n_target_classes": 2,
        },
        "adv_training": {
            "mix_clean": 0.0,
        },
    }

    trainer = PGDATrainer(model, optimizer, cfg, amp=False)
    _ = trainer._ensure_attack(torch.device("cpu"), loader)
    assert called["n_classes"] == 3
    assert called["target_class_sweep"] is True
    assert called["n_target_classes"] == 2
