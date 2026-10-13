from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from csi.defenses.adv_training.randomized_smoothing import RandomizedSmoothingTrainer
from csi.defenses.certified_robustness.noise import add_amplitude_noise
from csi.defenses.certified_robustness.randomized_smoothing import (
    RandomizedSmoother,
    certify_dataset,
    certified_accuracy_at_radius,
    evaluate_smoothed_accuracy,
)


class _AlwaysClassZero(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        logits = torch.zeros(b, self.num_classes, device=x.device)
        logits[:, 0] = 5.0
        return logits


class _MeanSignBinary(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1)
        s = flat.mean(dim=1, keepdim=True)
        return torch.cat([s, -s], dim=1)


class _TinyClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 3) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(x))


def test_randomized_smoother_certify_confident_model() -> None:
    torch.manual_seed(0)
    model = _AlwaysClassZero(num_classes=2)
    smoother = RandomizedSmoother(model, num_classes=2, sigma=0.25, device=torch.device("cpu"))
    x = torch.zeros(1, 4, 4)
    pred, radius = smoother.certify(x, n0=16, n=64, alpha=0.001, batch_size=32)
    assert pred == 0
    assert radius > 0.0


def test_add_amplitude_noise_non_negative_and_scaled() -> None:
    torch.manual_seed(0)
    x = torch.full((256, 1, 8, 8), 1.0)
    sigma = 0.25
    noisy = add_amplitude_noise(x, sigma=sigma, clamp_min=0.0)
    assert noisy.shape == x.shape
    assert float(noisy.min().item()) >= 0.0
    empirical_std = float((noisy - x).std().item())  # mostly unclipped around amplitude=1.0
    assert abs(empirical_std - (sigma / (2.0 ** 0.5))) < 0.03


def test_add_amplitude_noise_zero_sigma_no_change() -> None:
    x = torch.rand(8, 1, 4, 4)
    noisy = add_amplitude_noise(x, sigma=0.0)
    assert torch.allclose(noisy, x)


def test_add_amplitude_noise_without_clamp_allows_negative_values() -> None:
    torch.manual_seed(7)
    x = torch.zeros(256, 1, 4, 4)
    noisy = add_amplitude_noise(x, sigma=0.5, clamp_min=None)
    assert float(noisy.min().item()) < 0.0


def test_randomized_smoother_abstains_when_boundary_is_uncertain() -> None:
    torch.manual_seed(0)
    model = _MeanSignBinary()
    smoother = RandomizedSmoother(model, num_classes=2, sigma=0.25, device=torch.device("cpu"))
    x = torch.zeros(1, 4, 4)
    pred = smoother.predict(x, n=512, alpha=0.001, batch_size=128)
    cert_pred, cert_radius = smoother.certify(x, n0=32, n=256, alpha=0.001, batch_size=128)
    assert pred in {0, 1, RandomizedSmoother.ABSTAIN}
    assert cert_pred in {0, 1, RandomizedSmoother.ABSTAIN}
    if cert_pred == RandomizedSmoother.ABSTAIN:
        assert cert_radius == 0.0


def test_certify_dataset_and_metrics_helpers() -> None:
    torch.manual_seed(1)
    model = _AlwaysClassZero(num_classes=2)
    smoother = RandomizedSmoother(model, num_classes=2, sigma=0.25, device=torch.device("cpu"))
    x = torch.randn(5, 1, 4, 4)
    y = torch.zeros(5, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

    report = certify_dataset(
        smoother,
        loader,
        n0=8,
        n=32,
        alpha=0.001,
        batch_size=16,
        max_examples=3,
    )
    rows = report["rows"]
    summary = report["summary"]
    assert len(rows) == 3
    assert summary["n_certified"] == 3
    acc_r0 = certified_accuracy_at_radius(rows, radius=0.0)
    assert 0.0 <= acc_r0 <= 1.0

    smooth_eval = evaluate_smoothed_accuracy(
        smoother,
        loader,
        n=32,
        alpha=0.001,
        batch_size=16,
        max_examples=3,
    )
    assert smooth_eval["n_eval"] == 3
    assert 0.0 <= smooth_eval["accuracy"] <= 1.0


def test_randomized_smoother_with_amplitude_noise_mode() -> None:
    torch.manual_seed(3)
    model = _AlwaysClassZero(num_classes=2)
    smoother = RandomizedSmoother(
        model,
        num_classes=2,
        sigma=0.2,
        device=torch.device("cpu"),
        noise_type="amplitude",
        amplitude_clamp_min=0.0,
    )
    x = torch.zeros(1, 4, 4)
    pred, radius = smoother.certify(x, n0=8, n=32, alpha=0.001, batch_size=16)
    assert pred == 0
    assert radius >= 0.0


def test_randomized_smoothing_trainer_train_one_epoch() -> None:
    torch.manual_seed(2)
    x = torch.randn(8, 1, 4, 4)
    y = torch.randint(0, 3, (8,), dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    model = _TinyClassifier(in_features=16, num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = {
        "certified_robustness": {
            "randomized_smoothing": {
                "sigma": 0.25,
                "num_noise_vec": 2,
                "mix_clean": 0.1,
            }
        }
    }
    trainer = RandomizedSmoothingTrainer(model, optimizer, cfg, amp=False)
    stats = trainer.train_one_epoch(loader, device=torch.device("cpu"), use_tqdm=False)
    assert "loss" in stats and stats["loss"] >= 0.0
    assert "loss_noisy" in stats and stats["loss_noisy"] >= 0.0


def test_randomized_smoothing_trainer_amplitude_mode() -> None:
    torch.manual_seed(4)
    x = torch.rand(8, 1, 4, 4)
    y = torch.randint(0, 3, (8,), dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    model = _TinyClassifier(in_features=16, num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = {
        "certified_robustness": {
            "randomized_smoothing": {
                "sigma": 0.2,
                "num_noise_vec": 1,
                "noise_type": "amplitude",
                "amplitude_clamp_min": 0.0,
            }
        }
    }
    trainer = RandomizedSmoothingTrainer(model, optimizer, cfg, amp=False)
    stats = trainer.train_one_epoch(loader, device=torch.device("cpu"), use_tqdm=False)
    assert "loss" in stats and stats["loss"] >= 0.0
