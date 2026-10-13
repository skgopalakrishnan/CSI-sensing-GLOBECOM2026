import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from csi.attacks.evasion.attack_factory import create_attack, generate_X_adv


class _TinyLinear(nn.Module):
    def __init__(self, in_shape, num_classes: int):
        super().__init__()
        features = int(torch.tensor(in_shape).prod().item())
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(features, int(num_classes))

    def forward(self, x):
        return self.fc(self.flatten(x))


def _build_loader(shape=(1, 8, 8), num_classes: int = 3, n: int = 6):
    x = torch.randn(n, *shape)
    y = torch.tensor([i % num_classes for i in range(n)], dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)
    return loader, x, y


def test_create_attack_supports_only_apgd_and_apgdt():
    loader, _, _ = _build_loader()
    model = _TinyLinear((1, 8, 8), num_classes=3)

    atk_apgd = create_attack(
        model,
        loader,
        device="cpu",
        attack_method="apgd",
        epsilon=0.1,
        steps=2,
        random_start=False,
    )
    assert atk_apgd.attack == "APGD"

    atk_apgdt = create_attack(
        model,
        loader,
        device="cpu",
        attack_method="apgdt",
        epsilon=0.1,
        steps=2,
        n_classes=3,
        random_start=False,
    )
    assert atk_apgdt.attack == "APGDT"
    assert atk_apgdt.target_class_sweep is True
    assert atk_apgdt.n_target_classes == 2

    with pytest.raises(NotImplementedError, match="Use one of: apgd, apgdt"):
        create_attack(
            model,
            loader,
            device="cpu",
            attack_method="pgdl2",
            epsilon=0.1,
        )


def test_generate_x_adv_targeted_apgdt_reports_success_mask():
    loader, _, labels = _build_loader(shape=(1, 8, 8), num_classes=3, n=6)
    model = _TinyLinear((1, 8, 8), num_classes=3)
    label_mapping = {0: 0, 1: 1, 2: 2}

    adv_loader, adv_labels, summary, differences = generate_X_adv(
        model,
        loader,
        label_mapping=label_mapping,
        device="cpu",
        attack_method="apgdt",
        epsilon=0.1,
        targeted=True,
        steps=2,
        n_classes=3,
        n_target_classes=2,
        n_restarts=1,
        random_start=True,
    )

    assert len(adv_loader.dataset) == len(labels)
    assert int(adv_labels.numel()) == int(labels.numel())
    assert differences is not None
    assert differences.shape[0] == labels.numel()
    assert "attack_success_mask" in summary
    assert summary["attack_success_mask"].shape[0] == labels.numel()
    assert "attack_defined_success_rate_all" in summary
    assert summary["attack_defined_success_rate_all"] == pytest.approx(
        float(summary["attack_success_mask"].mean())
    )
