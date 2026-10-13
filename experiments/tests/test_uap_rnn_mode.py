import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.attacks_raw_csi.uap_attack_apgd import _estimate_uap_fooling


class _TinyBiLSTM(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 3, num_classes: int = 2):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        seq = x.squeeze(1)
        out, _ = self.rnn(seq)
        return self.fc(out[:, -1, :])


def _build_loader(n: int = 6, timesteps: int = 5, features: int = 4):
    x = torch.randn(n, 1, timesteps, features)
    y = torch.tensor([i % 2 for i in range(n)], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)


def test_estimate_uap_fooling_restores_training_mode_for_rnns():
    model = _TinyBiLSTM()
    model.train()
    loader = _build_loader()
    delta = torch.zeros(1, 5, 4)

    _ = _estimate_uap_fooling(
        model=model,
        loader=loader,
        delta=delta,
        device=torch.device("cpu"),
        attack_type="untargeted",
        attack_method="apgd",
        label_mapping={0: [0], 1: [1]},
        target_class_sweep=False,
        max_batches=1,
    )

    assert model.training is True


def test_estimate_uap_fooling_preserves_eval_mode_for_non_training_callers():
    model = _TinyBiLSTM()
    model.eval()
    loader = _build_loader()
    delta = torch.zeros(1, 5, 4)

    _ = _estimate_uap_fooling(
        model=model,
        loader=loader,
        delta=delta,
        device=torch.device("cpu"),
        attack_type="untargeted",
        attack_method="apgd",
        label_mapping={0: [0], 1: [1]},
        target_class_sweep=False,
        max_batches=1,
    )

    assert model.training is False
