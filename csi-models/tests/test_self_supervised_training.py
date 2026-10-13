"""Smoke tests for SSL training loops (SimCLR and WiMAE)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from csi.models.self_supervised import (
    WiMAEBackbone,
    build_simclr_timeseries_backbone,
    train_simclr,
    train_wimae,
)


class _ToyContrastiveDataset(Dataset):
    def __init__(self, n: int = 16, channels: int = 1, freq_bins: int = 16, time_steps: int = 16) -> None:
        super().__init__()
        self.base = torch.randn(n, channels, freq_bins, time_steps)

    def __len__(self) -> int:
        return int(self.base.shape[0])

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        noise_1 = 0.01 * torch.randn_like(sample)
        noise_2 = 0.01 * torch.randn_like(sample)
        return [sample + noise_1, sample + noise_2], 0


class _ToyImageDataset(Dataset):
    def __init__(self, n: int = 16, channels: int = 1, height: int = 16, width: int = 16) -> None:
        super().__init__()
        self.data = torch.randn(n, channels, height, width)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int):
        return self.data[idx], 0


def test_train_simclr_smoke(tmp_path: Path) -> None:
    model = build_simclr_timeseries_backbone(in_channels=1, hidden_dim=16)
    train_loader = DataLoader(_ToyContrastiveDataset(n=20), batch_size=4, shuffle=False)
    val_loader = DataLoader(_ToyContrastiveDataset(n=12), batch_size=4, shuffle=False)
    ckpt = tmp_path / "simclr_smoke.pt"

    summary = train_simclr(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.2,
        device="cpu",
        use_tqdm=False,
        checkpoint_path=ckpt,
    )

    assert len(summary.history) >= 1
    assert summary.best_epoch >= 1
    assert summary.best_state is not None
    assert summary.final_train_loss >= 0.0
    assert ckpt.exists()


def test_train_wimae_smoke(tmp_path: Path) -> None:
    model = WiMAEBackbone(
        img_size=(16, 16),
        patch_size=(4, 4),
        in_channels=1,
        embed_dim=16,
        depth=2,
        num_heads=4,
        decoder_embed_dim=8,
        decoder_depth=1,
        decoder_num_heads=4,
        mlp_ratio=2.0,
        mask_ratio=0.5,
        norm_pix_loss=True,
    )
    train_loader = DataLoader(_ToyImageDataset(n=20), batch_size=4, shuffle=False)
    val_loader = DataLoader(_ToyImageDataset(n=12), batch_size=4, shuffle=False)
    ckpt = tmp_path / "wimae_smoke.pt"

    summary = train_wimae(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=2,
        learning_rate=1e-3,
        weight_decay=5e-2,
        device="cpu",
        use_tqdm=False,
        checkpoint_path=ckpt,
    )

    assert len(summary.history) >= 1
    assert summary.best_epoch >= 1
    assert summary.best_state is not None
    assert summary.final_train_loss >= 0.0
    assert ckpt.exists()
