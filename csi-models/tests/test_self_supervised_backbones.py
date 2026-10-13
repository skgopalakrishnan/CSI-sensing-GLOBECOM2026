"""Smoke tests for self-supervised backbones."""

from __future__ import annotations

import torch

from csi.models.self_supervised import (
    WiMAEBackbone,
    build_simclr_image_backbone,
    build_simclr_timeseries_backbone,
    simclr_info_nce_loss,
)


def test_simclr_image_backbone_shape() -> None:
    model = build_simclr_image_backbone(hidden_dim=32)
    x = torch.randn(4, 3, 64, 64)
    feats = model(x)
    assert feats.shape == (4, 32)


def test_simclr_timeseries_backbone_shape() -> None:
    model = build_simclr_timeseries_backbone(in_channels=6, hidden_dim=48)
    x = torch.randn(3, 6, 48, 96)
    feats = model(x)
    assert feats.shape == (3, 48)


def test_simclr_info_nce_loss_outputs() -> None:
    feats = torch.randn(8, 24)
    loss, metrics = simclr_info_nce_loss(feats, temperature=0.2)
    assert loss.ndim == 0
    assert set(metrics.keys()) == {"acc_top1", "acc_top5", "acc_mean_pos"}


def test_wimae_backbone_shapes() -> None:
    model = WiMAEBackbone(
        img_size=(32, 48),
        patch_size=(4, 4),
        in_channels=2,
        embed_dim=32,
        depth=2,
        num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
        mlp_ratio=2.0,
        mask_ratio=0.5,
        norm_pix_loss=True,
    )

    x = torch.randn(2, 2, 32, 48)
    loss, pred, mask = model(x)

    num_patches = (32 // 4) * (48 // 4)
    patch_dim = 4 * 4 * 2

    assert loss.ndim == 0
    assert pred.shape == (2, num_patches, patch_dim)
    assert mask.shape == (2, num_patches)

    feats = model.forward_features(x)
    assert feats.shape == (2, 32)

    reconstructed = model.unpatchify(model.patchify(x))
    assert torch.allclose(x, reconstructed)
