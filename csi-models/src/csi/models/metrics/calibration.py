"""Calibration metrics for classifier outputs."""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "expected_calibration_error",
]


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
    *,
    eps: float = 1e-12,
) -> float:
    """Compute the classic Expected Calibration Error (ECE).

    Parameters
    ----------
    probabilities:
        Tensor of shape ``(N, num_classes)`` containing class probabilities.
    labels:
        Tensor of shape ``(N,)`` with integer class indices.
    num_bins:
        Number of confidence bins.
    eps:
        Small constant to avoid divide-by-zero.
    """

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D tensor of shape (N, num_classes)")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor")
    if probabilities.size(0) != labels.size(0):
        raise ValueError("probabilities and labels must share the batch dimension")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    confidences, predictions = probabilities.max(dim=1)
    accuracies = predictions.eq(labels)
    bin_boundaries = torch.linspace(0.0, 1.0, num_bins + 1, device=probabilities.device)
    total = confidences.numel()
    if total == 0:
        return 0.0
    ece = torch.zeros(1, device=probabilities.device)
    for idx in range(num_bins):
        lower = bin_boundaries[idx]
        upper = bin_boundaries[idx + 1]
        if idx < num_bins - 1:
            mask = (confidences >= lower) & (confidences < upper)
        else:
            mask = (confidences >= lower) & (confidences <= upper + eps)
        count = int(mask.sum().item())
        if count == 0:
            continue
        prob = count / total
        bin_acc = accuracies[mask].float().mean()
        bin_conf = confidences[mask].mean()
        ece += prob * (bin_conf - bin_acc).abs()
    return float(ece.item())
