from __future__ import annotations

import math
from typing import Optional

import torch


def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Inject i.i.d. Gaussian noise with std=sigma."""
    if not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor.")
    sigma_val = float(sigma)
    if sigma_val < 0.0:
        raise ValueError("sigma must be >= 0.")
    if sigma_val == 0.0:
        return x
    return x + (torch.randn_like(x) * sigma_val)


def add_amplitude_noise(
    csi_amplitude: torch.Tensor,
    sigma: float,
    *,
    clamp_min: Optional[float] = 0.0,
) -> torch.Tensor:
    """Inject amplitude noise under high-SNR Rician approximation.

    For amplitude-only CSI, physical RF noise is approximated as:
        amplitude_noise_std = sigma / sqrt(2)
    By default outputs are clamped to keep amplitudes non-negative.
    Set `clamp_min=None` to disable clamping.
    """
    if not torch.is_tensor(csi_amplitude):
        raise TypeError("csi_amplitude must be a torch.Tensor.")
    sigma_val = float(sigma)
    if sigma_val < 0.0:
        raise ValueError("sigma must be >= 0.")
    if sigma_val == 0.0:
        if clamp_min is None:
            return csi_amplitude
        return torch.clamp(csi_amplitude, min=float(clamp_min))

    amp_sigma = sigma_val / math.sqrt(2.0)
    noise = torch.randn_like(csi_amplitude) * amp_sigma
    noisy = csi_amplitude + noise
    if clamp_min is None:
        return noisy
    return torch.clamp(noisy, min=float(clamp_min))


def add_noise(
    x: torch.Tensor,
    sigma: float,
    *,
    noise_type: str = "gaussian",
    amplitude_clamp_min: Optional[float] = 0.0,
) -> torch.Tensor:
    """Dispatch helper for noise injection."""
    key = str(noise_type).strip().lower()
    if key in {"gaussian", "normal", "iid", "iid_gaussian"}:
        return add_gaussian_noise(x, sigma)
    if key in {"amplitude", "amp", "amplitude_only", "rician_high_snr", "rf_amplitude"}:
        return add_amplitude_noise(
            x,
            sigma,
            clamp_min=None if amplitude_clamp_min is None else float(amplitude_clamp_min),
        )
    raise ValueError(f"Unsupported noise_type={noise_type!r}. Use 'gaussian' or 'amplitude'.")
