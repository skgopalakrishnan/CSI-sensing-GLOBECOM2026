import math
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as torch_f
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import toeplitz, cholesky


_TORCH_CSI_CORR_CACHE: Dict[Tuple[Any, ...], Dict[str, torch.Tensor]] = {}


def _canonical_dataset_name(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    aliases = {
        "ntufihar": "NTU-Fi_HAR",
        "humanactivityrecognition": "HAR",
        "har": "HAR",
        "ntufihumanid": "NTU-Fi-HumanID",
        "humanidentification": "HID",
        "hid": "HID",
        # CSI-Bench tasks: treat as generic CSI profiles.
        "breathingdetection": "CSI-Bench",
        "falldetection": "CSI-Bench",
        "localization": "CSI-Bench",
        "motionsourcerecognition": "CSI-Bench",
        "proximityrecognition": "CSI-Bench",
        "rawcontinuousrecording": "CSI-Bench",
        "csibench": "CSI-Bench",
    }
    return aliases.get(key, name)


def _gaussian_kernel1d_torch(sigma: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if sigma <= 0:
        return torch.ones(1, dtype=dtype, device=device)
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.float64)
    kernel = torch.exp(-(offsets.square()) / (2.0 * float(sigma) * float(sigma)))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel.to(dtype=dtype)


def _build_fft_low_pass_mask_torch(
    length: int,
    cutoff: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(length, dtype=dtype, device=device)
    c = int(cutoff)
    if c <= 0:
        return mask
    if c >= length // 2:
        return torch.ones(length, dtype=dtype, device=device)
    mask[:c] = 1.0
    if c > 1:
        mask[-(c - 1):] = 1.0
    return mask


def _get_torch_csi_corr_profile(
    *,
    freq_len: int,
    cutoff_freq: int,
    subcarrier_spacing: float,
    tau_rms: float,
    pdp_model: str,
    sigma_freq: float,
    sigma_time: float,
    sigma_time_mimo: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    key = (
        int(freq_len),
        int(cutoff_freq),
        float(subcarrier_spacing),
        float(tau_rms),
        str(pdp_model),
        round(float(sigma_freq), 6),
        round(float(sigma_time), 6),
        round(float(sigma_time_mimo), 6),
        str(device),
        str(dtype),
    )
    cached = _TORCH_CSI_CORR_CACHE.get(key)
    if cached is not None:
        return cached

    corr_np = build_pdp_correlation_matrix(
        freq_len,
        subcarrier_spacing,
        tau_rms,
        model=pdp_model,
    )
    profile = {
        "corr_matrix": torch.tensor(corr_np, dtype=dtype, device=device),
        "fft_mask": _build_fft_low_pass_mask_torch(
            freq_len,
            cutoff_freq,
            dtype=dtype,
            device=device,
        ),
        "freq_kernel": _gaussian_kernel1d_torch(sigma_freq, dtype=dtype, device=device),
        "time_kernel": _gaussian_kernel1d_torch(sigma_time, dtype=dtype, device=device),
        "time_kernel_mimo": _gaussian_kernel1d_torch(sigma_time_mimo, dtype=dtype, device=device),
    }
    _TORCH_CSI_CORR_CACHE[key] = profile
    return profile


def _apply_conv1d_nearest_along_axis(
    x: torch.Tensor,
    kernel: torch.Tensor,
    *,
    axis: int,
) -> torch.Tensor:
    if kernel.numel() <= 1:
        return x

    radius = int(kernel.numel() // 2)
    kernel = kernel.view(1, 1, -1)

    if axis == 1:
        batch, freq_len, time_len = x.shape
        work = x.permute(0, 2, 1).reshape(batch * time_len, 1, freq_len)
        work = torch_f.pad(work, (radius, radius), mode="replicate")
        work = torch_f.conv1d(work, kernel)
        return work.reshape(batch, time_len, freq_len).permute(0, 2, 1)

    if axis == 2:
        batch, freq_len, time_len = x.shape
        work = x.reshape(batch * freq_len, 1, time_len)
        work = torch_f.pad(work, (radius, radius), mode="replicate")
        work = torch_f.conv1d(work, kernel)
        return work.reshape(batch, freq_len, time_len)

    raise ValueError(f"Unsupported axis {axis}; expected 1 or 2.")


def _project_to_budget_torch(delta: torch.Tensor, eps: float, norm: str = "linf") -> torch.Tensor:
    if norm == "linf":
        return delta.clamp(min=-float(eps), max=float(eps))
    if norm == "l2":
        flat = delta.reshape(delta.size(0), -1)
        norm_val = torch.linalg.norm(flat, ord=2, dim=1, keepdim=True)
        scale = (float(eps) / (norm_val + 1e-30)).clamp(max=1.0)
        return delta * scale.view(delta.size(0), *([1] * (delta.dim() - 1)))
    raise ValueError("norm must be 'linf' or 'l2'")


def _single_antenna_reference_scale(
    *,
    delta_tensor: torch.Tensor,
    clean_tensor: Optional[torch.Tensor],
) -> torch.Tensor:
    reference = clean_tensor if clean_tensor is not None else delta_tensor
    flat = reference[:, 0].reshape(reference.size(0), -1)
    centered = flat - flat.mean(dim=1, keepdim=True)
    denom = max(int(flat.size(1)) - 1, 1)
    var = centered.square().sum(dim=1) / float(denom)
    return torch.sqrt(var + 1e-6).view(reference.size(0), 1, 1)


def _process_single_antenna_cft_torch(
    delta_tensor: torch.Tensor,
    *,
    clean_tensor: Optional[torch.Tensor],
    cutoff_freq: int,
    subcarrier_spacing: float,
    tau_rms: float,
    pdp_model: str,
    sigma_freq: float,
    sigma_time: float,
    eps: Optional[float],
    eps_norm: str,
) -> torch.Tensor:
    if delta_tensor.ndim != 4 or delta_tensor.size(1) != 1:
        raise ValueError("single-antenna fast path expects tensors with shape (B, 1, F, T)")

    batch, _, freq_len, time_len = delta_tensor.shape
    signal = delta_tensor[:, 0]
    sigma_time_mimo = max(0.5, float(sigma_time) / 2.0) if sigma_time > 0 else 0.0
    profile = _get_torch_csi_corr_profile(
        freq_len=freq_len,
        cutoff_freq=cutoff_freq,
        subcarrier_spacing=subcarrier_spacing,
        tau_rms=tau_rms,
        pdp_model=pdp_model,
        sigma_freq=sigma_freq,
        sigma_time=sigma_time,
        sigma_time_mimo=sigma_time_mimo,
        device=signal.device,
        dtype=signal.dtype,
    )

    work = _apply_conv1d_nearest_along_axis(signal, profile["freq_kernel"], axis=1)
    work_fft = torch.fft.fft(work, dim=1)
    work = torch.fft.ifft(
        work_fft * profile["fft_mask"].view(1, freq_len, 1),
        dim=1,
    ).real.to(dtype=signal.dtype)

    delta_corr = torch.matmul(profile["corr_matrix"].unsqueeze(0), work)
    delta_corr = _apply_conv1d_nearest_along_axis(delta_corr, profile["time_kernel"], axis=2)
    delta_corr = 0.5 * delta_corr + 0.5 * (signal * _single_antenna_reference_scale(
        delta_tensor=delta_tensor,
        clean_tensor=clean_tensor,
    ))
    delta_corr = _apply_conv1d_nearest_along_axis(delta_corr, profile["time_kernel_mimo"], axis=2)

    if eps is not None:
        delta_corr = _project_to_budget_torch(delta_corr, eps=eps, norm=eps_norm)

    return torch.nan_to_num(delta_corr, nan=0.0, posinf=0.0, neginf=0.0).view(batch, 1, freq_len, time_len)


def _estimate_rx_covariance_torch(reference_tensor: torch.Tensor) -> torch.Tensor:
    if reference_tensor.ndim != 4:
        raise ValueError("reference_tensor must have shape (B, C, F, T)")
    batch, antennas, freq_len, time_len = reference_tensor.shape
    work_dtype = torch.float32 if reference_tensor.dtype in {torch.float16, torch.bfloat16} else reference_tensor.dtype
    flattened = reference_tensor.to(dtype=work_dtype).permute(0, 2, 3, 1).reshape(batch, freq_len * time_len, antennas)
    centered = flattened - flattened.mean(dim=1, keepdim=True)
    denom = max(int(freq_len * time_len) - 1, 1)
    cov = torch.matmul(centered.transpose(1, 2), centered) / float(denom)
    eye = torch.eye(antennas, dtype=work_dtype, device=reference_tensor.device).unsqueeze(0)
    cov = cov + (1e-6 * eye)
    try:
        return torch.linalg.cholesky(cov)
    except RuntimeError:
        return torch.linalg.cholesky(cov + (1e-4 * eye))


def _correlate_mimo_cholesky_torch(delta_tensor: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
    if delta_tensor.ndim != 4:
        raise ValueError("delta_tensor must have shape (B, C, F, T)")
    delta_work = delta_tensor.to(dtype=chol.dtype).permute(0, 2, 3, 1)
    correlated = torch.einsum("bfta,bca->bftc", delta_work, chol)
    return correlated.permute(0, 3, 1, 2).to(dtype=delta_tensor.dtype)


def _process_multi_antenna_cft_torch(
    delta_tensor: torch.Tensor,
    *,
    clean_tensor: Optional[torch.Tensor],
    eps: Optional[float],
    eps_norm: str,
) -> torch.Tensor:
    if delta_tensor.ndim != 4 or delta_tensor.size(1) <= 1:
        raise ValueError("multi-antenna fast path expects tensors with shape (B, C>1, F, T)")

    reference = clean_tensor if clean_tensor is not None else delta_tensor
    chol = _estimate_rx_covariance_torch(reference)
    delta_corr = _correlate_mimo_cholesky_torch(delta_tensor, chol)
    if eps is not None:
        delta_corr = _project_to_budget_torch(delta_corr, eps=eps, norm=eps_norm)
    return torch.nan_to_num(delta_corr, nan=0.0, posinf=0.0, neginf=0.0)


def estimate_tau_rms_from_deltafreq(deltafreqs_hz: np.ndarray, empirical_corr: np.ndarray) -> float:
    """Estimate tau_rms (RMS delay spread) from an empirical frequency correlation vector."""
    corr = np.abs(np.asarray(empirical_corr))
    if corr.ndim != 1:
        raise ValueError("empirical_corr must be 1D")
    df = np.asarray(deltafreqs_hz)
    if df.shape != corr.shape:
        raise ValueError("deltafreqs_hz and empirical_corr must have the same shape")

    mid_idx = int(np.argmax(corr))
    corr_pos = corr[mid_idx:]
    df_pos = df[mid_idx:]
    if corr_pos.size == 0:
        return 100e-9

    below_half = np.where(corr_pos <= 0.5)[0]
    if below_half.size == 0:
        return 100e-9

    df_half = df_pos[below_half[0]]
    if df_half <= 0:
        return 100e-9

    tau_rms = np.sqrt(np.log(2.0)) / (2.0 * np.pi * df_half)
    return float(tau_rms)


def fft_low_pass_filter(x: np.ndarray, cutoff: int) -> np.ndarray:
    """Low-pass filter a 1D vector in the frequency domain keeping the lowest ``cutoff`` bins."""
    X = np.fft.fft(np.asarray(x))
    N = X.size
    c = int(cutoff)
    if c <= 0:  # no frequencies kept
        return np.zeros_like(x, dtype=float)
    if c >= N // 2:  # all frequencies kept
        return np.real(np.fft.ifft(X))

    mask = np.zeros(N, dtype=float)
    mask[:c] = 1.0  # keep DC + next (c-1) positive freqs
    if c > 1:
        mask[-(c - 1):] = 1.0  # keep matching negative freqs (exclude duplicate Nyquist pair)
    return np.real(np.fft.ifft(X * mask))


def build_pdp_correlation_matrix(num_subcarriers: int,
                                 subcarrier_spacing: float,
                                 tau_rms: float,
                                 model: str = "gaussian") -> np.ndarray:
    """Construct a Toeplitz correlation matrix driven by a parametric PDP model."""
    dk = np.arange(num_subcarriers)
    df = dk * float(subcarrier_spacing)

    if model == "gaussian":
        r = np.exp(-(2.0 * np.pi * tau_rms * df) ** 2)
    elif model == "lorentzian":
        r = 1.0 / np.sqrt(1.0 + (2.0 * np.pi * tau_rms * df) ** 2)
    else:
        raise ValueError("model must be 'gaussian' or 'lorentzian'")

    C = toeplitz(r)
    diag = np.sqrt(np.diag(C))
    C /= np.outer(diag, diag) + 1e-30
    return C


def correlate_mimo_cholesky(delta_mimo: np.ndarray, R_rx: np.ndarray) -> np.ndarray:
    """Correlate MIMO perturbations with the Cholesky factor of ``R_rx``."""
    delta_mimo = np.asarray(delta_mimo)
    R_rx = np.asarray(R_rx)

    if delta_mimo.ndim != 3:
        raise ValueError("delta_mimo must have shape (freq_bins, time_steps, num_antennas)")
    freq_len, time_len, num_ant = delta_mimo.shape
    if R_rx.shape != (num_ant, num_ant):
        raise ValueError("R_rx shape mismatch with delta_mimo antennas")

    try:
        L = cholesky(R_rx, lower=True)
    except np.linalg.LinAlgError:
        jitter = 1e-6 * np.eye(num_ant)
        L = cholesky(R_rx + jitter, lower=True)

    out = np.empty_like(delta_mimo)
    for k in range(freq_len):
        out[k, :, :] = delta_mimo[k, :, :] @ L.T
    return out


def project_to_budget(delta: np.ndarray, eps: float, norm: str = "linf") -> np.ndarray:
    """Project ``delta`` to an L_inf or L_2 ball of radius ``eps``."""
    delta = np.asarray(delta)
    if norm == "linf":
        return np.clip(delta, -eps, eps)
    if norm == "l2":
        flat = delta.reshape(1, -1)
        norm_val = np.linalg.norm(flat, axis=1, keepdims=True)
        if norm_val.squeeze() <= eps:
            return delta
        scaled = delta * (eps / (norm_val.squeeze() + 1e-30))
        return scaled
    raise ValueError("norm must be 'linf' or 'l2'")


def _estimate_rx_covariance(delta_mimo: np.ndarray) -> np.ndarray:
    freq_len, time_len, num_ant = delta_mimo.shape
    flattened = delta_mimo.reshape(freq_len * time_len, num_ant)
    flattened -= np.mean(flattened, axis=0, keepdims=True)
    R_emp = np.cov(flattened, rowvar=False)
    return R_emp + 1e-6 * np.eye(num_ant)


def correlation_preservation_pipeline(delta_matrix: np.ndarray,
                                      delta_mimo: Optional[np.ndarray] = None,
                                      clean_mimo: Optional[np.ndarray] = None,
                                      subcarrier_spacing: float = 312.5e3,
                                      tau_rms: Optional[float] = None,
                                      pdp_model: str = "gaussian",
                                      cutoff_freq: int = 5,
                                      sigma_freq: float = 0.8,
                                      sigma_time: float = 1.0,
                                      R_rx: Optional[np.ndarray] = None,
                                      eps: Optional[float] = None,
                                      eps_norm: str = "linf",
                                      fast_mode: bool = False
                                      ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Principled correlation-preservation pipeline with optional MIMO support.

    Args:
        delta_matrix: (freq_bins, time_steps) perturbation slice.
        delta_mimo: optional (freq_bins, time_steps, num_antennas) perturbation.
        clean_mimo: optional clean CSI sample with shape matching ``delta_mimo`` used
            for antenna covariance estimation. Falls back to ``delta_mimo`` when omitted.
    """
    delta = np.asarray(delta_matrix)
    if delta.ndim != 2:
        raise ValueError("delta_matrix must be 2D (freq_bins, time_steps)")

    freq_len, time_len = delta.shape

    if tau_rms is None:
        tau_rms = 100e-9

    apply_freq_smoothing = (not fast_mode) and sigma_freq > 0
    if apply_freq_smoothing:
        delta_freq = gaussian_filter1d(delta, sigma=sigma_freq, axis=0, mode="nearest")
    else:
        delta_freq = delta.copy()

    delta_filtered = np.zeros_like(delta_freq)
    if cutoff_freq is None:
        cutoff_freq = freq_len // 2
    for t in range(time_len):
        delta_filtered[:, t] = fft_low_pass_filter(delta_freq[:, t], cutoff_freq)

    C = build_pdp_correlation_matrix(freq_len, subcarrier_spacing, tau_rms, model=pdp_model)
    delta_corr = C @ delta_filtered

    apply_time_smoothing = (not fast_mode) and sigma_time > 0
    if apply_time_smoothing:
        delta_corr = gaussian_filter1d(delta_corr, sigma=sigma_time, axis=1, mode="nearest")

    delta_mimo_corr = None
    if delta_mimo is not None:
        delta_mimo = np.asarray(delta_mimo)
        if delta_mimo.ndim != 3:
            raise ValueError("delta_mimo must be 3D (freq_bins, time_steps, num_antennas)")
        if delta_mimo.shape[:2] != (freq_len, time_len):
            raise ValueError("delta_mimo spatial dims must match delta_matrix")

        if R_rx is None:
            reference = clean_mimo if clean_mimo is not None else delta_mimo
            reference = np.asarray(reference)
            if reference.shape != delta_mimo.shape:
                raise ValueError("clean_mimo must match delta_mimo shape")
            R_rx = _estimate_rx_covariance(reference)
        delta_mimo_corr = correlate_mimo_cholesky(delta_mimo, R_rx)
        delta_corr_from_mimo = np.mean(delta_mimo_corr, axis=2)
        delta_corr = 0.5 * delta_corr + 0.5 * delta_corr_from_mimo
        if apply_time_smoothing:
            delta_corr = gaussian_filter1d(delta_corr, sigma=max(0.5, sigma_time / 2.0),
                                           axis=1, mode="nearest")

    if eps is not None:
        delta_corr = project_to_budget(delta_corr, eps, norm=eps_norm)
        if delta_mimo_corr is not None:
            delta_mimo_corr = project_to_budget(delta_mimo_corr, eps, norm=eps_norm)

    if delta_mimo is not None:
        return delta_corr, delta_mimo_corr # type: ignore
    return delta_corr


def process_csi_tensor_for_correlation(delta_tensor: torch.Tensor,
                                       dataset_name: str,
                                       clean_tensor: Optional[torch.Tensor] = None,
                                       subcarrier_spacing: float = 312.5e3,
                                       tau_rms_defaults: Optional[Dict[str, float]] = None,
                                       pdp_model: str = "gaussian",
                                       cutoff_freq_map: Optional[Dict[str, int]] = None,
                                       eps: Optional[float] = None,
                                       eps_norm: str = "linf",
                                       device: Optional[torch.device] = None
                                       ) -> torch.Tensor:
    """Process a batch of CSI perturbations using :func:`correlation_preservation_pipeline`.

    Args:
        delta_tensor: perturbation batch.
        dataset_name: dataset identifier.
        clean_tensor: optional clean (unperturbed) batch with the same shape used to estimate
            antenna covariance. Required to benefit from clean-driven covariance.

    The CSI tensors are expected to follow the CFT layout (channels/antennas, subcarriers, time).
    """
    if delta_tensor.ndim != 4:
        raise ValueError(f"delta_tensor must be 4D with a batch dimension; got shape {tuple(delta_tensor.shape)}")
    if clean_tensor is not None and clean_tensor.shape != delta_tensor.shape:
        raise ValueError("clean_tensor shape must match delta_tensor shape")
    if device is None:
        device = delta_tensor.device

    dataset_name = _canonical_dataset_name(dataset_name)

    if tau_rms_defaults is None:
        tau_rms_defaults = {
            "NTU-Fi_HAR": 120e-9,
            "NTU-Fi-HumanID": 120e-9,
            "HAR": 120e-9,
            "HID": 120e-9,
            "CSI-Bench": 120e-9,
        }
    if cutoff_freq_map is None:
        cutoff_freq_map = {
            "NTU-Fi_HAR": 5,
            "NTU-Fi-HumanID": 5,
            "HAR": 5,
            "HID": 5,
            "CSI-Bench": 5,
        }

    cft_specs: Dict[str, Tuple[Iterable[int], Iterable[int], Iterable[int]]] = {
        # Dataset-specific supported CFT layouts.
        "NTU-Fi_HAR": ({3}, {114}, {250, 500}),
        # Historical NTU-Fi HumanID variants appear as either 30 or 114 subcarriers.
        "NTU-Fi-HumanID": ({3}, {30, 114}, {250, 500}),
        # Current campaign CSI-Bench paper setup:
        # C=1 antenna/channel, F=56 subcarriers, T=250 timesteps.
        "HAR": ({1}, {56}, {250}),
        "HID": ({1}, {56}, {250}),
        "CSI-Bench": ({1}, {56}, {250}),
    }

    profile_name = dataset_name if dataset_name in cft_specs else "CSI-Bench"
    allowed_antennas, subcarrier_options, timestep_options = cft_specs[profile_name]
    antennas = int(delta_tensor.size(1))
    subcarriers = int(delta_tensor.size(2))
    timesteps = int(delta_tensor.size(3))
    if antennas not in set(allowed_antennas):
        raise ValueError(
            f"{dataset_name}: expected antennas in {sorted(set(allowed_antennas))}, got {antennas}."
        )
    if subcarrier_options and (subcarriers not in subcarrier_options):
        raise ValueError(
            f"{dataset_name}: expected subcarriers in {sorted(subcarrier_options)}, got {subcarriers}."
        )
    if timestep_options and (timesteps not in set(timestep_options)):
        raise ValueError(
            f"{dataset_name}: expected timesteps in {sorted(set(timestep_options))}, got {timesteps}."
        )

    if antennas == 1:
        tau_rms = tau_rms_defaults.get(profile_name, tau_rms_defaults.get("CSI-Bench", 100e-9))
        cutoff_default = cutoff_freq_map.get(profile_name, cutoff_freq_map.get("CSI-Bench", 5))
        cutoff = min(int(cutoff_default), max(int(subcarriers // 2), 1))
        return _process_single_antenna_cft_torch(
            delta_tensor.to(device=device),
            clean_tensor=clean_tensor.to(device=device, dtype=delta_tensor.dtype) if clean_tensor is not None else None,
            cutoff_freq=cutoff,
            subcarrier_spacing=subcarrier_spacing,
            tau_rms=tau_rms,
            pdp_model=pdp_model,
            sigma_freq=0.8,
            sigma_time=1.0,
            eps=eps,
            eps_norm=eps_norm,
        ).to(device=device, dtype=delta_tensor.dtype)

    return _process_multi_antenna_cft_torch(
        delta_tensor.to(device=device),
        clean_tensor=clean_tensor.to(device=device, dtype=delta_tensor.dtype) if clean_tensor is not None else None,
        eps=eps,
        eps_norm=eps_norm,
    ).to(device=device, dtype=delta_tensor.dtype)
