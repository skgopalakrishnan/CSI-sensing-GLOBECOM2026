import torch
from typing import Optional
from torch import device as torch_device
from torch.cuda import is_available as is_cuda_available


def _print_gpu_memory_usage():
    """Print current GPU memory usage if CUDA is available."""
    if is_cuda_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # Convert to GB
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def _optimize_gpu_settings():
    """Set optimal GPU settings for adversarial attack computations."""
    if is_cuda_available():
        # Enable cudnn benchmarking for consistent input shapes
        torch.backends.cudnn.benchmark = True
        # Enable cudnn deterministic for reproducible results (optional, may slow down)
        # torch.backends.cudnn.deterministic = True
        print("CUDA is available. Optimized GPU settings applied.")
    else:
        print("CUDA is not available. Running on CPU.")


def _setup_device() -> torch_device:
    device = torch_device("cuda" if is_cuda_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)} ("
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)"
        )
        _optimize_gpu_settings()
        # Prefer high-throughput math on Ampere+ (TF32)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        _print_gpu_memory_usage()
    # Allow faster matmul kernels where safe (PyTorch 2.x)
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    return device


def _configure_cuda_runtime() -> None:
    if not is_cuda_available():
        return
    try:
        deterministic = bool(torch.backends.cudnn.deterministic)
        torch.backends.cudnn.benchmark = not deterministic
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _resolve_device(requested: Optional[str]) -> torch_device:
    if requested is None:
        requested = "cuda" if is_cuda_available() else "cpu"
    if requested == "cuda" and not is_cuda_available():
        print("[WARN] CUDA requested but not available; falling back to CPU")
        requested = "cpu"
    resolved = torch_device(requested)
    if resolved.type == "cuda":
        _configure_cuda_runtime()
    return resolved
