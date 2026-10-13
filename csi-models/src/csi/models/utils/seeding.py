from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _parse_bool(val: object, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    s = str(val).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def set_all_seeds(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:  # pragma: no cover
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        import torch.backends.cudnn as cudnn  # type: ignore

        cudnn.deterministic = bool(deterministic)
        cudnn.benchmark = not bool(deterministic)
    except Exception:
        pass

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
        except Exception:
            pass
        if torch.cuda.is_available():
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    else:
        try:
            torch.use_deterministic_algorithms(False)  # type: ignore[attr-defined]
        except Exception:
            pass


def configure_from_env_and_config(cfg: Optional[dict]) -> Optional[int]:
    seed_raw = os.environ.get("SEED")
    if seed_raw is None and isinstance(cfg, dict):
        seed_raw = cfg.get("seed")
        if seed_raw is None:
            seed_raw = (cfg.get("eval", {}) or {}).get("seed")
        if seed_raw is None:
            seed_raw = (cfg.get("common", {}) or {}).get("seed")

    det_raw = os.environ.get("DETERMINISTIC")
    if det_raw is None and isinstance(cfg, dict):
        det_raw = (cfg.get("eval", {}) or {}).get("deterministic")
    deterministic = _parse_bool(det_raw, default=False)

    if seed_raw is None:
        return None
    try:
        seed = int(seed_raw)
    except Exception:
        return None

    set_all_seeds(seed, deterministic=deterministic)
    return seed


__all__ = ["configure_from_env_and_config", "set_all_seeds"]
