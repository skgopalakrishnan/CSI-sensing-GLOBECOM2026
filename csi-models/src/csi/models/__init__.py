"""Public namespace for CSI model architectures and helpers."""

from . import autoencoders
from . import core
from . import metrics
from . import model_factory as tailored
from . import supervised
from . import utils

try:
    from .archive import self_supervised
except Exception:  # pragma: no cover - optional namespace surface
    self_supervised = None

__all__ = [
    "core",
    "utils",
    "supervised",
    "metrics",
    "autoencoders",
    "tailored",
    "self_supervised",
]
