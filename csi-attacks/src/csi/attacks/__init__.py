"""csi.attacks package

Adversarial attacks for CSI-based WiFi sensing models.

Subpackages:
- evasion: APGD/APGDT attacks and attack factory wrappers
- utils: attack/correlation helpers and metrics
"""

from . import evasion
from . import utils

try:
    from . import evaluation
except Exception:  # pragma: no cover - optional dependency chain (e.g., csi.models)
    evaluation = None  # type: ignore

try:
    from . import visualization
except Exception:  # pragma: no cover - optional plotting deps
    visualization = None  # type: ignore

__all__ = [
    "evasion",
    "utils",
    "evaluation",
    "visualization",
]
