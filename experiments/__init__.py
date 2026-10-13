from __future__ import annotations

import sys
from pathlib import Path


def _prepend_local_src(path: Path) -> None:
    if path.exists():
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


_REPO_ROOT = Path(__file__).resolve().parents[1]

# Prefer the workspace sources over any older installed csi-* packages.
for _src in (
    _REPO_ROOT / "csi-models" / "src",
    _REPO_ROOT / "csi-data" / "src",
    _REPO_ROOT / "csi-attacks" / "src",
    _REPO_ROOT / "csi-defenses" / "src",
):
    _prepend_local_src(_src)
