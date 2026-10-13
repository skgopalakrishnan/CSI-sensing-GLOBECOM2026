import sys
from pathlib import Path


# Ensure the csi-data src is on sys.path for imports during tests without installation
ROOT = Path(__file__).resolve().parents[1]  # .../csi-data
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

