"""HandNet: a small, self-trained hand detector for the motion-capture demo.

Training code lives here; the runtime lives in ../motion/hands.py and is
imported from there so training evaluates exactly what the server runs.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
