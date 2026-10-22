"""Training targets: what the network should output for a set of hand boxes.

At 1/4 resolution, each hand puts an elliptical Gaussian bump (peak exactly 1)
on the heatmap at its centre cell, sized to the box (TTFNet's sigma = 0.54 *
size / 6).

Size and offset are supervised only at the centre cell (original CenterNet).
Supervising every cell near the centre (TTFNet style) was tried and was worse
for this small network: see docs/experiments.md.
"""

import numpy as np
from numpy.typing import NDArray

from motion.hands import STRIDE

ALPHA = 0.54
MIN_SIDE = 6.0  # px in the input: smaller hands are dropped from the targets


def encode(
    boxes: NDArray[np.float32], height: int, width: int
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """boxes: (N, 4) x, y, w, h in input pixels.

    Returns heat (Hs, Ws), regression (4, Hs, Ws) = log(w/4), log(h/4), dx, dy,
    and weights (Hs, Ws) for the regression loss: 1 at each centre, else 0.
    """
    hs, ws = height // STRIDE, width // STRIDE
    heat = np.zeros((hs, ws), np.float32)
    reg = np.zeros((4, hs, ws), np.float32)
    weight = np.zeros((hs, ws), np.float32)
    yy, xx = np.mgrid[0:hs, 0:ws].astype(np.float32)
    for x, y, w, h in boxes:
        if min(w, h) < MIN_SIDE:
            continue
        cx, cy = (x + w / 2) / STRIDE, (y + h / 2) / STRIDE
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < ws and 0 <= iy < hs):
            continue
        sx = max(0.8, ALPHA * w / STRIDE / 6)
        sy = max(0.8, ALPHA * h / STRIDE / 6)
        bump = np.exp(-((xx - ix) ** 2) / (2 * sx**2) - ((yy - iy) ** 2) / (2 * sy**2))
        np.maximum(heat, bump, out=heat)
        reg[:, iy, ix] = (np.log(w / STRIDE), np.log(h / STRIDE), cx - ix, cy - iy)
        weight[iy, ix] = 1.0
    return heat, reg, weight
