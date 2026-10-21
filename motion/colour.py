"""Name the dominant colour of a blob, from a fixed palette of twelve."""

import cv2
import numpy as np
from numpy.typing import NDArray

from motion.common import Image

COLOUR_NAMES = (
    "black", "white", "grey", "red", "orange", "yellow",
    "green", "cyan", "blue", "purple", "pink", "brown",
)  # fmt: skip

BLACK, WHITE, GREY, BROWN = (COLOUR_NAMES.index(n) for n in ("black", "white", "grey", "brown"))

# Hue upper bounds on OpenCV's 0-179 scale, checked in order. Hue is a circle,
# so red appears at both ends: below 10 and from 170 up to the wrap at 180.
_HUE_BANDS = ((10, "red"), (22, "orange"), (35, "yellow"), (85, "green"),
              (100, "cyan"), (130, "blue"), (150, "purple"), (170, "pink"),
              (180, "red"))  # fmt: skip


def classify_pixels(hsv: NDArray[np.uint8]) -> NDArray[np.intp]:
    """Map an (N, 3) array of HSV pixels to indices into COLOUR_NAMES."""
    h = hsv[:, 0].astype(np.int32)
    s = hsv[:, 1].astype(np.int32)
    v = hsv[:, 2].astype(np.int32)
    out = np.full(len(hsv), GREY, dtype=np.intp)

    lower = 0
    for upper, name in _HUE_BANDS:
        out[(h >= lower) & (h < upper)] = COLOUR_NAMES.index(name)
        lower = upper

    # Dark oranges and reds read as brown; low saturation is achromatic; very dark is black.
    out[(h < 22) & (v < 140) & (s >= 40)] = BROWN
    achromatic = s < 40
    out[achromatic] = GREY
    out[achromatic & (v > 200)] = WHITE
    out[v < 50] = BLACK
    return out


def dominant_colour(crop_bgr: Image, mask: Image | None = None) -> str:
    """Most common palette colour in the crop, counting only mask pixels if given."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    if mask is not None:
        hsv = hsv[mask.reshape(-1) > 0]
    if len(hsv) == 0:
        return "grey"
    counts = np.bincount(
        classify_pixels(np.asarray(hsv, dtype=np.uint8)), minlength=len(COLOUR_NAMES)
    )
    return COLOUR_NAMES[int(counts.argmax())]
