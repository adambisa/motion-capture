"""Small shared types."""

import numpy as np
from numpy.typing import NDArray

Box = tuple[int, int, int, int]
"""x, y, width, height in frame pixels."""

Point = tuple[int, int]
Image = NDArray[np.uint8]
