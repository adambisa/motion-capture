"""Synthetic frames: a plain grey scene with optional coloured squares."""

import numpy as np

from motion.common import Image

W, H = 320, 240


def scene(*squares: tuple[int, int, int, tuple[int, int, int]], base: int = 180) -> Image:
    """A grey frame with (x, y, size, bgr) squares drawn on it."""
    frame = np.full((H, W, 3), base, dtype=np.uint8)
    for x, y, size, bgr in squares:
        frame[y : y + size, x : x + size] = bgr
    return frame


def textured(size: int, seed: int = 0) -> Image:
    """A dark, patterned square: real objects have texture, a flat colour does not."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 120, (size // 8 + 1, size // 8 + 1, 1), dtype=np.uint8)
    pattern = np.repeat(np.repeat(blocks, 8, axis=0), 8, axis=1)[:size, :size]
    return np.repeat(pattern, 3, axis=2)  # 8 px blocks survive the blur


RED = (30, 30, 220)
GREEN = (40, 200, 40)
