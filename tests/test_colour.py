import numpy as np
import pytest

from motion.colour import COLOUR_NAMES, dominant_colour

PATCHES = {
    "black": (10, 10, 10),
    "white": (245, 245, 245),
    "grey": (128, 128, 128),
    "red": (30, 30, 220),
    "orange": (0, 140, 255),
    "yellow": (0, 230, 230),
    "green": (40, 200, 40),
    "cyan": (220, 220, 0),
    "blue": (220, 60, 20),
    "purple": (200, 40, 140),
    "pink": (180, 80, 255),
    "brown": (30, 70, 120),
}


def patch(bgr: tuple[int, int, int]) -> np.ndarray:
    return np.full((20, 20, 3), bgr, dtype=np.uint8)


def test_every_palette_colour_has_a_case() -> None:
    assert set(PATCHES) == set(COLOUR_NAMES)


@pytest.mark.parametrize(("name", "bgr"), PATCHES.items())
def test_solid_patch_is_named(name: str, bgr: tuple[int, int, int]) -> None:
    assert dominant_colour(patch(bgr)) == name


def test_majority_wins() -> None:
    crop = patch(PATCHES["green"])
    crop[:, :5] = PATCHES["red"]
    assert dominant_colour(crop) == "green"


def test_mask_restricts_to_the_blob() -> None:
    crop = patch(PATCHES["green"])
    crop[5:15, 5:15] = PATCHES["red"]
    mask = np.zeros((20, 20), np.uint8)
    mask[5:15, 5:15] = 255
    assert dominant_colour(crop, mask) == "red"


def test_empty_mask_falls_back_to_grey() -> None:
    assert dominant_colour(patch(PATCHES["red"]), np.zeros((20, 20), np.uint8)) == "grey"
