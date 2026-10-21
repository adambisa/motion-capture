from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from motion.objects import COCO, ROWS, YoloxLabeller, decode, letterbox

MODEL = Path("models/yolox_nano.onnx")


def test_letterbox_scales_and_pads_top_left() -> None:
    img = np.zeros((100, 200, 3), np.uint8)
    out, ratio = letterbox(img)
    assert out.shape == (416, 416, 3)
    assert ratio == pytest.approx(2.08)
    assert out[0, 0, 0] == 0  # image content top-left
    assert out[415, 0, 0] == 114  # padding below


def raw_with(
    cell: int, dx: float, dy: float, logw: float, logh: float, cls: int, score: float
) -> NDArray[np.float32]:
    raw = np.zeros((1, ROWS, 85), np.float32)
    raw[0, cell, :5] = (dx, dy, logw, logh, score)
    raw[0, cell, 5 + cls] = 1.0
    return raw


def test_decode_places_box_from_grid_cell_and_stride() -> None:
    # Row 52*52 + 3 is the stride-16 grid at (x=3, y=0).
    raw = raw_with(52 * 52 + 3, 0.5, 0.5, np.log(2), np.log(2), COCO.index("cat"), 0.9)
    (det,) = decode(raw, ratio=1.0, bounds=(416, 416))
    # centre = (3.5*16, 0.5*16) = (56, 8); size = 2*16 = 32
    assert det.name == "cat"
    assert det.score == pytest.approx(0.9)
    assert det.box == (40, 0, 32, 24)  # top clipped at 0


def test_decode_drops_low_scores_and_overlaps() -> None:
    raw = raw_with(100, 0.5, 0.5, 1, 1, 0, 0.8)
    raw[0, 101, :5] = (0.4, 0.5, 1, 1, 0.7)  # nearly the same box
    raw[0, 101, 5] = 1.0
    raw[0, 500, :5] = (0.5, 0.5, 1, 1, 0.1)  # too weak
    raw[0, 500, 5] = 1.0
    assert [d.score for d in decode(raw, 1.0, (416, 416))] == [pytest.approx(0.8)]


def test_missing_model_means_no_labeller(tmp_path: Path) -> None:
    assert YoloxLabeller.load(tmp_path / "nope.onnx") is None


@pytest.mark.skipif(not MODEL.is_file(), reason="model not downloaded")
def test_real_model_loads_and_finds_nothing_in_grey() -> None:
    labeller = YoloxLabeller.load(MODEL)
    assert labeller is not None
    assert labeller.label(np.full((120, 120, 3), 114, np.uint8)) is None
