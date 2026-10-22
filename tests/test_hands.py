from pathlib import Path

import numpy as np
import pytest

from motion.common import Image
from motion.hands import CHANNELS, STRIDE, Contract, HandBox, HandDetector, decode, letterbox
from motion.pipeline import HandPipeline


def raw_with(
    peaks: list[tuple[int, int, float, float, float]], hs: int = 48, ws: int = 64
) -> np.ndarray:
    """A network output with (cell x, cell y, score, w, h) peaks; offsets 0.5 (cell centre)."""
    raw = np.full((1, CHANNELS, hs, ws), 0.0, np.float32)
    raw[0, 0] = -10.0  # sigmoid ~ 0 everywhere
    for x, y, score, w, h in peaks:
        raw[0, 0, y, x] = np.log(score / (1 - score))
        raw[0, 1, y, x] = np.log(w / STRIDE)
        raw[0, 2, y, x] = np.log(h / STRIDE)
        raw[0, 3:5, y, x] = 0.5
    return raw


def test_decode_places_box_and_maps_back_through_the_letterbox() -> None:
    raw = raw_with([(10, 20, 0.9, 40, 60)])
    (hand,) = decode(raw, ratio=0.8, bounds=(320, 240), threshold=0.5)
    # centre (10.5, 20.5) cells * 4 px / 0.8 = (52.5, 102.5); size 50 x 75
    assert hand.box == (28, 65, 50, 75)
    assert hand.score == pytest.approx(0.9)


def test_decode_threshold_top_k_and_clipping() -> None:
    raw = raw_with([(2, 2, 0.9, 40, 40), (30, 30, 0.4, 20, 20), (50, 10, 0.7, 20, 20)])
    assert [h.score for h in decode(raw, 1.0, (256, 192), 0.5)] == pytest.approx([0.9, 0.7])
    assert len(decode(raw, 1.0, (256, 192), 0.3, max_hands=1)) == 1
    corner = decode(raw, 1.0, (256, 192), 0.85)[0]
    assert corner.box[0] == 0  # clipped at the frame edge


def test_letterbox_keeps_aspect_for_portrait_frames() -> None:
    canvas, ratio = letterbox(np.zeros((320, 180, 3), np.uint8), 192, 256)
    assert canvas.shape == (192, 256, 3)
    assert ratio == pytest.approx(0.6)
    assert canvas[0, 200, 0] == 114  # padding to the right of the image


def test_missing_model_means_no_hands(tmp_path: Path) -> None:
    assert HandDetector.load(tmp_path / "hand.onnx") is None


def test_tiny_fixture_model_loads_and_runs_on_any_frame_shape() -> None:
    detector = HandDetector.load(Path(__file__).parent / "fixtures" / "hand_tiny.onnx")
    assert detector is not None
    for shape in ((240, 320, 3), (320, 180, 3)):
        assert isinstance(detector.detect(np.zeros(shape, np.uint8)), list)


def test_contract_reads_measured_noise(tmp_path: Path) -> None:
    path = tmp_path / "hand.json"
    path.write_text('{"spawn": 0.6, "keep": 0.3, "noise": [2, 2, 3, 3], "unknown": 1}')
    c = Contract.read(path)
    assert (c.spawn, c.keep, c.noise) == (0.6, 0.3, (2.0, 2.0, 3.0, 3.0))


class FakeDetector:
    """Returns scripted detections, one list per call."""

    def __init__(self, script: list[list[HandBox]]) -> None:
        self.contract = Contract(spawn=0.6, keep=0.3, noise=(2, 2, 3, 3))
        self.script = script

    def detect(self, frame: Image, threshold: float | None = None) -> list[HandBox]:
        return self.script.pop(0) if self.script else []


def run(script: list[list[HandBox]]) -> list[list[tuple[int, str | None, str | None]]]:
    pipeline = HandPipeline(FakeDetector(script))
    frame = np.zeros((240, 320, 3), np.uint8)
    out = []
    for i in range(len(script)):
        result = pipeline.process(frame, now=i / 10)
        out.append([(t.id, t.label, t.colour) for t in result.tracks])
    return out


def test_weak_detections_never_start_a_hand() -> None:
    weak = [HandBox((100, 100, 40, 40), 0.4)]
    assert run([weak] * 5) == [[]] * 5


def test_confident_detection_starts_a_hand_and_weak_ones_keep_it() -> None:
    strong = [HandBox((100, 100, 40, 40), 0.9)]
    weak = [HandBox((104, 100, 40, 40), 0.4)]
    frames = run([strong, strong, weak, weak, weak])
    assert frames[0] == []  # seen once: not shown yet
    assert frames[1:] == [[(1, "hand", None)]] * 4  # same ID, no colour, ever


def test_one_frame_false_detection_is_never_shown() -> None:
    assert run([[HandBox((50, 50, 30, 30), 0.95)], [], [], []]) == [[]] * 4
