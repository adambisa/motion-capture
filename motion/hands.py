"""Find hands with HandNet, a small detector trained for this project (see train/).

The network sees one frame, letterboxed into a fixed canvas, and outputs a
1/4-resolution map with 5 channels: a hand-centre heatmap logit, the box size
as log(w/4), log(h/4), and the centre's offset inside its cell. Decoding is
CenterNet's: local maxima of the heatmap are hands, the other channels give
their boxes. This module is shared by the server and by training's evaluation,
so what gets measured is exactly what runs.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from motion.common import Box, Image

log = logging.getLogger(__name__)

STRIDE = 4
CHANNELS = 5
PAD = 114  # grey letterbox padding, as in training


@dataclass(frozen=True, slots=True)
class HandBox:
    box: Box
    score: float


@dataclass(frozen=True, slots=True)
class Contract:
    """What the weights were trained for, shipped next to the .onnx as JSON."""

    height: int = 192
    width: int = 256
    spawn: float = 0.5  # score to start a new track
    keep: float = 0.25  # score to keep an existing one going
    max_hands: int = 6
    # Measured wobble of the detector's boxes (std of centre x, y, width, height
    # in px), used as the Kalman filter's measurement noise; None means defaults.
    noise: tuple[float, float, float, float] | None = None

    @classmethod
    def read(cls, path: Path) -> "Contract":
        fields = {k: v for k, v in json.loads(path.read_text()).items() if k in cls.__slots__}
        if fields.get("noise") is not None:
            fields["noise"] = tuple(float(v) for v in fields["noise"])
        return cls(**fields)


def letterbox(frame: Image, height: int, width: int) -> tuple[Image, float]:
    """Scale to fit height x width, pasted top-left on grey. Returns the canvas and the scale."""
    h, w = frame.shape[:2]
    ratio = min(height / h, width / w)
    size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
    resized = cv2.resize(
        frame, size, interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    )
    canvas = np.full((height, width, 3), PAD, dtype=np.uint8)
    canvas[: size[1], : size[0]] = resized
    return canvas, ratio


def to_blob(canvas: Image) -> NDArray[np.float32]:
    """RGB scaled to -1..1, as (1, 3, H, W): what the network was trained on."""
    blob = cv2.dnn.blobFromImage(canvas, 1 / 127.5, mean=(127.5, 127.5, 127.5), swapRB=True)
    return np.asarray(blob, dtype=np.float32)


def decode(
    raw: NDArray[np.float32],
    ratio: float,
    bounds: tuple[int, int],
    threshold: float,
    max_hands: int = 6,
) -> list[HandBox]:
    """Boxes in the original frame's pixels from the (1, 5, H/4, W/4) output.

    `bounds` is the original frame's (width, height).
    """
    out = raw.reshape(CHANNELS, raw.shape[-2], raw.shape[-1])
    heat = 1 / (1 + np.exp(-out[0]))
    # A cell is a hand centre if it is the largest in its 3x3 neighbourhood.
    peaks = (heat == cv2.dilate(heat, np.ones((3, 3), np.uint8))) & (heat >= threshold)
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-heat[ys, xs])[:max_hands]
    width, height = bounds
    found = []
    for y, x in zip(ys[order], xs[order], strict=True):
        cx = (x + out[3, y, x]) * STRIDE / ratio
        cy = (y + out[4, y, x]) * STRIDE / ratio
        w = np.exp(out[1, y, x]) * STRIDE / ratio
        h = np.exp(out[2, y, x]) * STRIDE / ratio
        x0, y0 = max(0, round(cx - w / 2)), max(0, round(cy - h / 2))
        x1, y1 = min(width, round(cx + w / 2)), min(height, round(cy + h / 2))
        if x1 > x0 and y1 > y0:
            found.append(HandBox((x0, y0, x1 - x0, y1 - y0), float(heat[y, x])))
    return found


class HandDetector:
    """HandNet with one network per canvas orientation.

    The weights are fully convolutional, so the same training runs on a
    landscape canvas (hand.onnx) and, if exported, a portrait one
    (hand_portrait.onnx). Each frame uses the canvas matching its orientation,
    so a phone's portrait frame is not shrunk into a landscape box.
    """

    def __init__(self, nets: dict[tuple[int, int], cv2.dnn.Net], contract: Contract) -> None:
        self._nets = nets  # (canvas height, canvas width) -> network
        self.contract = contract

    @property
    def canvases(self) -> list[tuple[int, int]]:
        return sorted(self._nets)

    @classmethod
    def load(cls, onnx: Path) -> "HandDetector | None":
        """The detector, or None (hands off) if the model is missing or not what we expect."""
        meta = onnx.with_suffix(".json")
        if not onnx.is_file() or not meta.is_file():
            log.warning("No hand model at %s; hand tracking is off.", onnx)
            return None
        try:
            c = Contract.read(meta)
            files = {(c.height, c.width): onnx}
            portrait = onnx.with_name(f"{onnx.stem}_portrait{onnx.suffix}")
            if portrait.is_file():
                files[(c.width, c.height)] = portrait
            detector = cls({size: cv2.dnn.readNetFromONNX(str(f)) for size, f in files.items()}, c)
            for h, w in files:
                shape = detector.forward(np.full((h, w, 3), PAD, np.uint8)).shape
                expected = (1, CHANNELS, h // STRIDE, w // STRIDE)
                if shape != expected:
                    log.warning("Hand model output %s, expected %s; hands off.", shape, expected)
                    return None
        except (cv2.error, ValueError, TypeError):
            log.exception("Could not load hand model %s; hand tracking is off.", onnx)
            return None
        return detector

    def forward(self, canvas: Image) -> NDArray[np.float32]:
        net = self._nets[(canvas.shape[0], canvas.shape[1])]
        net.setInput(to_blob(canvas))
        return np.asarray(net.forward(), dtype=np.float32)

    def canvas_for(self, frame: Image) -> tuple[int, int]:
        """Portrait canvas for portrait frames when one was exported, else landscape."""
        c = self.contract
        tall = frame.shape[0] > frame.shape[1]
        return (
            (c.width, c.height)
            if tall and (c.width, c.height) in self._nets
            else (c.height, c.width)
        )

    def detect(self, frame: Image, threshold: float | None = None) -> list[HandBox]:
        c = self.contract
        height, width = self.canvas_for(frame)
        canvas, ratio = letterbox(frame, height, width)
        bounds = (frame.shape[1], frame.shape[0])
        return decode(
            self.forward(canvas), ratio, bounds, c.keep if threshold is None else threshold
        )
