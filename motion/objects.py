"""Name what is moving, with YOLOX-nano (Apache-2.0) run through OpenCV's DNN module.

The detector only ever sees crops around tracks that are already moving, so the
cheap classic pipeline decides where to look and the network only says what it is.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from motion.common import Box, Image

log = logging.getLogger(__name__)

INPUT = 416
STRIDES = (8, 16, 32)

COCO = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)  # fmt: skip

ROWS = sum((INPUT // s) ** 2 for s in STRIDES)  # 3549 grid cells over the three strides
COLS = 5 + len(COCO)  # cx, cy, w, h, objectness, then one score per class


class Labeller(Protocol):
    def label(self, crop_bgr: Image) -> str | None: ...


@dataclass(frozen=True, slots=True)
class Detection:
    box: Box
    score: float
    name: str


def letterbox(img: Image, size: int = INPUT) -> tuple[Image, float]:
    """Resize to fit size x size, pasted top-left on grey (114), as YOLOX was trained."""
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    resized = cv2.resize(img, (round(w * ratio), round(h * ratio)), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return canvas, ratio


def _grid(size: int = INPUT) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    grids, strides = [], []
    for s in STRIDES:
        n = size // s
        xs, ys = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xs, ys), axis=-1).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return (np.concatenate(grids).astype(np.float32),
            np.concatenate(strides).astype(np.float32))  # fmt: skip


_GRID, _STRIDE = _grid()


def decode(
    raw: NDArray[np.float32],
    ratio: float,
    bounds: tuple[int, int],
    conf: float = 0.3,
    iou: float = 0.45,
) -> list[Detection]:
    """Turn the raw (1, ROWS, COLS) output into boxes in the original crop's pixels.

    Each row is relative to its grid cell; objectness and class scores are
    already sigmoided in the export.
    """
    rows = raw.reshape(-1, COLS)
    xy = (rows[:, :2] + _GRID) * _STRIDE
    wh = np.exp(rows[:, 2:4]) * _STRIDE
    cls = rows[:, 5:].argmax(axis=1)
    scores = rows[:, 4] * rows[np.arange(len(rows)), 5 + cls]

    keep = scores >= conf
    if not keep.any():
        return []
    xy, wh, cls, scores = xy[keep], wh[keep], cls[keep], scores[keep]

    height, width = bounds
    boxes: list[Box] = []
    for (cx, cy), (w, h) in zip(xy / ratio, wh / ratio, strict=True):
        x0, y0 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
        x1, y1 = min(width, int(cx + w / 2)), min(height, int(cy + h / 2))
        boxes.append((x0, y0, max(0, x1 - x0), max(0, y1 - y0)))

    picked = cv2.dnn.NMSBoxes(boxes, scores.tolist(), conf, iou)
    order = sorted((int(i) for i in np.asarray(picked).reshape(-1)), key=lambda i: -scores[i])
    return [Detection(boxes[i], float(scores[i]), COCO[int(cls[i])]) for i in order]


class YoloxLabeller:
    def __init__(self, net: cv2.dnn.Net) -> None:
        self._net = net

    @classmethod
    def load(cls, path: Path) -> "YoloxLabeller | None":
        """The model, or None (labels off) if it is missing or not what we expect."""
        if not path.is_file():
            log.warning("No object model at %s; object labels are off.", path)
            return None
        try:
            labeller = cls(cv2.dnn.readNetFromONNX(str(path)))
            shape = labeller._forward(np.full((INPUT, INPUT, 3), 114, np.uint8)).shape
        except cv2.error:
            log.exception("Could not load object model %s; object labels are off.", path)
            return None
        if shape != (1, ROWS, COLS):
            log.warning("Unexpected model output %s; object labels are off.", shape)
            return None
        return labeller

    def _forward(self, img: Image) -> NDArray[np.float32]:
        blob = cv2.dnn.blobFromImage(img, 1.0, (INPUT, INPUT), swapRB=False)
        self._net.setInput(blob)
        return np.asarray(self._net.forward(), dtype=np.float32)

    def detect(self, img_bgr: Image) -> list[Detection]:
        padded, ratio = letterbox(img_bgr)
        return decode(self._forward(padded), ratio, img_bgr.shape[:2])

    def label(self, crop_bgr: Image) -> str | None:
        found = self.detect(crop_bgr)
        return found[0].name if found else None
