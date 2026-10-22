"""The dataset as the live demo would see it. No PyTorch, so baselines can use it too.

Every evaluated image is downscaled to 320 px on the long side and JPEG-compressed
at quality 70, like the browser's frames, before any detector sees it.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

DATA = Path(__file__).resolve().parent.parent / "data"
FRAME_SIDE = 320
JPEG_QUALITY = 70

Boxes = NDArray[np.float32]  # (N, 4) x, y, w, h in pixels


def load_rows(split: str, fraction: float = 1.0) -> list[dict]:
    rows = [json.loads(line) for line in (DATA / "manifest.jsonl").open()]
    rows = [r for r in rows if r["split"] == split]
    if fraction < 1:
        rows = rows[:: max(1, round(1 / fraction))]
    return rows


def read(row: dict) -> tuple[NDArray[np.uint8], Boxes]:
    image = cv2.imread(str(DATA / "images" / f"{row['key']}.jpg"), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    boxes = np.array(row["boxes"], np.float32).reshape(-1, 4) * [w, h, w, h]
    return image, boxes


def as_frame(image: NDArray[np.uint8], boxes: Boxes) -> tuple[NDArray[np.uint8], Boxes]:
    """What the server would receive: 320 px long side, browser-quality JPEG."""
    h, w = image.shape[:2]
    scale = FRAME_SIDE / max(h, w)
    frame = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), boxes * scale


def frames(split: str) -> list[tuple[dict, NDArray[np.uint8], Boxes]]:
    return [(row, *as_frame(*read(row))) for row in load_rows(split)]
