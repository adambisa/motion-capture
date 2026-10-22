"""HaGRID subset as training batches, with webcam-style augmentation.

Augmentation groups (ablation switches):
  geometric    zoom in and out, 4:3 crops with aspect jitter, rotation, mirror,
               and hand-free crops (so the net learns what is not a hand)
  photometric  brightness, contrast, gamma, white balance, saturation, grayscale
  webcam       motion blur, sensor noise, soft downscale, JPEG, partial occlusion
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from handnet.frames import Boxes, load_rows, read
from handnet.targets import encode
from motion.hands import PAD, letterbox

ALL_AUGMENT = frozenset({"geometric", "photometric", "webcam"})

__all__ = ["ALL_AUGMENT", "HandData", "load_rows", "normalise", "read"]


# ---- geometric -----------------------------------------------------------------


def _affine(boxes: Boxes, m: NDArray[np.float64]) -> Boxes:
    """Transform boxes by a 2x3 affine matrix: the box around the moved corners."""
    if len(boxes) == 0:
        return boxes
    x, y, w, h = boxes.T
    corners = np.stack(
        [
            np.stack([x, y], 1),
            np.stack([x + w, y], 1),
            np.stack([x, y + h], 1),
            np.stack([x + w, y + h], 1),
        ],
        1,
    )
    moved = corners @ m[:, :2].T + m[:, 2]
    lo, hi = moved.min(1), moved.max(1)
    return np.concatenate([lo, hi - lo], 1).astype(np.float32)


def _clip(boxes: Boxes, before: Boxes, width: int, height: int) -> Boxes:
    """Clip to the canvas and drop boxes that are now almost entirely outside it."""
    if len(boxes) == 0:
        return boxes
    x0 = boxes[:, 0].clip(0, width)
    y0 = boxes[:, 1].clip(0, height)
    x1 = (boxes[:, 0] + boxes[:, 2]).clip(0, width)
    y1 = (boxes[:, 1] + boxes[:, 3]).clip(0, height)
    clipped = np.stack([x0, y0, x1 - x0, y1 - y0], 1)
    visible = (clipped[:, 2] * clipped[:, 3]) / np.maximum(before[:, 2] * before[:, 3], 1e-6)
    return clipped[visible > 0.25].astype(np.float32)  # a quarter of a hand still reads as a hand


def geometric(
    image: NDArray[np.uint8], boxes: Boxes, height: int, width: int, rng: np.random.Generator
) -> tuple[NDArray[np.uint8], Boxes]:
    """One random view of the image at the network's input size."""
    ih, iw = image.shape[:2]
    if rng.random() < 0.15 and len(boxes):
        return _hand_free(image, boxes, height, width, rng)
    zoom = math.exp(rng.uniform(math.log(0.6), math.log(2.2)))
    aspect = width / height * rng.uniform(0.9, 1.1)
    # Window in source pixels; fits the long side, then zoom.
    win_h = min(ih, iw / aspect) / zoom
    win_w = win_h * aspect
    if len(boxes) and rng.random() < 0.8:  # usually keep a hand in view
        bx, by, bw, bh = boxes[rng.integers(len(boxes))]
        cx = bx + bw / 2 + rng.uniform(-0.5, 0.5) * win_w
        cy = by + bh / 2 + rng.uniform(-0.5, 0.5) * win_h
    else:
        cx, cy = rng.uniform(0, iw), rng.uniform(0, ih)
    scale = width / win_w
    angle = rng.uniform(-15, 15)
    m = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, 1.0)
    m[:, :] *= scale
    m[0, 2] += width / 2 - cx * scale
    m[1, 2] += height / 2 - cy * scale
    if rng.random() < 0.5:  # mirror
        m[0, :] *= -1
        m[0, 2] += width
    out = cv2.warpAffine(
        image,
        m,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(PAD, PAD, PAD),
    )
    moved = _affine(boxes, m)
    return out, _clip(moved, moved, width, height)


def _hand_free(
    image: NDArray[np.uint8], boxes: Boxes, height: int, width: int, rng: np.random.Generator
) -> tuple[NDArray[np.uint8], Boxes]:
    """A crop with no hand in it: faces, sleeves and walls, labelled 'nothing here'."""
    ih, iw = image.shape[:2]
    for _ in range(20):
        win_h = rng.uniform(0.3, 0.7) * ih
        win_w = min(iw, win_h * width / height)
        x0, y0 = rng.uniform(0, iw - win_w), rng.uniform(0, ih - win_h)
        bx0, by0 = boxes[:, 0] - 4, boxes[:, 1] - 4
        bx1, by1 = boxes[:, 0] + boxes[:, 2] + 4, boxes[:, 1] + boxes[:, 3] + 4
        overlap = (bx0 < x0 + win_w) & (bx1 > x0) & (by0 < y0 + win_h) & (by1 > y0)
        if not overlap.any():
            crop = image[int(y0) : int(y0 + win_h), int(x0) : int(x0 + win_w)]
            canvas, _ = letterbox(crop, height, width)
            return canvas, np.zeros((0, 4), np.float32)
    canvas, ratio = letterbox(image, height, width)  # no room for one: plain view
    return canvas, boxes * ratio


# ---- photometric and webcam ----------------------------------------------------


def photometric(image: NDArray[np.uint8], rng: np.random.Generator) -> NDArray[np.uint8]:
    x = image.astype(np.float32) / 255
    x = x * rng.uniform(0.6, 1.4) + rng.uniform(-0.15, 0.15)  # contrast, brightness
    x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.4)  # gamma
    x = x * rng.uniform(0.8, 1.2, 3)  # white balance per channel
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.5, 1.4)
    x = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.1:
        x = cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return x


def webcam(image: NDArray[np.uint8], boxes: Boxes, rng: np.random.Generator) -> NDArray[np.uint8]:
    x = image
    if rng.random() < 0.25:  # motion blur along a random direction
        k = int(rng.integers(3, 12)) | 1
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2] = 1 / k
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        kernel = cv2.warpAffine(kernel, rot, (k, k))
        x = cv2.filter2D(x, -1, kernel / max(kernel.sum(), 1e-6))
    if rng.random() < 0.2:  # soft: downscale then back up
        h, w = x.shape[:2]
        f = rng.uniform(0.4, 0.8)
        x = cv2.resize(cv2.resize(x, (int(w * f), int(h * f))), (w, h))
    if rng.random() < 0.3:  # sensor noise
        noise = rng.normal(0, rng.uniform(2, 8), x.shape)
        x = np.clip(x.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:  # the browser sends JPEGs at quality 0.7
        _, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 91))])
        x = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if rng.random() < 0.2 and len(boxes):  # something in front of part of a hand
        bx, by, bw, bh = boxes[rng.integers(len(boxes))]
        cw, ch = bw * rng.uniform(0.2, 0.5), bh * rng.uniform(0.2, 0.5)
        cx, cy = bx + rng.uniform(0, bw - cw), by + rng.uniform(0, bh - ch)
        x = x.copy()
        x[int(cy) : int(cy + ch), int(cx) : int(cx + cw)] = rng.integers(0, 256, 3)
    return x


# ---- dataset -----------------------------------------------------------------


@dataclass
class HandData(Dataset[tuple[torch.Tensor, ...]]):
    rows: list[dict]
    height: int
    width: int
    augment: frozenset[str] = frozenset()
    seed: int = 0
    epoch: int = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        rng = np.random.default_rng([self.seed, self.epoch, i])
        image, boxes = read(self.rows[i])
        if "geometric" in self.augment:
            image, boxes = geometric(image, boxes, self.height, self.width, rng)
        else:
            image, ratio = letterbox(image, self.height, self.width)
            boxes = boxes * ratio
        if "photometric" in self.augment:
            image = photometric(image, rng)
        if "webcam" in self.augment:
            image = webcam(image, boxes, rng)
        heat, reg, mask = encode(boxes, self.height, self.width)
        rgb = np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))
        return (
            torch.from_numpy(rgb),
            torch.from_numpy(heat),
            torch.from_numpy(reg),
            torch.from_numpy(mask),
        )


def normalise(batch: torch.Tensor) -> torch.Tensor:
    """uint8 RGB -> -1..1 floats, done on the training device."""
    return batch.float() / 127.5 - 1
