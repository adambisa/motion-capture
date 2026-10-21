"""Adaptive background: a running average the scene slowly melts into.

v1 diffed every frame against the very first one, so a moved chair or a cloud
in front of the sun meant "motion" forever. Here the background keeps learning,
and sudden whole-frame changes (lights, auto-exposure) are absorbed fast.

It starts from the per-pixel median of the first `warmup` frames, so anything
that moves during that first second drops out instead of being smeared in.

A plain running average soaks up whatever passes through, so a person walking
away leaves a "ghost" of themselves behind. So the learning rate depends on
whether something is really moving there:

- foreground learns at the much slower `absorb`, so the background under a
  moving person stays clean, including flat-coloured parts of them that show no
  frame-to-frame change as they slide past;
- foreground that has had no change within REACH px for `still_frames` frames in a
  row (a ghost of something that left, or something that parked) learns at the
  fast `heal` and fades in within about a second, even when it touches a mover;
- quiet background learns at `alpha`.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from motion.common import Box, Image

BLUR = (11, 11)
KERNEL = np.ones((3, 3), np.uint8)
REACH = 12  # px around frame-to-frame change that count as "something moving here"
NEAR_CHANGE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * REACH + 1, 2 * REACH + 1))


def to_gray(frame_bgr: Image) -> Image:
    """Grayscale plus blur, so sensor noise doesn't look like motion."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return np.asarray(cv2.GaussianBlur(gray, BLUR, 0), dtype=np.uint8)


def _threshold(a: Image, b: Image, threshold: int) -> Image:
    _, mask = cv2.threshold(cv2.absdiff(a, b), threshold, 255, cv2.THRESH_BINARY)
    return np.asarray(mask, dtype=np.uint8)


class Background:
    def __init__(
        self,
        alpha: float = 0.05,
        absorb: float = 0.003,
        heal: float = 0.2,
        still_frames: int = 8,
        warmup: int = 10,
        flood: float = 0.6,
    ) -> None:
        self.alpha = alpha
        self.absorb = absorb
        self.heal = heal
        self.still_frames = still_frames
        self.warmup = warmup
        self.flood = flood
        self._model: NDArray[np.float32] | None = None
        self._previous: Image | None = None
        self._still: NDArray[np.uint16] | None = None  # frames without change nearby
        self._warmup_frames: list[Image] = []
        self._seen = 0

    def reset(self) -> None:
        self._model = None
        self._previous = None
        self._still = None
        self._warmup_frames = []
        self._seen = 0

    def apply(self, gray: Image, threshold: int) -> Image:
        """Return a 0/255 mask of pixels that differ from the background."""
        if (
            self._model is None
            or self._previous is None
            or self._still is None
            or self._model.shape != gray.shape
        ):
            self._model = gray.astype(np.float32)
            self._previous = gray
            self._still = np.zeros(gray.shape, np.uint16)
            self._warmup_frames = []
            self._seen = 0
        self._seen += 1

        if self._seen <= self.warmup:
            self._warmup_frames.append(gray)
            median = np.median(np.stack(self._warmup_frames), axis=0)
            self._model = median.astype(np.float32)
            self._previous = gray
            return np.zeros_like(gray)
        self._warmup_frames = []
        previous, self._previous = self._previous, gray

        model = np.asarray(cv2.convertScaleAbs(self._model), dtype=np.uint8)
        mask = _threshold(gray, model, threshold)
        mask = np.asarray(cv2.dilate(mask, KERNEL, iterations=2), dtype=np.uint8)

        if np.count_nonzero(mask) > self.flood * mask.size:  # lights, exposure: re-learn fast
            cv2.accumulateWeighted(gray, self._model, 0.5)
            return np.zeros_like(gray)

        near_change = cv2.dilate(_threshold(gray, previous, threshold), NEAR_CHANGE) > 0
        foreground = mask > 0
        self._still = np.where(foreground & ~near_change, self._still + 1, 0).astype(np.uint16)
        stale = foreground & (self._still > self.still_frames)
        live = foreground & ~stale
        for rate, where in ((self.alpha, ~foreground), (self.absorb, live), (self.heal, stale)):
            cv2.accumulateWeighted(gray, self._model, rate, mask=where.astype(np.uint8))
        return mask


def find_blobs(mask: Image, min_area: int) -> list[Box]:
    """Bounding boxes of connected foreground regions at least min_area big."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes
