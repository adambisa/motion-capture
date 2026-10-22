"""Tracker: gives each moving blob an ID that survives across frames, and smooths
its box with a Kalman filter.

Raw blob boxes are noisy: the outline of "what changed" grows, shrinks and
shifts from frame to frame. Each track keeps a Kalman filter (OpenCV's) over
its box centre, size and their velocities. Every frame the filter predicts where
the box should be now, the tracker matches new blobs against that prediction,
and the matched blob corrects it. The result is a steady box, a velocity the
browser can use to draw ahead between replies, and IDs that survive fast motion
because matching is done against where things are going, not where they were.
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from motion.common import Box, Point

STATE = 8  # cx, cy, w, h, then each one's rate of change in px/s
MEASURED = 4  # cx, cy, w, h of the blob

# How much a blob box wobbles around the truth, in px² (std ~4 px for position, ~5 px for size).
MEASUREMENT_NOISE = np.diag([16, 16, 25, 25]).astype(np.float32)
# How freely things may speed up or slow down, per second: px² for position, (px/s)² for
# velocity. Tuned on synthetic walks: lower is smoother but overshoots more when someone
# stops dead; this overshoots ~4 px and settles in ~0.6 s at 10 fps.
PROCESS_NOISE = np.diag([1, 1, 1, 1, 1000, 1000, 250, 250]).astype(np.float32)

Vector = NDArray[np.float32]


def centroid(box: Box) -> Point:
    x, y, w, h = box
    return (x + w // 2, y + h // 2)


def _measurement(box: Box) -> Vector:
    x, y, w, h = box
    return np.array([[x + w / 2], [y + h / 2], [w], [h]], dtype=np.float32)


def _new_filter(box: Box, measurement_noise: NDArray[np.float32]) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(STATE, MEASURED)
    kf.measurementMatrix = np.eye(MEASURED, STATE, dtype=np.float32)
    kf.measurementNoiseCov = measurement_noise.copy()
    kf.statePost = np.vstack([_measurement(box), np.zeros((4, 1), np.float32)])
    # Sure about where it is, clueless about how fast it is going.
    kf.errorCovPost = np.diag([9, 9, 25, 25, 1e4, 1e4, 1e4, 1e4]).astype(np.float32)
    return kf


def _predict(kf: cv2.KalmanFilter, dt: float, process_noise: NDArray[np.float32]) -> None:
    """Constant velocity: position += velocity * dt."""
    transition = np.eye(STATE, dtype=np.float32)
    transition[:4, 4:] = np.eye(4, dtype=np.float32) * dt
    kf.transitionMatrix = transition
    kf.processNoiseCov = process_noise * dt
    kf.predict()  # OpenCV also copies the prediction into statePost


@dataclass(slots=True)
class Track:
    id: int
    box: Box
    trail: deque[Point]
    kf: cv2.KalmanFilter = field(repr=False)
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # px/s: cx, cy, w, h
    missed: int = 0
    age: int = 1
    colour: str | None = None
    label: str | None = None
    since_label_try: int = 0  # frames since the labeller last looked at this track

    def centre(self) -> tuple[float, float]:
        state = self.kf.statePost
        return float(state[0, 0]), float(state[1, 0])

    def sync(self, bounds: tuple[int, int] | None) -> None:
        """Copy the filter's state into box and velocity, clipped to the frame."""
        cx, cy, w, h, vx, vy, vw, vh = (float(v) for v in self.kf.statePost[:, 0])
        w, h = max(w, 2.0), max(h, 2.0)
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        if bounds is not None:
            width, height = bounds
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(width), x1), min(float(height), y1)
        self.box = (round(x0), round(y0), max(1, round(x1 - x0)), max(1, round(y1 - y0)))
        self.velocity = (round(vx, 1), round(vy, 1), round(vw, 1), round(vh, 1))


class Tracker:
    """Greedy matching against each track's predicted position.

    Every (track, blob) pair is ranked by distance and accepted nearest-first,
    so two objects passing close by keep their own IDs. A track that loses
    its blob coasts on its prediction for up to max_missed frames, then is dropped.

    The noise matrices default to values tuned for motion blobs; the hand
    pipeline passes ones measured from its detector.
    """

    def __init__(
        self,
        max_missed: int = 8,
        max_dist: float = 60.0,
        trail_len: int = 20,
        measurement_noise: NDArray[np.float32] = MEASUREMENT_NOISE,
        process_noise: NDArray[np.float32] = PROCESS_NOISE,
    ) -> None:
        self.max_missed = max_missed
        self.max_dist = max_dist
        self.trail_len = trail_len
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._last: float | None = None

    @property
    def alive(self) -> bool:
        """Any track still held, including ones coasting through a missed frame."""
        return bool(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._last = None

    def update(
        self,
        boxes: Sequence[Box],
        now: float,
        bounds: tuple[int, int] | None = None,
        can_start: Sequence[bool] | None = None,
    ) -> list[Track]:
        """Match this frame's boxes to tracks and return the visible tracks.

        `now` is in seconds; `bounds` is the frame's (width, height) to clip boxes to.
        `can_start[i]` False means box i may continue a track but not start one
        (a low-confidence detection: enough to keep a hand, not to invent one).
        """
        # Frames arrive at an uneven rate, so the filter steps by real elapsed time.
        dt = 0.1 if self._last is None else min(max(now - self._last, 1 / 60), 0.5)
        self._last = now
        for track in self._tracks.values():
            _predict(track.kf, dt, self.process_noise)

        pairs: list[tuple[float, int, int]] = []
        for track in self._tracks.values():
            tx, ty = track.centre()
            for i, box in enumerate(boxes):
                bx, by = box[0] + box[2] / 2, box[1] + box[3] / 2
                dist = ((tx - bx) ** 2 + (ty - by) ** 2) ** 0.5
                if dist <= self.max_dist:
                    pairs.append((dist, track.id, i))
        pairs.sort()

        matched_tracks: set[int] = set()
        matched_boxes: set[int] = set()
        for _, track_id, i in pairs:
            if track_id in matched_tracks or i in matched_boxes:
                continue
            matched_tracks.add(track_id)
            matched_boxes.add(i)
            track = self._tracks[track_id]
            track.kf.correct(_measurement(boxes[i]))
            track.sync(bounds)
            track.trail.append(centroid(track.box))
            track.missed = 0
            track.age += 1

        for track_id in list(self._tracks):
            if track_id in matched_tracks:
                continue
            track = self._tracks[track_id]
            track.missed += 1
            if track.missed > self.max_missed:
                del self._tracks[track_id]
            else:
                track.sync(bounds)  # coast on the prediction

        for i, box in enumerate(boxes):
            if i in matched_boxes or (can_start is not None and not can_start[i]):
                continue
            trail = deque([centroid(box)], maxlen=self.trail_len)
            track = Track(self._next_id, box, trail, _new_filter(box, self.measurement_noise))
            track.sync(bounds)
            self._tracks[track.id] = track
            self._next_id += 1

        return [t for t in self._tracks.values() if t.missed == 0]
