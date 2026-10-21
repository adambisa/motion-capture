"""Turn per-frame "is anything moving" into debounced start/end events."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class MotionEvent:
    kind: Literal["start", "end"]
    at: float
    duration: float | None = None


class MotionEvents:
    """A one-frame flicker is not an event: start needs start_frames moving
    frames in a row, end needs end_frames quiet frames in a row."""

    def __init__(self, start_frames: int = 2, end_frames: int = 10) -> None:
        self.start_frames = start_frames
        self.end_frames = end_frames
        self._moving = False
        self._streak = 0
        self._started_at = 0.0

    def update(self, moving: bool, now: float) -> MotionEvent | None:
        if moving == self._moving:
            self._streak = 0
            return None
        self._streak += 1
        needed = self.start_frames if moving else self.end_frames
        if self._streak < needed:
            return None
        self._moving = moving
        self._streak = 0
        if moving:
            self._started_at = now
            return MotionEvent("start", now)
        return MotionEvent("end", now, round(now - self._started_at, 2))
