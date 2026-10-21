"""One camera stream, frame by frame: background -> blobs -> tracks -> colour, labels, events."""

from dataclasses import dataclass

from motion.background import Background, find_blobs, to_gray
from motion.colour import dominant_colour
from motion.common import Box, Image
from motion.events import MotionEvent, MotionEvents
from motion.objects import Labeller
from motion.tracker import Track, Tracker

THRESHOLD_RANGE = (5, 100)
MIN_AREA_RANGE = (100, 5000)


@dataclass(frozen=True, slots=True)
class Settings:
    """What the visitor controls. Numbers are clamped to their ranges on creation."""

    threshold: int = 25
    min_area: int = 600  # pixels of the ~320 px wide frame
    identify: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", _clamp(self.threshold, THRESHOLD_RANGE))
        object.__setattr__(self, "min_area", _clamp(self.min_area, MIN_AREA_RANGE))

    def merged(
        self,
        *,
        threshold: int | None = None,
        min_area: int | None = None,
        identify: bool | None = None,
    ) -> "Settings":
        """A copy with the given fields changed; None means "leave as is"."""
        return Settings(
            self.threshold if threshold is None else threshold,
            self.min_area if min_area is None else min_area,
            self.identify if identify is None else identify,
        )


@dataclass(frozen=True, slots=True)
class LabelPolicy:
    first_after: int = 5  # frames a new track must survive before the first look
    refresh_after: int = 30  # frames before a labelled track is looked at again
    per_frame: int = 1  # at most this many network runs per frame
    pad: float = 0.15
    min_side: int = 24


@dataclass(frozen=True, slots=True)
class FrameResult:
    width: int
    height: int
    tracks: list[Track]
    events: list[MotionEvent]


def _clamp(value: int, bounds: tuple[int, int]) -> int:
    return max(bounds[0], min(bounds[1], value))


def padded_crop(frame: Image, box: Box, pad: float) -> tuple[Image, Box]:
    x, y, w, h = box
    dx, dy = int(w * pad), int(h * pad)
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(frame.shape[1], x + w + dx), min(frame.shape[0], y + h + dy)
    return frame[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


class Pipeline:
    """Per-connection state. Not thread-safe; feed it one frame at a time."""

    COLOUR_EVERY = 5

    def __init__(
        self,
        labeller: Labeller | None = None,
        policy: LabelPolicy | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.labeller = labeller
        self.policy = policy or LabelPolicy()
        self.settings = settings or Settings()
        self._background = Background()
        self._tracker = Tracker()
        self._events = MotionEvents()
        self._shape: tuple[int, ...] | None = None

    def process(self, frame: Image, now: float) -> FrameResult:
        if frame.shape != self._shape:  # camera or size changed: start over
            self._shape = frame.shape
            self._background.reset()
            self._tracker.reset()

        mask = self._background.apply(to_gray(frame), self.settings.threshold)
        bounds = (frame.shape[1], frame.shape[0])
        tracks = self._tracker.update(find_blobs(mask, self.settings.min_area), now, bounds)

        for track in tracks:
            track.since_label_try += 1
            if track.colour is None or track.age % self.COLOUR_EVERY == 0:
                x, y, w, h = track.box
                track.colour = dominant_colour(
                    frame[y : y + h, x : x + w], mask[y : y + h, x : x + w]
                )

        if self.settings.identify and self.labeller is not None:
            self._label(frame, tracks)

        event = self._events.update(self._tracker.alive, now)
        return FrameResult(frame.shape[1], frame.shape[0], tracks, [event] if event else [])

    def _label(self, frame: Image, tracks: list[Track]) -> None:
        assert self.labeller is not None
        p = self.policy

        def is_due(t: Track) -> bool:
            wait = p.first_after if t.label is None else p.refresh_after
            return t.since_label_try >= wait

        due = sorted(filter(is_due, tracks), key=lambda t: t.box[2] * t.box[3], reverse=True)
        for track in due[: p.per_frame]:
            track.since_label_try = 0
            crop, (_, _, w, h) = padded_crop(frame, track.box, p.pad)
            if min(w, h) < p.min_side:
                continue
            track.label = self.labeller.label(crop) or track.label
