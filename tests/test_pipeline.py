from motion.common import Image
from motion.pipeline import FrameResult, LabelPolicy, Pipeline, Settings
from tests.helpers import RED, scene


class FakeLabeller:
    def __init__(self, answer: str = "cat") -> None:
        self.answer = answer
        self.calls = 0

    def label(self, crop_bgr: Image) -> str | None:
        self.calls += 1
        return self.answer


def run(pipeline: Pipeline, frames: int = 30, moving: int = 12) -> list[FrameResult]:
    results = []
    for i in range(frames):
        frame = scene() if i < frames - moving else scene((20 + 8 * i, 90, 50, RED))
        results.append(pipeline.process(frame, now=float(i)))
    return results


def test_moving_red_square_is_one_red_track_with_a_start_event() -> None:
    results = run(Pipeline())
    last = results[-1]
    assert [t.colour for t in last.tracks] == ["red"]
    assert len({t.id for r in results for t in r.tracks}) == 1
    assert [e.kind for r in results for e in r.events] == ["start"]
    assert (last.width, last.height) == (320, 240)


def test_labeller_is_only_used_when_identify_is_on() -> None:
    fake = FakeLabeller()
    pipeline = Pipeline(fake)
    run(pipeline)
    assert fake.calls == 0
    pipeline.settings = pipeline.settings.merged(identify=True)
    last = run(pipeline, frames=12, moving=12)[-1]
    assert fake.calls >= 1
    assert last.tracks[0].label == "cat"


def test_label_is_cached_between_refreshes() -> None:
    fake = FakeLabeller()
    pipeline = Pipeline(
        fake, LabelPolicy(first_after=2, refresh_after=100), Settings(identify=True)
    )
    run(pipeline, frames=30, moving=20)
    assert fake.calls == 1


def test_no_labeller_means_no_labels() -> None:
    pipeline = Pipeline(None, settings=Settings(identify=True))
    assert all(t.label is None for r in run(pipeline) for t in r.tracks)


def test_settings_are_clamped() -> None:
    s = Settings().merged(threshold=999, min_area=1)
    assert (s.threshold, s.min_area) == (100, 100)
    assert Settings(threshold=-5).threshold == 5  # clamped on creation too
    assert Settings().merged(identify=True).threshold == Settings().threshold
