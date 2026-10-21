from collections.abc import Sequence

import numpy as np

from motion.tracker import Track, Tracker

FPS = 10


def walk(tracker: Tracker, xs: Sequence[float], y: int = 100, size: int = 40) -> list[list[Track]]:
    """Feed one box per frame at the given x positions; return the visible tracks per frame."""
    return [tracker.update([(round(x), y, size, size)], now=i / FPS) for i, x in enumerate(xs)]


def test_moving_box_keeps_its_id_and_grows_a_capped_trail() -> None:
    tracker = Tracker(trail_len=5)
    frames = walk(tracker, [10 + 5 * i for i in range(12)])
    assert {t.id for f in frames for t in f} == {1}
    assert len(frames[-1][0].trail) == 5


def test_far_jump_is_a_new_object() -> None:
    tracker = Tracker(max_dist=50)
    tracker.update([(10, 10, 20, 20)], now=0.0)
    (track,) = tracker.update([(250, 200, 20, 20)], now=0.1)
    assert track.id == 2


def test_two_objects_keep_their_ids_when_close() -> None:
    tracker = Tracker()
    a, b = tracker.update([(100, 100, 20, 20), (140, 100, 20, 20)], now=0.0)
    moved = tracker.update([(145, 100, 20, 20), (105, 100, 20, 20)], now=0.1)
    by_id = {t.id: t.box[0] for t in moved}
    assert by_id[a.id] < 120 < by_id[b.id]


def test_track_coasts_then_is_dropped_and_ids_are_not_reused() -> None:
    tracker = Tracker(max_missed=3)
    first = tracker.update([(10, 10, 20, 20)], now=0.0)[0].id
    for i in range(3):
        assert tracker.update([], now=0.1 * (i + 1)) == []
        assert tracker.alive
    assert tracker.update([(10, 10, 20, 20)], now=0.4)[0].id == first
    for i in range(4):
        tracker.update([], now=0.5 + 0.1 * i)
    assert not tracker.alive
    assert tracker.update([(10, 10, 20, 20)], now=1.0)[0].id == first + 1


def jitter(xs: list[float]) -> float:
    """Mean frame-to-frame change in speed: what the eye sees as a twitching box."""
    return float(np.abs(np.diff(xs, n=2)).mean())


def test_filtered_box_twitches_far_less_than_the_raw_one() -> None:
    rng = np.random.default_rng(0)
    noisy = [20 + 4 * i + rng.normal(0, 4) for i in range(50)]  # 40 px/s plus noise
    frames = walk(Tracker(), noisy)
    filtered = [float(f[0].box[0]) for f in frames]
    assert jitter(filtered[10:]) < 0.5 * jitter(noisy[10:])


def test_filter_learns_the_velocity() -> None:
    frames = walk(Tracker(), [20 + 6 * i for i in range(30)])  # 60 px/s
    vx, vy, _, _ = frames[-1][0].velocity
    assert abs(vx - 60) < 6
    assert abs(vy) < 6


def test_accelerating_mover_keeps_its_id_thanks_to_the_prediction() -> None:
    # Speeds up to 35 px a frame (a quick arm swing): beyond max_dist from where
    # it last was, but close to where the filter predicts it.
    steps = [10, 15, 20, 25, 30, 35, 35, 35, 35]
    xs = [20 + sum(steps[:i]) for i in range(len(steps) + 1)]
    frames = walk(Tracker(max_dist=30), xs)
    assert {t.id for f in frames for t in f} == {1}
    assert max(steps) > 30  # the old last-position matching would have lost it


def test_boxes_are_clipped_to_the_frame() -> None:
    tracker = Tracker()
    for i in range(8):
        tracks = tracker.update([(280 + 5 * i, 10, 60, 60)], now=i / FPS, bounds=(320, 240))
    (track,) = tracks
    x, y, w, _h = track.box
    assert x >= 0
    assert y >= 0
    assert x + w <= 320
