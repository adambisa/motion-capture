from motion.events import MotionEvent, MotionEvents


def feed(events: MotionEvents, pattern: str, t0: float = 0.0) -> list[MotionEvent]:
    out = []
    for i, c in enumerate(pattern):
        ev = events.update(c == "x", t0 + i)
        if ev:
            out.append(ev)
    return out


def test_single_frame_flicker_is_not_an_event() -> None:
    assert feed(MotionEvents(start_frames=2), "..x..x..") == []


def test_start_then_end_with_duration() -> None:
    got = feed(MotionEvents(start_frames=2, end_frames=3), "xxxxx...")
    assert got == [MotionEvent("start", 1.0), MotionEvent("end", 7.0, 6.0)]


def test_short_pause_does_not_end_motion() -> None:
    got = feed(MotionEvents(start_frames=1, end_frames=3), "xx..xx")
    assert [e.kind for e in got] == ["start"]
