from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from motion.common import Image
from server import create_app, jpeg_size, origin_allowed
from tests.helpers import RED, scene

ORIGIN = {"origin": "http://testserver"}


def jpeg(frame: Image) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def client(max_sessions: int = 4, session_seconds: float = 30) -> TestClient:
    app = create_app(
        allowed_origins=frozenset({"https://portfolio.example"}),
        max_sessions=max_sessions,
        session_seconds=session_seconds,
        model_path=Path("/nonexistent.onnx"),
    )
    return TestClient(app)


def test_round_trip_tracks_a_moving_square() -> None:
    with client().websocket_connect("/ws", headers=ORIGIN) as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert hello["labels"] is False
        ws.send_text('{"type":"settings","threshold":25,"minArea":400}')
        results = []
        for i in range(25):
            frame = scene() if i < 12 else scene((10 + 10 * i, 90, 50, RED))
            ws.send_bytes(jpeg(frame))
            results.append(ws.receive_json())
    assert {r["type"] for r in results} == {"result"}
    last = results[-1]
    assert (last["w"], last["h"]) == (320, 240)
    (track,) = last["tracks"]
    assert track["colour"] == "red"
    assert set(track) == {"id", "box", "velocity", "colour", "label", "trail"}
    assert [e["kind"] for r in results for e in r["events"]] == ["start"]


def test_bad_frame_is_reported_and_session_continues() -> None:
    with client().websocket_connect("/ws", headers=ORIGIN) as ws:
        ws.receive_json()
        ws.send_bytes(b"not a jpeg")
        assert ws.receive_json()["code"] == "bad_frame"
        ws.send_text('{"type":"settings","threshold":"high"}')
        assert ws.receive_json()["code"] == "bad_settings"
        ws.send_bytes(jpeg(scene()))
        assert ws.receive_json()["type"] == "result"


def test_foreign_origin_is_refused() -> None:
    with (
        pytest.raises(WebSocketDisconnect),
        client().websocket_connect("/ws", headers={"origin": "https://evil.example"}) as ws,
    ):
        ws.receive_json()


def test_second_visitor_gets_busy_when_full() -> None:
    c = client(max_sessions=1)
    with c.websocket_connect("/ws", headers=ORIGIN) as first:
        first.receive_json()
        with c.websocket_connect("/ws", headers=ORIGIN) as second:
            assert second.receive_json()["type"] == "busy"


def test_session_expires() -> None:
    with client(session_seconds=0.2).websocket_connect("/ws", headers=ORIGIN) as ws:
        ws.receive_json()
        assert ws.receive_json()["type"] == "expired"


def test_page_and_embedding_headers() -> None:
    res = client().get("/")
    assert res.status_code == 200
    csp = res.headers["content-security-policy"]
    assert csp.endswith("frame-ancestors 'self' https://portfolio.example")
    assert "connect-src 'self'" in csp
    assert "googleapis" not in res.text  # fonts are served locally
    assert res.headers["x-content-type-options"] == "nosniff"


def test_origin_rules() -> None:
    allowed = frozenset({"https://a.example"})
    assert origin_allowed("https://a.example", "demo.example", allowed)
    assert origin_allowed("https://demo.example", "demo.example", allowed)
    assert not origin_allowed("https://b.example", "demo.example", allowed)
    assert not origin_allowed(None, "demo.example", allowed)


def test_jpeg_size_reads_the_header_without_decoding() -> None:
    assert jpeg_size(jpeg(scene())) == (320, 240)
    assert jpeg_size(b"not a jpeg") is None
    assert jpeg_size(b"\xff\xd8\xff") is None  # truncated


def test_huge_declared_image_is_refused_before_decoding() -> None:
    small = jpeg(np.zeros((8, 8, 3), np.uint8))
    sof = small.index(b"\xff\xc0")
    bomb = bytearray(small)
    bomb[sof + 5 : sof + 9] = (30000).to_bytes(2) + (30000).to_bytes(2)  # 30000 x 30000
    with client().websocket_connect("/ws", headers=ORIGIN) as ws:
        ws.receive_json()
        ws.send_bytes(bytes(bomb))
        assert ws.receive_json()["code"] == "bad_frame"


def test_oversized_or_nested_settings_are_refused() -> None:
    with client().websocket_connect("/ws", headers=ORIGIN) as ws:
        ws.receive_json()
        ws.send_text('{"type":"settings","threshold":' + "1" * 2000 + "}")
        assert ws.receive_json()["code"] == "bad_settings"
        ws.send_text("[" * 900 + "]" * 900)
        assert ws.receive_json()["code"] == "bad_settings"
        ws.send_bytes(jpeg(scene()))
        assert ws.receive_json()["type"] == "result"


def test_quiet_session_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("server.IDLE_SECONDS", 0.2)
    with client().websocket_connect("/ws", headers=ORIGIN) as ws:
        ws.receive_json()
        assert ws.receive_json()["type"] == "expired"


def test_one_visitor_cannot_take_every_slot() -> None:
    c = client(max_sessions=4)
    with (
        c.websocket_connect("/ws", headers=ORIGIN) as a,
        c.websocket_connect("/ws", headers=ORIGIN) as b,
    ):
        a.receive_json()
        b.receive_json()
        with c.websocket_connect("/ws", headers=ORIGIN) as third:
            assert third.receive_json()["type"] == "busy"
