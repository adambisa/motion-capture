"""Serves the demo page and runs one motion pipeline per WebSocket connection.

The browser sends small JPEG frames; Python does all the vision work and sends
back boxes, IDs, colours, labels and events as JSON. Frames live only in memory.

It is a public demo on a small box, so it defends itself: frame size is read from
the JPEG header before anything is decoded, settings messages are capped, visitors
get a few minutes each (at most two sessions per address), and a session that goes
quiet is closed. Run uvicorn with --ws-max-size 524288 so oversized messages are
refused before they are buffered.
"""

import asyncio
import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import RequestResponseEndpoint

from motion.objects import YoloxLabeller
from motion.pipeline import MIN_AREA_RANGE, THRESHOLD_RANGE, FrameResult, Pipeline

log = logging.getLogger("motion.server")

ROOT = Path(__file__).parent
MAX_FRAME_BYTES = 256 * 1024
MAX_FRAME_SIZE = (640, 480)  # the client sends 320 px wide; leave room, refuse the rest
MAX_SETTINGS_CHARS = 1024
IDLE_SECONDS = 10.0
PER_CLIENT = 2
# script/style/font/img from this server only; the page talks only to its own /ws.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; "
    "img-src 'self' data: blob:; connect-src 'self'; frame-ancestors {ancestors}"
)

Message = dict[str, Any]


def to_message(result: FrameResult) -> Message:
    return {
        "type": "result",
        "w": result.width,
        "h": result.height,
        "tracks": [
            {
                "id": t.id,
                "box": list(t.box),
                "velocity": list(t.velocity),
                "colour": t.colour,
                "label": t.label,
                "trail": [list(p) for p in t.trail],
            }
            for t in result.tracks
        ],
        "events": [{"kind": e.kind, "at": e.at, "duration": e.duration} for e in result.events],
    }


def error(code: str, message: str) -> Message:
    return {"type": "error", "code": code, "message": message}


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a JPEG's frame header, without decoding it.

    A tiny JPEG can declare an enormous image, and decoding it first would
    allocate all of it, so the size is checked before cv2.imdecode ever runs.
    """
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        length = int.from_bytes(data[i + 2 : i + 4])
        # SOF0..SOF15 carry the size; C4 (Huffman), C8 and CC (arithmetic) do not.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 9 > len(data):
                return None
            height = int.from_bytes(data[i + 5 : i + 7])
            width = int.from_bytes(data[i + 7 : i + 9])
            return width, height
        i += 2 + length
    return None


def handle_frame(pipeline: Pipeline, data: bytes) -> Message:
    """Decode one JPEG and run it through the pipeline. Runs in a worker thread."""
    if len(data) > MAX_FRAME_BYTES:
        return error("too_large", "Frame too large.")
    size = jpeg_size(data)
    if size is None or size[0] > MAX_FRAME_SIZE[0] or size[1] > MAX_FRAME_SIZE[1]:
        return error("bad_frame", "Frames must be JPEGs of at most 640 x 480.")
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return error("bad_frame", "Could not read that frame.")
    try:
        return to_message(pipeline.process(np.asarray(frame, dtype=np.uint8), time.time()))
    except Exception:
        log.exception("Pipeline failed on a frame")
        return error("internal", "Something went wrong on that frame.")


def apply_settings(pipeline: Pipeline, text: str) -> Message | None:
    """Merge a settings message into the pipeline. Returns an error, or None when fine."""
    if len(text) > MAX_SETTINGS_CHARS:
        return error("bad_settings", "Settings message too long.")
    try:
        msg = json.loads(text)
    except (ValueError, RecursionError):  # bad JSON, absurd numbers, deep nesting
        return error("bad_settings", "Settings must be JSON.")
    if not isinstance(msg, dict) or msg.get("type") != "settings":
        return error("bad_settings", "Unknown message.")
    threshold, min_area, identify = msg.get("threshold"), msg.get("minArea"), msg.get("identify")
    numbers_ok = all(
        v is None or (isinstance(v, int) and not isinstance(v, bool)) for v in (threshold, min_area)
    )
    if not numbers_ok or not (identify is None or isinstance(identify, bool)):
        return error("bad_settings", "Settings have the wrong types.")
    pipeline.settings = pipeline.settings.merged(
        threshold=threshold, min_area=min_area, identify=identify
    )
    return None


def origin_allowed(origin: str | None, host: str | None, allowed: frozenset[str]) -> bool:
    """Same-origin (the demo page itself) or an explicitly allowed origin."""
    if not origin:
        return False
    return origin in allowed or urlsplit(origin).netloc == host


class Sessions:
    """Counts live demo sessions, overall and per client address.

    Only touched from the event loop, so no lock. Behind a proxy, set uvicorn's
    FORWARDED_ALLOW_IPS to the proxy's address so the client address is the visitor's.
    """

    def __init__(self, limit: int, per_client: int = PER_CLIENT) -> None:
        self.limit = limit
        self.per_client = per_client
        self.active = 0
        self._by_client: dict[str, int] = {}

    def enter(self, client: str) -> bool:
        if self.active >= self.limit or self._by_client.get(client, 0) >= self.per_client:
            return False
        self.active += 1
        self._by_client[client] = self._by_client.get(client, 0) + 1
        return True

    def leave(self, client: str) -> None:
        self.active -= 1
        self._by_client[client] -= 1
        if not self._by_client[client]:
            del self._by_client[client]


async def send_and_close(socket: WebSocket, message: Message, code: int) -> None:
    """Best effort: the visitor may already be gone."""
    with contextlib.suppress(WebSocketDisconnect, RuntimeError):
        await socket.send_json(message)
        await socket.close(code=code)


def _env_origins() -> frozenset[str]:
    return frozenset(
        o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()
    )


def create_app(
    *,
    allowed_origins: frozenset[str] | None = None,
    max_sessions: int | None = None,
    session_seconds: float | None = None,
    model_path: Path | None = None,
    static_dir: Path = ROOT / "web",
) -> FastAPI:
    """Build the app. Unset arguments come from the environment:

    ALLOWED_ORIGINS  comma-separated sites that may embed the demo (the portfolio)
    MAX_SESSIONS     concurrent demo sessions (4)
    SESSION_SECONDS  length of one session (180)
    MOTION_MODEL     path to yolox_nano.onnx (models/yolox_nano.onnx)
    """
    env = os.environ
    origins = _env_origins() if allowed_origins is None else allowed_origins
    max_sessions = max_sessions or int(env.get("MAX_SESSIONS", "4"))
    session_seconds = session_seconds or float(env.get("SESSION_SECONDS", "180"))
    model_path = model_path or Path(env.get("MOTION_MODEL", ROOT / "models" / "yolox_nano.onnx"))
    sessions = Sessions(max_sessions)
    labels_available = YoloxLabeller.load(model_path) is not None
    cv2.setNumThreads(1)  # a few sessions on a small box: don't oversubscribe the CPU

    app = FastAPI(title="motion-capture", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def headers(request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        ancestors = " ".join(["'self'", *sorted(origins)])
        response.headers["Content-Security-Policy"] = CSP.format(ancestors=ancestors)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True, "labels": labels_available}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        if not origin_allowed(socket.headers.get("origin"), socket.headers.get("host"), origins):
            await socket.close(code=1008)
            return
        await socket.accept()
        client = socket.client.host if socket.client else "unknown"
        if not sessions.enter(client):
            busy = "All demo slots are taken. Try again in a minute."
            await send_and_close(socket, {"type": "busy", "message": busy}, 1013)
            return
        try:
            await run_session(socket)
        finally:
            sessions.leave(client)

    async def run_session(socket: WebSocket) -> None:
        # Each session loads its own network: cv2.dnn.Net is not safe to share across
        # threads, and at ~3.6 MB it loads in tens of milliseconds.
        labeller = (
            await asyncio.to_thread(YoloxLabeller.load, model_path) if labels_available else None
        )
        pipeline = Pipeline(labeller)
        s = pipeline.settings
        await socket.send_json(
            {
                "type": "hello",
                "labels": labeller is not None,
                "maxSeconds": session_seconds,
                "settings": {
                    "threshold": s.threshold,
                    "minArea": s.min_area,
                    "identify": s.identify,
                },
                "limits": {"threshold": THRESHOLD_RANGE, "minArea": MIN_AREA_RANGE},
            }
        )
        try:
            async with asyncio.timeout(session_seconds):
                while True:
                    try:
                        msg = await asyncio.wait_for(socket.receive(), IDLE_SECONDS)
                    except TimeoutError:
                        idle = "Paused because no frames arrived. Press Start to go again."
                        await send_and_close(socket, {"type": "expired", "message": idle}, 1000)
                        return
                    if msg["type"] == "websocket.disconnect":
                        return
                    reply: Message | None
                    if msg.get("bytes") is not None:
                        reply = await asyncio.to_thread(handle_frame, pipeline, msg["bytes"])
                    else:
                        reply = apply_settings(pipeline, msg.get("text") or "")
                    if reply is not None:
                        await socket.send_json(reply)
        except TimeoutError:
            minutes = round(session_seconds / 60)
            ended = f"The {minutes}-minute demo ended. Press Start to run it again."
            await send_and_close(socket, {"type": "expired", "message": ended}, 1000)
        except WebSocketDisconnect:
            return

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")
    return app


app = create_app()
