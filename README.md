# Motion Capture

[![CI](https://github.com/adambisa/motion-capture/actions/workflows/ci.yml/badge.svg)](https://github.com/adambisa/motion-capture/actions/workflows/ci.yml)

A webcam motion tracker. OpenCV in Python finds what moves, follows each thing with
its own ID, and a small neural network names it. Your browser only provides the
camera and draws the boxes.

**Try it:** the live demo runs on the Motion Detection page of my portfolio, or locally in a minute (see [Run it](#run-it)).

## How it grew

**v1 (2024), [`capture.py`](capture.py).** One script, about 60 lines. It diffs every
frame against the very first one, boxes anything big that changed, and logs when
motion starts and ends to `times.csv`. It works, but the first frame is the background
forever: move a chair or let a cloud pass, and it reports motion until you restart.

**v2 (2026), [`motion/`](motion) and [`server.py`](server.py).** The same idea, rebuilt
to run live on a web page:

| | v1 | v2 |
|---|---|---|
| Background | First frame, forever | Running average that keeps learning, blob by blob |
| Objects | Boxes, no identity | Kalman-filtered tracks with IDs, velocity and trails |
| What is it | Nothing | YOLOX-nano names moving things (opt-in) |
| Where it runs | Your machine, OpenCV windows | Python server; any browser with a camera |
| Events | Start and end times in a CSV | Debounced start and end, live in the page |

## How it works

```
browser                          server (FastAPI, one pipeline per visitor)
camera ─► 320 px JPEG ─ WebSocket ─► grayscale + blur
                                     background model ─► foreground mask
                                     contours ─► blobs ─► tracker (IDs, trails)
                                     colour per blob, YOLOX on moving crops
canvas ◄─ boxes, IDs, labels, events ◄─ JSON
```

- **Background** ([`background.py`](motion/background.py)). It starts from the
  per-pixel median of the first second, so anything moving then drops out. After that a
  running average keeps learning, at three speeds. Quiet pixels learn normally.
  Foreground learns very slowly, so the background under a moving person stays clean;
  a plain running average soaks them up and leaves a "ghost" when they walk away.
  Foreground with no change nearby for most of a second (a ghost, or something that
  parked) learns fast and fades in. A whole-frame change, like lights or auto-exposure,
  is absorbed at once instead of reported.
- **Tracker** ([`tracker.py`](motion/tracker.py)). Each track has a Kalman filter
  (OpenCV's) over its box centre, size and their velocities. Every frame it predicts
  where the box should be now; new blobs are matched against that prediction, nearest
  first, and correct it. The box stops twitching, fast movers keep their IDs, and a
  track that loses its blob coasts on the prediction for a few frames.
- **Drawing ahead** ([`app.js`](web/app.js)). Boxes describe a frame taken one round trip
  ago. The page redraws at 60 fps, moving each box along its velocity to where the
  object should be now, and eases into each new reply instead of jumping. The status bar
  shows the round-trip time.
- **Labels** ([`objects.py`](motion/objects.py)). YOLOX-nano (Apache-2.0) runs through
  OpenCV's DNN module, with no PyTorch. It only sees crops around tracks that are already
  moving, at most one per frame, and the label is cached on the track. The cheap classic
  pipeline decides where to look, and the network only says what it is.
- **Colour** ([`colour.py`](motion/colour.py)). The dominant colour of a blob's moving
  pixels, snapped to twelve names. The page shows it as a swatch for things, and never
  for people.
- **Events** ([`events.py`](motion/events.py)). A one-frame flicker is not an event:
  start needs a couple of moving frames in a row, end needs about a second of quiet.

**Privacy.** Frames are processed in memory and never written to disk or logged. The
page loads nothing from other sites; its font is served from here.

**Abuse limits.** The server reads each JPEG's size from its header and refuses anything
over 640 x 480 before decoding. Settings messages are capped. Each visitor gets a few
minutes, at most two sessions per address, and a session that sends nothing for 10
seconds is closed.

**Limits.** Labels come from small crops of a 320 px frame, so small or distant things
are often unnamed. Large flat-coloured objects moving slowly can be partly absorbed into
the background, because their middle shows no change from frame to frame. Prediction
assumes steady speed, so a box overshoots slightly when someone stops dead.

## Run it

Needs [uv](https://docs.astral.sh/uv/). Python 3.13 is installed for you.

```bash
uv sync
curl -L --create-dirs -o models/yolox_nano.onnx \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
uv run uvicorn server:app --port 8000 --ws-max-size 524288
```

Open http://localhost:8000 and press **Start camera**. Without the model file
everything works except **Identify objects**.

**Docker.** The image downloads the model at build time, checked against a pinned SHA-256.

```bash
docker build -t motion-capture .
docker run -p 8000:8000 -e ALLOWED_ORIGINS=https://your-site.example motion-capture
```

**v1** needs only OpenCV with its GUI build:

```bash
uv run --no-project --with opencv-python capture.py
```

### Settings

| Variable | Default | Meaning |
|---|---|---|
| `ALLOWED_ORIGINS` | none | Comma-separated sites allowed to embed the demo in an iframe |
| `MAX_SESSIONS` | 4 | Visitors at once; the next one is told to try again shortly |
| `SESSION_SECONDS` | 180 | Length of one session |
| `MOTION_MODEL` | `models/yolox_nano.onnx` | Path to the YOLOX-nano model |
| `PORT` | 8000 | Docker only |
| `FORWARDED_ALLOW_IPS` | 127.0.0.1 | Your proxy's address, so per-visitor limits see real IPs |

Run a single worker: the session limits live in the process. Browsers only allow the
camera on HTTPS pages (or localhost), so put it behind TLS.

### Embedding

```html
<iframe src="https://demo.example/" allow="camera" title="Motion capture demo"></iframe>
```

The embedding site's origin must be in `ALLOWED_ORIGINS`.

## Development

```bash
uv run pytest            # tests on synthetic frames, plus a WebSocket round trip
uv run ruff check . && uv run ruff format --check .
uv run mypy              # strict
npm ci && npm run lint && npm run typecheck   # ESLint and tsc for web/app.js (plain JS, JSDoc types)
```

CI runs all of this on every pull request and push to `main`, builds the Docker image
and checks that the model loads. Pushes to `main` then publish the image to
`ghcr.io/adambisa/motion-capture`. Actions are pinned to commits; Dependabot proposes
weekly updates.

## Licence

MIT. YOLOX-nano is by Megvii, under Apache-2.0; see [NOTICE](NOTICE).
