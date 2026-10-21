// @ts-check
// The browser half of the demo: camera in, small JPEGs out, boxes back.
// All the vision work happens in Python on the server; this file only draws.

/**
 * @typedef {[number, number, number, number]} Box  x, y, width, height
 * @typedef {{ id: number, box: Box, velocity: Box, colour: string | null,
 *             label: string | null, trail: [number, number][] }} Track
 *   velocity: px/s of the box centre x, y and of width, height (from the Kalman filter)
 * @typedef {{ kind: 'start' | 'end', at: number, duration: number | null }} MotionEvent
 * @typedef {{ threshold: number, minArea: number, identify: boolean }} Settings
 * @typedef {{ type: 'hello', labels: boolean, maxSeconds: number, settings: Settings,
 *             limits: { threshold: [number, number], minArea: [number, number] } }} Hello
 * @typedef {{ type: 'result', w: number, h: number, tracks: Track[], events: MotionEvent[] }} Result
 * @typedef {{ type: 'busy' | 'expired', message: string }} Notice
 * @typedef {{ type: 'error', code: string, message: string }} ErrorMessage
 * @typedef {Hello | Result | Notice | ErrorMessage} ServerMessage
 * @typedef {{ track: Track, capturedAt: number, display: Box | null }} Shown
 */

const SEND_SIDE = 320; // long side of each frame sent, portrait or landscape
const JPEG_QUALITY = 0.7;
const MIN_FRAME_GAP_MS = 100; // at most ~10 frames a second
const MAX_EVENTS = 30;
// Boxes describe a frame taken a round trip ago. Between replies, each box moves
// along its velocity to where the object should be now, for at most this long...
const MAX_LOOKAHEAD_S = 0.35;
// ...and the drawn box eases toward that target instead of jumping (per second).
const EASE_RATE = 18;

/** Palette names from the server, drawn as small swatches. */
const SWATCH = /** @type {Record<string, string>} */ ({
  black: '#111', white: '#f4f4f4', grey: '#8a8a8a', red: '#e0322b', orange: '#ff8a1c',
  yellow: '#f2d22e', green: '#3cc45a', cyan: '#2ed3d8', blue: '#2f6bff', purple: '#9b4dff',
  pink: '#ff6fb5', brown: '#8b5a2b',
});

/**
 * @template {HTMLElement} T
 * @param {string} id
 * @param {new () => T} type
 * @returns {T}
 */
function el(id, type) {
  const node = document.getElementById(id);
  if (!(node instanceof type)) throw new Error(`#${id} missing`);
  return node;
}

const ui = {
  video: el('video', HTMLVideoElement),
  overlay: el('overlay', HTMLCanvasElement),
  stage: el('stage', HTMLDivElement),
  cover: el('cover', HTMLDivElement),
  message: el('message', HTMLParagraphElement),
  status: el('status', HTMLSpanElement),
  start: el('start', HTMLButtonElement),
  stop: el('stop', HTMLButtonElement),
  threshold: el('threshold', HTMLInputElement),
  thresholdOut: el('threshold-out', HTMLOutputElement),
  minArea: el('min-area', HTMLInputElement),
  minAreaOut: el('min-area-out', HTMLOutputElement),
  identify: el('identify', HTMLInputElement),
  events: el('events', HTMLOListElement),
};

const DEFAULT_NOTE = ui.message.textContent ?? '';
const frameCanvas = document.createElement('canvas');

/** @type {{ ws: WebSocket | null, stream: MediaStream | null, inFlight: boolean,
 *           lastSend: number, capturedAt: number, latency: number, results: number[],
 *           timer: number, raf: number, lastTick: number, frame: { w: number, h: number },
 *           shown: Map<number, Shown>, touched: boolean, errors: number }} */
const state = {
  ws: null, stream: null, inFlight: false, lastSend: 0, capturedAt: 0, latency: 0,
  results: [], timer: 0, raf: 0, lastTick: 0, frame: { w: 1, h: 1 }, shown: new Map(),
  touched: false,
  errors: 0,
};

// ---- session ---------------------------------------------------------------

async function start() {
  ui.start.disabled = true;
  ui.message.textContent = 'Asking for the camera…';
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, facingMode: 'user' },
      audio: false,
    });
  } catch {
    stop('No camera, or permission was denied. Allow camera access and try again.');
    return;
  }
  ui.video.srcObject = state.stream;
  await ui.video.play();
  ui.stage.style.aspectRatio = `${ui.video.videoWidth} / ${ui.video.videoHeight}`;

  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${scheme}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;
  /** @type {string | null} */
  let closingNote = null;

  ws.onmessage = (event) => {
    const msg = /** @type {ServerMessage} */ (JSON.parse(String(event.data)));
    if (msg.type === 'hello') onHello(msg);
    else if (msg.type === 'result') onResult(msg);
    else if (msg.type === 'error') onError(msg);
    else closingNote = msg.message;
  };
  ws.onclose = () => {
    if (state.ws === ws) stop(closingNote ?? 'Connection closed. Press Start to try again.');
  };
}

/** @param {string} [note] */
function stop(note) {
  const ws = state.ws;
  state.ws = null;
  ws?.close();
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  clearTimeout(state.timer);
  cancelAnimationFrame(state.raf);
  state.shown.clear();
  state.latency = 0;
  state.inFlight = false;
  ui.video.srcObject = null;
  clearOverlay();
  ui.cover.hidden = false;
  ui.start.disabled = false;
  ui.stop.hidden = true;
  ui.message.textContent = note ?? DEFAULT_NOTE;
  setStatus('Idle', false);
}

/** @param {Hello} hello */
function onHello(hello) {
  ui.threshold.min = String(hello.limits.threshold[0]);
  ui.threshold.max = String(hello.limits.threshold[1]);
  ui.minArea.min = String(hello.limits.minArea[0]);
  ui.minArea.max = String(hello.limits.minArea[1]);
  // The server's defaults win unless the visitor already moved a slider.
  if (!state.touched) {
    ui.threshold.value = String(hello.settings.threshold);
    ui.minArea.value = String(hello.settings.minArea);
  }
  showValues();
  ui.identify.disabled = !hello.labels;
  ui.identify.parentElement?.setAttribute(
    'title', hello.labels ? '' : 'The object model is not installed on this server.');
  ui.cover.hidden = true;
  ui.stop.hidden = false;
  setStatus('Live', true);
  sendSettings();
  sendFrame();
  state.lastTick = 0;
  state.raf = requestAnimationFrame(tick);
}

/** @param {Result} result */
function onResult(result) {
  const now = performance.now();
  state.results = state.results.filter((t) => now - t < 1000);
  state.results.push(now);
  const roundTrip = now - state.capturedAt;
  state.latency = state.latency ? state.latency * 0.8 + roundTrip * 0.2 : roundTrip;
  setStatus(
    `Live · ${state.results.length} fps · ${Math.round(state.latency)} ms · ` +
      `${result.tracks.length} moving`,
    true,
  );

  state.errors = 0;
  state.frame = { w: result.w, h: result.h };
  const ids = new Set(result.tracks.map((t) => t.id));
  for (const id of state.shown.keys()) if (!ids.has(id)) state.shown.delete(id);
  for (const track of result.tracks) {
    const shown = state.shown.get(track.id);
    state.shown.set(track.id, { track, capturedAt: state.capturedAt, display: shown?.display ?? null });
  }
  result.events.forEach(addEvent);
  scheduleNextFrame();
}

/**
 * A frame the server could not use. One is noise; a run of them means every frame
 * is failing, so say so on the page rather than keep sending into the void.
 * @param {ErrorMessage} msg
 */
function onError(msg) {
  state.errors += 1;
  if (state.errors >= 5) setStatus(`Server: ${msg.message}`, false);
  scheduleNextFrame();
}

/** Any reply frees the one frame slot; the next frame waits for the minimum gap. */
function scheduleNextFrame() {
  state.inFlight = false;
  const wait = Math.max(0, MIN_FRAME_GAP_MS - (performance.now() - state.lastSend));
  state.timer = window.setTimeout(sendFrame, wait);
}

function sendFrame() {
  const ws = state.ws;
  if (!ws || ws.readyState !== WebSocket.OPEN || state.inFlight || !ui.video.videoWidth) return;
  const scale = SEND_SIDE / Math.max(ui.video.videoWidth, ui.video.videoHeight);
  frameCanvas.width = Math.round(ui.video.videoWidth * scale);
  frameCanvas.height = Math.round(ui.video.videoHeight * scale);
  frameCanvas.getContext('2d')?.drawImage(ui.video, 0, 0, frameCanvas.width, frameCanvas.height);
  state.capturedAt = performance.now();
  state.inFlight = true;
  frameCanvas.toBlob(
    (blob) => {
      if (blob && state.ws === ws && ws.readyState === WebSocket.OPEN) {
        state.lastSend = performance.now();
        ws.send(blob);
      } else {
        state.inFlight = false;
      }
    },
    'image/jpeg',
    JPEG_QUALITY,
  );
}

// ---- settings --------------------------------------------------------------

function showValues() {
  ui.thresholdOut.value = ui.threshold.value;
  ui.minAreaOut.value = `${ui.minArea.value} px`;
}

function sendSettings() {
  if (state.ws?.readyState !== WebSocket.OPEN) return;
  state.ws.send(JSON.stringify({
    type: 'settings',
    threshold: Number(ui.threshold.value),
    minArea: Number(ui.minArea.value),
    identify: ui.identify.checked && !ui.identify.disabled,
  }));
}

let settingsTimer = 0;
function queueSettings() {
  state.touched = true;
  showValues();
  clearTimeout(settingsTimer);
  settingsTimer = window.setTimeout(sendSettings, 150);
}

// ---- drawing ---------------------------------------------------------------

function clearOverlay() {
  const ctx = ui.overlay.getContext('2d');
  ctx?.clearRect(0, 0, ui.overlay.width, ui.overlay.height);
}

/**
 * Where a box should be `ahead` seconds after its frame was captured, moving
 * each edge along the filter's velocity.
 * @param {Track} track @param {number} ahead @returns {Box}
 */
function predict(track, ahead) {
  const [x, y, w, h] = track.box;
  const [vx, vy, vw, vh] = track.velocity;
  const width = Math.max(1, w + vw * ahead);
  const height = Math.max(1, h + vh * ahead);
  const cx = x + w / 2 + vx * ahead;
  const cy = y + h / 2 + vy * ahead;
  return [cx - width / 2, cy - height / 2, width, height];
}

/** The animation loop: move every box toward its predicted position, then draw. @param {number} now */
function tick(now) {
  const dt = state.lastTick ? (now - state.lastTick) / 1000 : 0;
  state.lastTick = now;
  const ease = 1 - Math.exp(-EASE_RATE * dt);
  for (const shown of state.shown.values()) {
    const ahead = Math.min((now - shown.capturedAt) / 1000, MAX_LOOKAHEAD_S);
    const target = predict(shown.track, ahead);
    const from = shown.display;
    shown.display = from ? /** @type {Box} */ (from.map((v, i) => v + (target[i] - v) * ease)) : target;
  }
  draw();
  state.raf = requestAnimationFrame(tick);
}

/**
 * Boxes are in the small sent frame. Scale them to the displayed video
 * (object-fit: cover crops the edges) and mirror x to match the selfie view.
 */
function draw() {
  const canvas = ui.overlay;
  const dpr = window.devicePixelRatio || 1;
  const width = Math.round(canvas.clientWidth * dpr);
  const height = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const { w: frameW, h: frameH } = state.frame;
  const scale = Math.max(canvas.width / frameW, canvas.height / frameH);
  const offX = (canvas.width - frameW * scale) / 2;
  const offY = (canvas.height - frameH * scale) / 2;
  /** @param {number} x */
  const sx = (x) => canvas.width - (offX + x * scale);
  /** @param {number} y */
  const sy = (y) => offY + y * scale;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2 * dpr;
  ctx.font = `${12 * dpr}px 'Sometype Mono', ui-monospace, monospace`;
  ctx.textBaseline = 'top';

  for (const { track, display } of state.shown.values()) {
    if (!display) continue;
    const [x, y, w, h] = display;
    const left = sx(x + w);
    const top = sy(y);

    ctx.strokeStyle = 'rgba(255, 74, 28, 0.55)';
    ctx.beginPath();
    const trail = [...track.trail, [x + w / 2, y + h / 2]];
    trail.forEach(([px, py], i) => {
      if (i === 0) ctx.moveTo(sx(px), sy(py));
      else ctx.lineTo(sx(px), sy(py));
    });
    ctx.stroke();

    ctx.strokeStyle = '#ff4a1c';
    ctx.strokeRect(left, top, w * scale, h * scale);
    drawChip(ctx, track, left, top, dpr);
  }
}

/**
 * The label above a box: "#id · label", with a colour swatch for things.
 * Colour is shown as a swatch, never a word, and never for people: on a webcam
 * the moving blob is usually a person, and a colour name beside them reads as
 * describing their skin. It is for things, like a red mug.
 * @param {CanvasRenderingContext2D} ctx @param {Track} track
 * @param {number} left @param {number} top @param {number} dpr
 */
function drawChip(ctx, track, left, top, dpr) {
  const colour = track.label === 'person' ? null : track.colour;
  const text = [`#${track.id}`, track.label].filter(Boolean).join(' · ');
  const pad = 4 * dpr;
  const swatch = colour ? 10 * dpr : 0;
  const textX = left + pad + (swatch ? swatch + pad : 0);
  const width = ctx.measureText(text).width + (textX - left) + pad;
  const height = 12 * dpr + pad * 2;
  const chipY = Math.max(0, top - height);
  ctx.fillStyle = '#0d0d0d';
  ctx.fillRect(left, chipY, width, height);
  if (colour) {
    ctx.fillStyle = SWATCH[colour] ?? '#8a8a8a';
    ctx.fillRect(left + pad, chipY + (height - swatch) / 2, swatch, swatch);
  }
  ctx.fillStyle = '#ecebe6';
  ctx.fillText(text, textX, chipY + pad);
}

// ---- events and status -----------------------------------------------------

/** @param {MotionEvent} event */
function addEvent(event) {
  ui.events.querySelector('.empty')?.remove();
  const item = document.createElement('li');
  const time = document.createElement('time');
  const at = new Date(event.at * 1000);
  time.dateTime = at.toISOString();
  time.textContent = at.toLocaleTimeString();
  item.append(time, event.kind === 'start'
    ? 'Motion started'
    : `Motion ended · ${event.duration?.toFixed(1)} s`);
  ui.events.prepend(item);
  while (ui.events.children.length > MAX_EVENTS) ui.events.lastElementChild?.remove();
}

/** @param {string} text @param {boolean} live */
function setStatus(text, live) {
  ui.status.textContent = text;
  ui.status.classList.toggle('live', live);
}

// ---- wiring ----------------------------------------------------------------

ui.start.addEventListener('click', () => void start());
ui.stop.addEventListener('click', () => stop());
ui.threshold.addEventListener('input', queueSettings);
ui.minArea.addEventListener('input', queueSettings);
ui.identify.addEventListener('change', sendSettings);
showValues();
