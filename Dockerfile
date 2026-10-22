# syntax=docker/dockerfile:1.7
FROM python:3.13-slim-bookworm AS deps
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.13-slim-bookworm
WORKDIR /app
# YOLOX-nano, Apache-2.0, from the official Megvii release (pinned by checksum).
ADD --checksum=sha256:c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d \
    https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx \
    /app/models/yolox_nano.onnx
COPY --from=deps /app/.venv /app/.venv
COPY motion ./motion
COPY server.py ./
COPY web ./web
# HandNet, trained for this project (see train/); weights under the HaGRID licence.
COPY weights ./weights
RUN useradd --system --uid 10001 app && chmod 644 /app/models/yolox_nano.onnx
USER app
# OPENCV_IO_MAX_IMAGE_PIXELS backs up the server's own JPEG header check.
# FORWARDED_ALLOW_IPS: set to your proxy's address so per-visitor limits see real IPs.
ENV PATH="/app/.venv/bin:$PATH" PORT=8000 PYTHONUNBUFFERED=1 \
    OPENCV_IO_MAX_IMAGE_PIXELS=1000000 FORWARDED_ALLOW_IPS=127.0.0.1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"
# One worker on purpose: the session limits live in this process.
# --ws-max-size refuses oversized messages before they are buffered.
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --ws-max-size 524288"]
