"""Gate 0: before any training, is the architecture fast enough and exportable?

For each (input size, width): count params and MACs, export an untrained
net, check cv2.dnn matches torch, and time cv2.dnn on one thread.
"""

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from handnet.export import blob, export, parity
from handnet.model import HandNet, complexity


def time_cv2(path: Path, height: int, width: int, runs: int = 200) -> tuple[float, float]:
    cv2.setNumThreads(1)
    net = cv2.dnn.readNetFromONNX(str(path))
    image = np.random.default_rng(1).integers(0, 256, (height, width, 3), dtype=np.uint8)
    times = []
    for i in range(runs + 20):
        t0 = time.perf_counter()
        net.setInput(blob(image, height, width))
        net.forward()
        if i >= 20:
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        header = ("input", "width", "params", "MMACs", "parity", "p50 ms", "p95 ms")
        print(" ".join(f"{h:>9}" for h in header))
        for height, width in ((192, 256), (288, 384)):
            for w in (0.5, 0.75, 1.0):
                model = HandNet(w)
                params, macs = complexity(model, height, width)
                path = Path(tmp) / f"h{height}_w{w}.onnx"
                export(model, path, height, width)
                diff = parity(model, path, height, width, n=3)
                p50, p95 = time_cv2(path, height, width)
                cells = (
                    f"{height}x{width}",
                    w,
                    f"{params / 1e6:.2f}M",
                    f"{macs / 1e6:.0f}",
                    f"{diff:.1e}",
                    f"{p50:.2f}",
                    f"{p95:.2f}",
                )
                print(" ".join(f"{c:>9}" for c in cells))
