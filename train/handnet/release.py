"""Turn a training run into the shipped model: weights/hand.onnx plus hand.json.

  uv run python -m handnet.release runs/base/best.pt

1. Export the EMA weights to ONNX; check cv2.dnn agrees with torch.
2. Pick the two tracking thresholds on val (never test):
   spawn = lowest score at which val precision (IoU 0.5) reaches 0.90,
   keep  = half of that, enough to continue a track but not to start one.
3. Measure how much the boxes wobble when nothing moves (the same val image
   re-shot: 1-2 px shifts, JPEG quality, noise, brightness) and store the
   standard deviations as the Kalman filter's measurement noise.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from handnet.evaluate import onnx_predictor, run, summarise
from handnet.export import export, parity
from handnet.frames import frames
from handnet.metrics import Matched, iou
from handnet.model import HandNet
from motion.hands import HandDetector

WEIGHTS = Path(__file__).resolve().parents[2] / "weights"
TARGET_PRECISION = 0.90


def pick_spawn(per_image: list[list[Matched]]) -> float:
    """Lowest score threshold whose precision at IoU 0.5 reaches the target."""
    at50 = [m[0] for m in per_image]
    scores = np.concatenate([m.scores for m in at50])
    tp = np.concatenate([m.tp for m in at50])
    for t in np.round(np.arange(0.20, 0.91, 0.05), 2):
        kept = scores >= t
        if kept.any() and tp[kept].mean() >= TARGET_PRECISION:
            return float(t)
    return 0.9


def jitter(
    detector: HandDetector, samples: int = 150, shots: int = 12, seed: int = 0
) -> list[float]:
    """Std of a still hand's box centre x, y, width, height (px) across re-shot frames."""
    rng = np.random.default_rng(seed)
    data = [(row, f, g) for row, f, g in frames("val") if len(g) == 1][:samples]
    spreads = []
    for _, frame, gts in data:
        h, w = frame.shape[:2]
        boxes = []
        for _ in range(shots):
            dx, dy = rng.integers(-2, 3, 2)
            shot = cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)  # fmt: skip
            shot = np.clip(shot.astype(np.float32) * rng.uniform(0.95, 1.05)
                           + rng.normal(0, 3, shot.shape), 0, 255).astype(np.uint8)  # fmt: skip
            _, buf = cv2.imencode(
                ".jpg", shot, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 91))]
            )
            found = detector.detect(cv2.imdecode(buf, cv2.IMREAD_COLOR), detector.contract.keep)
            if not found:
                continue
            truth = np.array([[gts[0][0] + dx, gts[0][1] + dy, gts[0][2], gts[0][3]]], float)
            best = max(found, key=lambda f: iou(np.array([f.box], float), truth)[0, 0])
            if iou(np.array([best.box], float), truth)[0, 0] < 0.5:
                continue
            x, y, bw, bh = best.box
            boxes.append((x + bw / 2 - dx, y + bh / 2 - dy, bw, bh))  # undo the known shift
        if len(boxes) >= shots // 2:
            spreads.append(np.std(np.array(boxes), axis=0))
    return [round(float(v), 2) for v in np.median(np.array(spreads), axis=0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, default=WEIGHTS / "hand.onnx")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = HandNet(cfg["width"])
    model.load_state_dict(ckpt["ema"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export(model, args.out, cfg["height"], cfg["input_width"])
    diff = parity(model, args.out, cfg["height"], cfg["input_width"])
    if diff > 1e-3:
        raise SystemExit(f"cv2.dnn disagrees with torch by {diff:.2e}")

    meta_path = args.out.with_suffix(".json")
    contract = {"height": cfg["height"], "width": cfg["input_width"], "spawn": 0.5, "keep": 0.25}
    meta_path.write_text(json.dumps(contract))
    per_image, rows, _ = run(onnx_predictor(args.out, 0.05), frames("val"))
    spawn = pick_spawn(per_image)
    contract.update(spawn=spawn, keep=round(spawn / 2, 3))
    meta_path.write_text(json.dumps(contract))

    detector = HandDetector.load(args.out)
    assert detector is not None
    contract["noise"] = jitter(detector)
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    contract.update(
        {
            "model": "HandNet",
            "trained_on": "HaGRIDv2 subset, see train/README.md",
            "licence": "weights/LICENSE (HaGRID share-alike terms)",
            "run": str(args.checkpoint.parent.name),
            "epoch": ckpt["epoch"],
            "width_multiplier": cfg["width"],
            "params": cfg["params"],
            "macs": cfg["macs"],
            "torch_vs_cv2_max_diff": float(f"{diff:.2e}"),
            "val": summarise(per_image, rows, spawn, ci=False),
            "git": sha,
            "released": time.strftime("%Y-%m-%d"),
        }
    )
    meta_path.write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
