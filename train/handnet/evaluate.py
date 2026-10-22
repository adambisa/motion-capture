"""Score a detector the way the live demo uses it.

Every image goes through the demo's path (handnet.frames: 320 px long side,
browser-quality JPEG), then the detector; HandNet is decoded by motion.hands,
the server's own code. Ground-truth hands smaller than 8 px at that scale are
ignored, along with detections on them.

  uv run python -m handnet.evaluate weights/hand.onnx --split val          # any time
  uv run python -m handnet.evaluate weights/hand.onnx --split test --final  # once, at the end
  uv run python -m handnet.evaluate skin --split val          # baseline: skin colour
  uv run python -m handnet.evaluate skin-faces --split val    # baseline: skin colour minus faces
  uv run python -m handnet.evaluate mediapipe --split val     # reference (run its script first)
"""

import argparse
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from handnet.frames import DATA, as_frame, frames
from handnet.metrics import Matched, at_threshold, average_precision, bootstrap, iou, match
from motion.hands import HandBox, HandDetector

IGNORE_BELOW = 8.0  # px, long side of a hand in the 320 px frame
IOUS = np.round(np.arange(0.5, 0.96, 0.05), 2)
RESULTS = Path(__file__).resolve().parents[2] / "docs" / "results"

Predict = Callable[[NDArray[np.uint8], dict], list[HandBox]]  # (frame, its row) -> hands
Frames = Iterable[tuple[dict, NDArray[np.uint8], NDArray[np.float32]]]

__all__ = ["as_frame", "run", "score_image", "summarise"]


def score_image(found: list[HandBox], gts: NDArray[np.float32]) -> list[Matched]:
    """Matches at every IoU threshold, after removing tiny ground truth and detections on it."""
    big = np.maximum(gts[:, 2], gts[:, 3]) >= IGNORE_BELOW if len(gts) else np.zeros(0, bool)
    keep, ignored = gts[big], gts[~big]
    dets = [(h.box, h.score) for h in found]
    if len(ignored) and dets:
        boxes = np.array([d[0] for d in dets], float)
        on_ignored = iou(boxes, ignored.astype(float)).max(1) >= 0.5
        dets = [d for d, drop in zip(dets, on_ignored, strict=True) if not drop]
    gt_list = [tuple(map(float, b)) for b in keep]
    return [match(dets, gt_list, t) for t in IOUS]  # type: ignore[arg-type]


def run(predict: Predict, data: Frames) -> tuple[list[list[Matched]], list[dict], float]:
    """Per-image matches, the rows they belong to, and median ms per frame."""
    per_image, kept, times = [], [], []
    for row, frame, gts in data:
        t0 = time.perf_counter()
        found = predict(frame, row)
        times.append((time.perf_counter() - t0) * 1000)
        per_image.append(score_image(found, gts))
        kept.append(row)
    return per_image, kept, float(np.median(times)) if times else 0.0


def summarise(per_image: list[list[Matched]], rows: list[dict], threshold: float, ci: bool) -> dict:
    at50 = [m[0] for m in per_image]
    out = {
        "images": len(rows),
        "hands": sum(m.n_gt for m in at50),
        "ap50": average_precision(at50),
        "ap50_95": float(
            np.mean([average_precision([m[k] for m in per_image]) for k in range(len(IOUS))])
        ),
        "threshold": threshold,
        **at_threshold(at50, threshold),
    }
    if ci:
        out["ap50_ci95"] = bootstrap(at50, [r["user"] for r in rows])
    return out


def by_group(per_image: list[list[Matched]], rows: list[dict], key: str, threshold: float) -> dict:
    """AP50 and recall per value of a metadata field (e.g. HaGRID's automatic 'race' estimate)."""
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        value = r.get("meta", {}).get(key)
        value = value[0] if isinstance(value, list) and value else value
        if value is not None:
            groups.setdefault(str(value), []).append(i)
    return {
        g: summarise([per_image[i] for i in ix], [rows[i] for i in ix], threshold, ci=True)
        for g, ix in sorted(groups.items())
        if len(ix) >= 50
    }


def onnx_predictor(path: Path, threshold: float) -> Predict:
    detector = HandDetector.load(path)
    if detector is None:
        raise SystemExit(f"could not load {path}")
    cv2.setNumThreads(1)
    return lambda frame, _row: detector.detect(frame, threshold)


def mediapipe_predictor(split: str, pad: float) -> Predict:
    """Recorded MediaPipe output, each landmark box grown by `pad` of its size per side."""
    recorded = {}
    for line in (DATA / "mediapipe" / f"{split}.jsonl").open():
        r = json.loads(line)
        recorded[r["key"]] = r["hands"]

    def predict(frame: NDArray[np.uint8], row: dict) -> list[HandBox]:
        h, w = frame.shape[:2]
        out = []
        for x, y, bw, bh, score in recorded[row["key"]]:
            x0, y0 = max(0, x - pad * bw), max(0, y - pad * bh)
            x1, y1 = min(w, x + bw + pad * bw), min(h, y + bh + pad * bh)
            out.append(HandBox((round(x0), round(y0), round(x1 - x0), round(y1 - y0)), score))
        return out

    return predict


def fit_mediapipe_pad(split: str = "val") -> float:
    """The padding that makes MediaPipe's boxes best match HaGRID's, chosen on val only."""
    data = frames(split)
    scored = []
    for pad in np.round(np.arange(0.0, 0.41, 0.05), 2):
        per_image = run(mediapipe_predictor(split, float(pad)), data)[0]
        scored.append((average_precision([m[0] for m in per_image]), float(pad)))
    return max(scored)[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("detector", help="path to a .onnx, or: skin, skin-faces, mediapipe")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true", help="required for the test split")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("the test split is for the final numbers only: pass --final")

    extra: dict[str, object] = {}
    predict: Predict
    if args.detector in ("skin", "skin-faces"):
        from handnet.baselines.skin import Skin

        skin = Skin(remove_faces=args.detector == "skin-faces")
        cv2.setNumThreads(1)
        predict, threshold, name = (lambda frame, _row: skin(frame)), 0.5, args.detector
    elif args.detector == "mediapipe":
        pad = fit_mediapipe_pad("val")
        predict, threshold, name = mediapipe_predictor(args.split, pad), 0.5, "mediapipe"
        extra["box_padding_fitted_on_val"] = pad
    else:
        onnx = Path(args.detector)
        meta = json.loads(onnx.with_suffix(".json").read_text())
        predict = onnx_predictor(onnx, threshold=0.05)  # low: AP needs the whole score range
        threshold, name = meta["spawn"], onnx.stem

    per_image, rows, ms = run(predict, frames(args.split))
    result = {
        "detector": name,
        "split": args.split,
        "median_ms_per_frame_one_thread": ms,
        **extra,
        **summarise(per_image, rows, threshold, ci=True),
        "by_skin_tone_estimate": by_group(per_image, rows, "race", threshold),
        "by_gender_estimate": by_group(per_image, rows, "gender", threshold),
    }
    print(json.dumps(result, indent=2))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = args.name or f"{name}_{args.split}"
    (RESULTS / f"{out}.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.split == "test":
        with (RESULTS / "test_access.jsonl").open("a") as log:
            entry = {"detector": name, "at": time.strftime("%Y-%m-%d %H:%M")}
            log.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
