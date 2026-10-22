"""Reference: Google's MediaPipe hand landmarker, run on the same frames.

Runs in its own throwaway environment (MediaPipe brings its own OpenCV build):

  uv run --no-project --with mediapipe==1.1.1 python -I handnet/baselines/mediapipe_ref.py val

Writes data/mediapipe/<split>.jsonl: for each image, every hand's landmark box
(the tight box around its 21 points) and handedness score. MediaPipe has no
detection score; handedness confidence stands in, so its AP is indicative.
The box padding that best matches HaGRID's looser boxes is fitted on val
by the evaluator, then frozen.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT))

import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

from handnet.frames import DATA, frames  # noqa: E402

MODEL = DATA / "mediapipe" / "hand_landmarker.task"


def main(split: str) -> None:
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=4,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
    )
    out = (DATA / "mediapipe" / f"{split}.jsonl").open("w")
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for row, frame, _ in frames(split):
            h, w = frame.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame[..., ::-1].copy())
            result = landmarker.detect(image)
            hands = []
            for points, side in zip(result.hand_landmarks, result.handedness, strict=True):
                xs = [p.x * w for p in points]
                ys = [p.y * h for p in points]
                box = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                hands.append([*box, side[0].score])
            out.write(json.dumps({"key": row["key"], "hands": hands}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "val")
