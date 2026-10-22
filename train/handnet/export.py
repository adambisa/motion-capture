"""Export HandNet to ONNX and check OpenCV's DNN module gives the same answer."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from handnet.model import HandNet


def export(model: HandNet, path: Path, height: int, width: int) -> None:
    model.eval()
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, height, width),),
        str(path),
        input_names=["image"],
        output_names=["out"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def blob(image_bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    """The server's preprocessing: RGB, scaled to -1..1 (x / 127.5 - 1)."""
    return cv2.dnn.blobFromImage(image_bgr, 1 / 127.5, (width, height), (127.5,) * 3, swapRB=True)


def parity(model: HandNet, path: Path, height: int, width: int, n: int = 8) -> float:
    """Largest absolute difference between torch and cv2.dnn outputs on random images."""
    net = cv2.dnn.readNetFromONNX(str(path))
    rng = np.random.default_rng(0)
    worst = 0.0
    model.eval()
    for _ in range(n):
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        b = blob(image, height, width)
        net.setInput(b)
        ours = net.forward()
        with torch.no_grad():
            ref = model(torch.from_numpy(b)).numpy()
        worst = max(worst, float(np.abs(ours - ref).max()))
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = HandNet(cfg["width"])
    model.load_state_dict(ckpt["ema"])
    export(model, args.out, cfg["height"], cfg["input_width"])
    diff = parity(model, args.out, cfg["height"], cfg["input_width"])
    print(f"exported {args.out}; max |torch - cv2.dnn| = {diff:.2e}")
    if diff > 1e-3:
        raise SystemExit("parity check failed")
