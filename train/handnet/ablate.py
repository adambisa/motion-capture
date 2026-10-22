"""Ablations: what input size, width and webcam augmentation buy, at what speed.

Each variant trains on a shorter budget than the shipped model (30% of the
training images, 30 epochs) so the set runs in an afternoon; compare variants
with each other, not with the full run. Each is exported to ONNX and timed in
cv2.dnn on one thread. Results: docs/results/ablations.json.

  uv run python -m handnet.ablate
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from handnet.export import export, parity
from handnet.gate import time_cv2
from handnet.model import HandNet

RUNS = Path(__file__).resolve().parent.parent / "runs"
OUT = Path(__file__).resolve().parents[2] / "docs" / "results" / "ablations.json"
BUDGET = ["--set", "fraction=0.3", "--set", "epochs=30", "--set", "eval_every=10"]

VARIANTS: dict[str, list[str]] = {
    "abl_base": [],
    "abl_input288": ["--set", "height=288", "--set", "input_width=384"],
    "abl_width050": ["--set", "width=0.5"],
    "abl_width075": ["--set", "width=0.75"],
    "abl_no_webcam_aug": ["--set", 'augment=["geometric","photometric"]'],
}


def best_val(run: Path) -> dict:
    rows = [json.loads(line) for line in (run / "metrics.jsonl").open()]
    return max((r for r in rows if "val" in r), key=lambda r: r["val"]["ap50"])


def main() -> None:
    results = {}
    for name, extra in VARIANTS.items():
        run = RUNS / name
        if not (run / "best.pt").exists():
            cmd = [sys.executable, "-m", "handnet.train", "--name", name, *BUDGET, *extra]
            subprocess.run(cmd, check=True)
        ckpt = torch.load(run / "best.pt", map_location="cpu")
        cfg = ckpt["config"]
        model = HandNet(cfg["width"])
        model.load_state_dict(ckpt["ema"])
        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "m.onnx"
            export(model, onnx, cfg["height"], cfg["input_width"])
            diff = parity(model, onnx, cfg["height"], cfg["input_width"], n=3)
            p50, p95 = time_cv2(onnx, cfg["height"], cfg["input_width"])
        best = best_val(run)
        results[name] = {
            "input": f"{cfg['height']}x{cfg['input_width']}",
            "width": cfg["width"],
            "augment": cfg["augment"],
            "params": cfg["params"],
            "macs": cfg["macs"],
            "best_epoch": best["epoch"],
            "val_ap50": best["val"]["ap50"],
            "val_ap50_95": best["val"]["ap50_95"],
            "cv2_ms_p50_one_thread_mac": round(p50, 2),
            "cv2_ms_p95_one_thread_mac": round(p95, 2),
            "torch_vs_cv2_max_diff": float(f"{diff:.1e}"),
        }
        print(name, json.dumps(results[name]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
