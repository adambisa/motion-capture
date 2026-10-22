# HandNet training

Everything needed to rebuild `../weights/hand.onnx` from public data. This is
its own uv project: PyTorch lives here and never in the server image.

## Reproduce

```bash
cd train
uv sync
uv run python -m handnet.gate                 # speed and export check, untrained
uv run python -m handnet.prepare_data         # ~35k HaGRIDv2 images, ~1 GB, by HTTP Range
uv run python -m handnet.train --name base    # ~2 h on an Apple Silicon GPU (MPS)
uv run python -m handnet.release runs/base/best.pt   # -> ../weights/hand.onnx + hand.json
uv run python -m handnet.evaluate ../weights/hand.onnx --split val
uv run python -m handnet.ablate               # optional, ~2 h
```

Baselines, on the same frames and metric code:

```bash
uv run python -m handnet.evaluate skin --split val
uv run python -m handnet.evaluate skin-faces --split val
uv run --no-project --with mediapipe==1.1.1 python -I handnet/baselines/mediapipe_ref.py val
uv run python -m handnet.evaluate mediapipe --split val
```

The test split is touched once, for the final numbers, with `--split test --final`;
every use is logged in `../docs/results/test_access.jsonl`.

## Layout

| File | What it does |
|---|---|
| `handnet/model.py` | HandNet: MobileNetV2-style backbone, light FPN, 5-channel CenterNet head |
| `handnet/targets.py`, `loss.py` | Gaussian heatmap targets; focal + L1 loss |
| `handnet/data.py` | Dataset and webcam-style augmentation |
| `handnet/frames.py` | The demo's view of an image: 320 px, browser-quality JPEG |
| `handnet/metrics.py` | AP with people-resampled confidence intervals (checked against pycocotools) |
| `handnet/train.py`, `evaluate.py`, `release.py`, `ablate.py` | Train, score, ship, compare |
| `handnet/remote_zip.py`, `prepare_data.py` | Fetch only the chosen images out of the 119 GB zip |
| `handnet/baselines/` | Skin-colour baseline and the MediaPipe reference |

Decoding lives in `../motion/hands.py` and is imported here, so evaluation
measures exactly the code the server runs.

## Data and licences

- **HaGRIDv2** (https://github.com/hukenovs/hagrid): a variant of CC BY-SA 4.0.
  Ten one-handed gestures plus `no_gesture`, HaGRID's own person-disjoint
  train/val/test split, at most 4 images per person and gesture in training.
  No HaGRID images are committed. The trained weights are released under the
  same share-alike terms: `../weights/LICENSE`.
- **MediaPipe Hand Landmarker** (Apache-2.0): reference baseline only, downloaded, not committed.
- **YuNet face detector** (MIT, OpenCV model zoo): only for the skin-colour
  baseline's face removal, downloaded, not committed.
