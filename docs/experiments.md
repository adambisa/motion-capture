# HandNet experiment log

Short notes on what was tried and what happened, in order. Numbers are from
`train/runs/<name>/metrics.jsonl`; the write-up draws on this.

## 2026-10-22 · Gates before training

- **Speed (untrained net, cv2.dnn, one thread, Mac M-series):** 192x256 input: 1.5 ms (width 0.5),
  2.6 ms (0.75), 3.5 ms (1.0). 288x384: 3.4 / 5.7 / 7.5 ms. torch vs cv2.dnn outputs agree to 1e-6.
  Input sides must be multiples of 32 (five stride-2 stages), so 240x320 is out.
- **Training throughput (batch 64, width 1.0):** MPS 386 img/s at 192x256, CPU 17 img/s. Train on MPS.

## 2026-10-22 · Overfit test (64 training images, no augmentation, 1,200 steps)

- **Centre-cell regression (CenterNet):** loss 0.047, AP50 0.91, recall 0.90 on the memorised images.
  The misses were all large hands with boxes about half the right size; size loss was still
  falling when the learning rate ran out. Targets round-trip exactly through the decoder, so
  not a bug: size simply learns slower than the heatmap with one supervised cell per hand.
- **Dense regression (TTFNet-style, every cell with bump >= 0.3, weighted):** worse. AP50 0.87,
  recall 0.72, size loss stalled at 0.14. For a network this small, making every nearby cell
  predict the offset to the centre from its own position costs more than the extra signal gives.
- **Decision:** keep centre-cell regression; a real run has ~20x the steps.

## 2026-10-22 · Full run and release (base, 60 epochs, ~26k images)

- **Val (3,200 images, through ONNX + cv2.dnn):** AP50 0.674, AP50:95 0.252, precision 0.92 and
  recall 0.49 at the release threshold. Reference on the same frames: MediaPipe AP50 0.80
  [0.77-0.84], skin colour 0.003.
- **Released thresholds:** start a track at 0.50 (lowest score with val precision >= 0.90), keep it at 0.25.
- **Measured box wobble on a still hand** (std, px): centre 1.2 / 1.3, size 2.0 / 2.7. Now the
  Kalman measurement noise.
- **First live test (owner, portrait webcam):** misses hands, some boxes on the wrong things, flicker.
  On val, portrait images score lower than landscape (AP50 0.66 [0.62-0.70] vs 0.72 [0.64-0.80];
  intervals overlap). A portrait frame is letterboxed into the landscape canvas at 0.6 scale,
  shrinking hands. Next: a portrait export from the same weights, then numbers-only logging from a
  live session. Ablations and the one-time test-set evaluation not run yet.

## 2026-10-22 · Portrait canvas (no retraining)

HandNet is fully convolutional, so the same weights were exported a second time on a
portrait canvas (256 tall x 192 wide), and the detector now picks the canvas matching
each frame. On the 2,438 portrait val images: AP50 0.658 [0.62-0.70] -> **0.766 [0.74-0.80]**,
AP50:95 0.241 -> 0.309, precision 0.92 -> 0.94, recall 0.47 -> 0.58, same 3.5 ms.
Squeezing portrait frames into the landscape canvas was costing a lot.
