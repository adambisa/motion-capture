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
