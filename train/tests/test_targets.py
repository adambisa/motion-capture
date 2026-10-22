import numpy as np

from handnet.targets import encode
from motion.hands import decode


def test_encode_then_decode_gives_the_boxes_back() -> None:
    boxes = np.array([[30, 40, 48, 64], [150, 90, 20, 30]], np.float32)
    heat, reg, mask = encode(boxes, 192, 256)
    assert heat.shape == (48, 64)
    assert heat.max() == 1.0
    assert mask.sum() == 2  # one supervised cell per hand
    # Build the raw output a perfect network would give: logit of the heatmap, regression channels.
    logit = np.log(np.clip(heat, 1e-6, 1 - 1e-6) / (1 - np.clip(heat, 1e-6, 1 - 1e-6)))
    raw = np.concatenate([logit[None], reg], axis=0)[None].astype(np.float32)
    found = sorted(decode(raw, ratio=1.0, bounds=(256, 192), threshold=0.5), key=lambda h: h.box)
    assert [h.box for h in found] == [(30, 40, 48, 64), (150, 90, 20, 30)]


def test_tiny_boxes_are_not_targets() -> None:
    heat, _, mask = encode(np.array([[10, 10, 3, 3]], np.float32), 192, 256)
    assert heat.max() == 0
    assert mask.sum() == 0
