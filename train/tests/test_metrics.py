import numpy as np
import pytest

from handnet.metrics import Matched, average_precision, bootstrap, match


def random_case(seed: int) -> tuple[list[list], list[list]]:
    rng = np.random.default_rng(seed)
    gts, dets = [], []
    for _ in range(40):
        g = [
            [float(x), float(y), float(w), float(h)]
            for x, y, w, h in rng.uniform(
                [0, 0, 10, 10], [200, 200, 60, 60], (rng.integers(0, 4), 4)
            )
        ]
        d = [
            (
                [
                    x + rng.normal(0, 6),
                    y + rng.normal(0, 6),
                    w * rng.uniform(0.8, 1.2),
                    h * rng.uniform(0.8, 1.2),
                ],
                float(rng.uniform()),
            )
            for x, y, w, h in g
            if rng.random() < 0.8
        ]
        d += [
            (
                [float(v) for v in rng.uniform([0, 0, 10, 10], [200, 200, 60, 60])],
                float(rng.uniform(0, 0.6)),
            )
            for _ in range(rng.integers(0, 3))
        ]
        gts.append(g)
        dets.append(d)
    return gts, dets


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ap50_matches_pycocotools(seed: int) -> None:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gts, dets = random_case(seed)
    ours = average_precision(
        [
            match([(tuple(b), s) for b, s in d], [tuple(b) for b in g])
            for g, d in zip(gts, dets, strict=True)
        ]
    )

    coco = COCO()
    coco.dataset = {
        "images": [{"id": i} for i in range(len(gts))],
        "categories": [{"id": 1, "name": "hand"}],
        # ids start at 1: pycocotools treats id 0 as "no match"
        "annotations": [
            {
                "id": k + 1,
                "image_id": i,
                "category_id": 1,
                "bbox": b,
                "area": b[2] * b[3],
                "iscrowd": 0,
            }
            for k, (i, b) in enumerate((i, b) for i, g in enumerate(gts) for b in g)
        ],
    }
    coco.createIndex()
    results = coco.loadRes(
        [
            {"image_id": i, "category_id": 1, "bbox": b, "score": s}
            for i, d in enumerate(dets)
            for b, s in d
        ]
    )
    ev = COCOeval(coco, results, "bbox")
    ev.params.iouThrs = np.array([0.5])
    ev.params.areaRng = [[0, 1e10]]
    ev.params.areaRngLbl = ["all"]
    ev.params.maxDets = [100]
    ev.evaluate()
    ev.accumulate()
    reference = float(ev.eval["precision"][0, :, 0, 0, 0].mean())
    assert ours == pytest.approx(reference, abs=1e-3)


def test_perfect_detector_scores_one_and_bootstrap_brackets_it() -> None:
    images = [Matched(np.array([0.9]), np.array([True]), 1) for _ in range(30)]
    assert average_precision(images) == pytest.approx(1.0)
    lo, hi = bootstrap(images, [str(i // 3) for i in range(30)], n=50)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)
