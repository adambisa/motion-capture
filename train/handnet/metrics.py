"""Average precision for one class, with error bars that resample people.

pycocotools gives one number. For confidence intervals we need to resample
*people* (all images of a person in or out together, since their images are
not independent), recomputing AP each time, so matching is done once per
image here and AP is recomputed cheaply from the stored matches.
test_metrics.py checks AP50 against pycocotools.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Box = tuple[float, float, float, float]  # x, y, w, h


@dataclass(frozen=True, slots=True)
class Matched:
    scores: NDArray[np.float64]
    tp: NDArray[np.bool_]
    n_gt: int


def iou(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """(N, 4) x (M, 4) xywh -> (N, M)."""
    ax1, ay1 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1[:, None], bx1) - np.maximum(a[:, 0, None], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1) - np.maximum(a[:, 1, None], b[:, 1]), 0, None)
    inter = iw * ih
    union = a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1) - inter
    return inter / np.maximum(union, 1e-9)


def match(dets: list[tuple[Box, float]], gts: list[Box], threshold: float = 0.5) -> Matched:
    """Greedy by score: each detection takes the best still-free ground truth above threshold."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i][1])
    scores = np.array([dets[i][1] for i in order], np.float64)
    tp = np.zeros(len(order), bool)
    if dets and gts:
        overlaps = iou(np.array([dets[i][0] for i in order], np.float64), np.array(gts, np.float64))
        taken = np.zeros(len(gts), bool)
        for k in range(len(order)):
            cand = np.where(taken, -1, overlaps[k])
            j = int(cand.argmax())
            if cand[j] >= threshold:
                tp[k], taken[j] = True, True
    return Matched(scores, tp, len(gts))


def average_precision(images: list[Matched]) -> float:
    """101-point interpolated AP over a set of images (COCO's definition)."""
    n_gt = sum(m.n_gt for m in images)
    if n_gt == 0:
        return float("nan")
    scores = np.concatenate([m.scores for m in images]) if images else np.zeros(0)
    tp = np.concatenate([m.tp for m in images]) if images else np.zeros(0, bool)
    order = np.argsort(-scores, kind="mergesort")
    tp = tp[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # make it monotone
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    return float(
        np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0).mean()
    )


def at_threshold(images: list[Matched], threshold: float) -> dict[str, float]:
    tp = sum(int(m.tp[m.scores >= threshold].sum()) for m in images)
    fp = sum(int((~m.tp[m.scores >= threshold]).sum()) for m in images)
    n_gt = sum(m.n_gt for m in images)
    p = tp / max(tp + fp, 1)
    r = tp / max(n_gt, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)}


def bootstrap(
    images: list[Matched], people: list[str], n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% interval of AP, resampling people with replacement."""
    by_person: dict[str, list[Matched]] = defaultdict(list)
    for m, person in zip(images, people, strict=True):
        by_person[person].append(m)
    groups = list(by_person.values())
    rng = np.random.default_rng(seed)
    aps = [
        average_precision(
            [m for g in rng.integers(len(groups), size=len(groups)) for m in groups[g]]
        )
        for _ in range(n)
    ]
    lo, hi = np.percentile(aps, [2.5, 97.5])
    return float(lo), float(hi)
