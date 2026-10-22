"""Classic baseline: skin colour, then shapes. No learning, just thresholds.

Skin pixels in YCrCb (the standard Cr/Cb box), cleaned with morphology, each
blob big enough becomes a box. It finds faces and arms as readily as hands,
so variant B first removes faces found by OpenCV's YuNet face detector.
Score = how solid the blob is, scaled by size: rough, but monotone, so AP is defined.
"""

from pathlib import Path

import cv2

from motion.common import Image
from motion.hands import HandBox

CR = (135, 175)
CB = (80, 130)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
YUNET = Path(__file__).resolve().parents[2] / "data" / "face_yunet.onnx"


class Skin:
    def __init__(self, remove_faces: bool = False, min_area: float = 0.002) -> None:
        self.min_area = min_area
        self.faces = (
            cv2.FaceDetectorYN.create(str(YUNET), "", (320, 240), 0.6) if remove_faces else None
        )

    def __call__(self, frame: Image) -> list[HandBox]:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, (0, CR[0], CB[0]), (255, CR[1], CB[1]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
        if self.faces is not None:
            h, w = frame.shape[:2]
            self.faces.setInputSize((w, h))
            _, found = self.faces.detect(frame)
            for f in found if found is not None else []:
                x, y, fw, fh = (int(v) for v in f[:4])
                # The face box, grown to cover the neck and hair that read as skin.
                cv2.rectangle(
                    mask, (x - fw // 4, y - fh // 4), (x + fw + fw // 4, y + fh + fh // 2), 0, -1
                )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_min = self.min_area * frame.shape[0] * frame.shape[1]
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < area_min:
                continue
            hull = cv2.contourArea(cv2.convexHull(c))
            solidity = area / max(hull, 1.0)
            x, y, w, h = cv2.boundingRect(c)
            size = min(1.0, area / (0.05 * frame.shape[0] * frame.shape[1]))
            out.append(HandBox((x, y, w, h), float(solidity * (0.5 + 0.5 * size))))
        return out
