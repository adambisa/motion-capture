"""Build the local HaGRID subset: pick images, fetch only those, downscale, write a manifest.

Selection is deterministic (sorted by a hash of user_id and image id), uses
HaGRID's official user-disjoint splits, and caps images per person so the
training set is about many people, not many photos of a few.

  uv run python -m handnet.prepare_data            # full subset
  uv run python -m handnet.prepare_data --limit 50 # quick timing run
"""

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from handnet.remote_zip import HAGRID_512, Member, read_member

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "raw"
IMAGES = DATA / "images"
MANIFEST = DATA / "manifest.jsonl"

# One-handed gestures, chosen for variety of hand shapes; two-handed classes are left out.
CLASSES = ("palm", "stop", "fist", "one", "peace", "like", "ok", "call", "four", "rock")
CAPS = {"train": 2500, "val": 300, "test": 500}  # images per class and split
PER_USER = {"train": 4, "val": 2, "test": 2}  # images per person, per class
LONG_SIDE = 384  # stored size: room for the 288x384 ablation, the demo uses 192x256


def _order(user: str, key: str) -> str:
    return hashlib.sha1(f"{user}/{key}".encode()).hexdigest()


def select(split: str, cls: str) -> list[dict[str, object]]:
    records = json.loads((RAW / "annotations" / split / f"{cls}.json").read_text())
    per_user: dict[str, int] = defaultdict(int)
    chosen = []
    cap = None if cls == "no_gesture" else CAPS[split]
    for key, r in sorted(records.items(), key=lambda kv: _order(kv[1]["user_id"], kv[0])):
        if r.get("united_bbox"):
            continue
        if cls != "no_gesture" and per_user[r["user_id"]] >= PER_USER[split]:
            continue
        per_user[r["user_id"]] += 1
        chosen.append(
            {
                "key": key,
                "split": split,
                "cls": cls,
                "user": r["user_id"],
                "boxes": r["bboxes"],  # normalised [x, y, w, h], one per hand; every hand is "hand"
                "meta": r.get("meta", {}),
            }
        )
        if cap is not None and len(chosen) >= cap:
            break
    return chosen


def store(data: bytes, path: Path) -> None:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"undecodable image for {path.name}")
    h, w = image.shape[:2]
    scale = LONG_SIDE / max(h, w)
    if scale < 1:
        image = cv2.resize(
            image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.jpg")  # imwrite picks the format from the extension
    if not cv2.imwrite(str(tmp), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"could not write {tmp}")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only this many images (timing run)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--manifest-only", action="store_true", help="index what is already downloaded")
    args = ap.parse_args()

    rows = [r for split in CAPS for cls in (*CLASSES, "no_gesture") for r in select(split, cls)]
    users = {s: {r["user"] for r in rows if r["split"] == s} for s in CAPS}
    leaks = (
        (users["train"] & users["val"])
        | (users["train"] & users["test"])
        | (users["val"] & users["test"])
    )
    assert not leaks, f"{len(leaks)} people appear in more than one split"
    print(
        {s: sum(r["split"] == s for r in rows) for s in CAPS},
        "images;",
        {s: len(u) for s, u in users.items()},
        "people",
    )

    index = {m["name"]: Member(**m) for m in json.loads((RAW / "zip_index.json").read_text())}
    todo = [r for r in rows if not (IMAGES / f"{r['key']}.jpg").exists()]
    if args.limit:
        todo = todo[: args.limit]
    if args.manifest_only:
        todo = []
    print(f"{len(todo)} images to fetch")

    def job(r: dict[str, object]) -> None:
        m = index[f"HaGRIDv2_dataset_512/{r['cls']}/{r['key']}.jpg"]
        store(read_member(HAGRID_512, m), IMAGES / f"{r['key']}.jpg")

    failed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(job, r) for r in todo]
        for i, f in enumerate(as_completed(futures), 1):
            if f.exception():
                failed += 1
            if i % 1000 == 0:
                print(f"  {i}/{len(todo)} ({failed} failed)", flush=True)

    have = [r for r in rows if (IMAGES / f"{r['key']}.jpg").exists()]
    with MANIFEST.open("w") as out:
        for r in have:
            out.write(json.dumps(r) + "\n")
    print(f"manifest: {len(have)} images, {failed} failed this run")


if __name__ == "__main__":
    main()
