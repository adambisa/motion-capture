"""Train HandNet.

  uv run python -m handnet.train --name base
  uv run python -m handnet.train --name w075 --set width=0.75
  uv run python -m handnet.train --name overfit --overfit 64   # sanity check
  uv run python -m handnet.train --name base --resume          # continue a stopped run

Each run writes runs/<name>/: config.json, env.json, metrics.jsonl, best.pt
(by val AP50, of the EMA weights) and last.pt.
"""

import argparse
import copy
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from handnet.data import ALL_AUGMENT, HandData, load_rows, normalise
from handnet.evaluate import as_frame, score_image, summarise
from handnet.loss import hand_loss
from handnet.model import HandNet, complexity
from motion.hands import decode, letterbox

RUNS = Path(__file__).resolve().parent.parent / "runs"

DEFAULTS: dict[str, object] = {
    "width": 1.0,
    "height": 192,
    "input_width": 256,
    "epochs": 60,
    "batch": 64,
    "lr": 2e-3,
    "weight_decay": 5e-4,
    "warmup_epochs": 1,
    "ema": 0.999,
    "augment": sorted(ALL_AUGMENT),
    "fraction": 1.0,
    "seed": 0,
    "eval_every": 5,
    "val_images": 1200,
    "workers": 8,
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
}


def parse_value(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


@torch.no_grad()
def evaluate(model: torch.nn.Module, rows: list[dict], cfg: dict, device: str) -> dict:
    """Val metrics through the demo's path (320 px JPEG frames, letterbox, motion.hands.decode)."""
    model.eval()
    per_image, batch, meta = [], [], []

    def flush() -> None:
        x = normalise(torch.from_numpy(np.stack(batch)).to(device))
        raw = model(x).float().cpu().numpy()
        for out, (ratio, bounds, gts) in zip(raw, meta, strict=True):
            found = decode(out[None], ratio, bounds, threshold=0.05)
            per_image.append(score_image(found, gts))
        batch.clear()
        meta.clear()

    from handnet.data import read

    for row in rows:
        image, boxes = read(row)
        frame, gts = as_frame(image, boxes)
        canvas, ratio = letterbox(frame, cfg["height"], cfg["input_width"])
        batch.append(np.ascontiguousarray(canvas[..., ::-1].transpose(2, 0, 1)))
        meta.append((ratio, (frame.shape[1], frame.shape[0]), gts))
        if len(batch) == 64:
            flush()
    if batch:
        flush()
    model.train()
    return summarise(per_image, rows, threshold=0.5, ci=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument(
        "--overfit",
        type=int,
        default=0,
        help="train and score on this many images, no augmentation",
    )
    ap.add_argument("--resume", action="store_true", help="continue runs/<name>/last.pt")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    for item in args.set:
        key, value = item.split("=", 1)
        if key not in cfg:
            raise SystemExit(f"unknown setting {key}")
        cfg[key] = parse_value(value)
    if args.overfit:
        cfg.update(
            augment=[],
            epochs=cfg["epochs"] if "epochs=" in " ".join(args.set) else 150,
            eval_every=25,
            warmup_epochs=0,
        )

    run = RUNS / args.name
    resumed = torch.load(run / "last.pt", map_location="cpu") if args.resume else None
    if resumed is not None:
        cfg = resumed["config"]  # the run's own settings, not today's defaults or --set
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = str(cfg["device"])
    run.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows("train", float(cfg["fraction"]))
    val_rows = load_rows("val")[:: max(1, len(load_rows("val")) // int(cfg["val_images"]))]
    if args.overfit:
        train_rows = train_rows[: args.overfit]
        val_rows = train_rows

    model = HandNet(float(cfg["width"]))
    params, macs = complexity(model, int(cfg["height"]), int(cfg["input_width"]))
    cfg.update(
        params=params, macs=macs, train_images=len(train_rows), val_images_used=len(val_rows)
    )
    if resumed is None:
        (run / "config.json").write_text(json.dumps(cfg, indent=2))
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    )
    (run / "env.json").write_text(
        json.dumps(
            {
                "git": sha,
                "dirty": dirty,
                "torch": torch.__version__,
                "python": platform.python_version(),
                "machine": platform.machine(),
                "device": device,
                "resumed_at_epoch": resumed["epoch"] if resumed else None,
            },
            indent=2,
        )
    )

    model = model.to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    decay, no_decay = [], []
    for _name, p in model.named_parameters():
        (no_decay if p.ndim == 1 else decay).append(p)  # no weight decay on BN and biases
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(cfg["lr"]),
    )

    data = HandData(
        train_rows,
        int(cfg["height"]),
        int(cfg["input_width"]),
        frozenset(cfg["augment"]),
        int(cfg["seed"]),
    )
    steps_per_epoch = math.ceil(len(data) / int(cfg["batch"]))
    total = steps_per_epoch * int(cfg["epochs"])
    warmup = steps_per_epoch * float(cfg["warmup_epochs"])

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    best, start, step = -1.0, 0, 0
    if resumed is not None:
        model.load_state_dict(resumed["model"])
        ema.load_state_dict(resumed["ema"])
        if "opt" in resumed:  # older checkpoints lack it; AdamW then re-learns its moments
            opt.load_state_dict(resumed["opt"])
        start = int(resumed["epoch"])
        step = start * steps_per_epoch
        evals = [json.loads(line) for line in (run / "metrics.jsonl").open()]
        evals = [r for r in evals if "val" in r and r["epoch"] <= start]
        best = max((r["val"]["ap50"] for r in evals), default=-1.0)
        print(f"resuming {args.name} after epoch {start}, best val AP50 so far {best:.3f}")
    log = (run / "metrics.jsonl").open("a")
    for epoch in range(start, int(cfg["epochs"])):
        data.epoch = epoch
        loader = DataLoader(
            data,
            batch_size=int(cfg["batch"]),
            shuffle=True,
            drop_last=len(data) > int(cfg["batch"]),
            num_workers=int(cfg["workers"]),
            generator=torch.Generator().manual_seed(int(cfg["seed"]) + epoch),
        )
        sums: dict[str, float] = {}
        seen, t0 = 0, time.perf_counter()
        for image, heat, reg, mask in loader:
            out = model(normalise(image.to(device)))
            losses = hand_loss(out, heat.to(device), reg.to(device), mask.to(device))
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            for group in opt.param_groups:  # schedule by global step, so resuming lines up
                group["lr"] = float(cfg["lr"]) * lr_at(step)
            opt.step()
            step += 1
            d = min(
                float(cfg["ema"]), (1 + step) / (10 + step)
            )  # EMA warms up over the first steps
            with torch.no_grad():
                for e, m in zip(
                    ema.state_dict().values(), model.state_dict().values(), strict=True
                ):
                    if e.dtype.is_floating_point:
                        e.mul_(d).add_(m.detach(), alpha=1 - d)
                    else:
                        e.copy_(m)
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach()) * len(image)
            seen += len(image)
        record: dict[str, object] = {
            "epoch": epoch + 1,
            "lr": opt.param_groups[0]["lr"],
            "img_per_s": round(seen / (time.perf_counter() - t0)),
            **{k: round(v / seen, 4) for k, v in sums.items()},
        }
        last = epoch + 1 == int(cfg["epochs"])
        if (epoch + 1) % int(cfg["eval_every"]) == 0 or last:
            val = evaluate(ema, val_rows, cfg, device)
            record["val"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in val.items()}
            if val["ap50"] > best:
                best = val["ap50"]
                torch.save(
                    {"ema": ema.state_dict(), "config": cfg, "epoch": epoch + 1}, run / "best.pt"
                )
        torch.save(
            {
                "ema": ema.state_dict(),
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "config": cfg,
                "epoch": epoch + 1,
            },
            run / "last.pt",
        )
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
