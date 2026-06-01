"""Qualitative detection figures: predictions vs ground truth on val images.

Saves overlay figures (green = ground truth, red = prediction w/ score) for a
mix of random val images plus the deliberately hard cases: the densest scenes
(most ships) and the images dominated by the smallest ships. This is the
failure-analysis figure, not a cherry-picked highlight reel.

Usage:
    python scripts/visualize.py --weights runs/detect/yolov8s_s0/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def load_gt(lbl_path: Path, w: int, h: int) -> list:
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, bw, bh = (float(v) for v in line.split())
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h])
    return boxes


def pick_images(img_dir: Path, lbl_dir: Path, n_each: int) -> list[str]:
    """Return filenames: densest scenes, smallest-ship scenes, and random."""
    stats = []
    for img in sorted(img_dir.glob("*.bmp")):
        lines = [l for l in (lbl_dir / f"{img.stem}.txt").read_text().splitlines() if l.strip()]
        if not lines:
            continue
        areas = [float(l.split()[3]) * float(l.split()[4]) for l in lines]  # normalized area
        stats.append((img.name, len(lines), min(areas)))
    densest = [s[0] for s in sorted(stats, key=lambda s: -s[1])[:n_each]]
    smallest = [s[0] for s in sorted(stats, key=lambda s: s[2])[:n_each]]
    mid = [s[0] for s in stats[:: max(1, len(stats) // n_each)]][:n_each]
    # de-dup preserving order
    seen, picks = set(), []
    for name in densest + smallest + mid:
        if name not in seen:
            seen.add(name)
            picks.append(name)
    return picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img-dir", default="data/yolo/images/val")
    ap.add_argument("--lbl-dir", default="data/yolo/labels/val")
    ap.add_argument("--out", default=None, help="default: <weights-run>/qualitative")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--n-each", type=int, default=3)
    args = ap.parse_args()

    from ultralytics import YOLO

    img_dir, lbl_dir = Path(args.img_dir), Path(args.lbl_dir)
    out = Path(args.out) if args.out else Path(args.weights).resolve().parent.parent / "qualitative"
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    picks = pick_images(img_dir, lbl_dir, args.n_each)

    for name in picks:
        img = Image.open(img_dir / name).convert("RGB")
        w, h = img.size
        gt = load_gt(lbl_dir / f"{Path(name).stem}.txt", w, h)
        res = model.predict(source=str(img_dir / name), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        for x, y, bw, bh in gt:
            ax.add_patch(patches.Rectangle((x, y), bw, bh, fill=False, edgecolor="lime", linewidth=1.5))
        for box, score in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            x1, y1, x2, y2 = box
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=1.5))
            ax.text(x1, max(y1 - 4, 0), f"{score:.2f}", color="red", fontsize=8)
        ax.set_title(f"{name}  | GT(green)={len(gt)}  pred(red)={len(res.boxes)}  conf>={args.conf}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"{Path(name).stem}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {len(picks)} qualitative figures -> {out}")


if __name__ == "__main__":
    main()
