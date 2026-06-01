"""Train a YOLOv8 single-class ship detector on ShipRSImageNet (val = eval set).

Deterministic (seeded) training. Headline metrics (mAP@0.5, mAP@[.5:.95]) are
reported by ultralytics on the official val split; run scripts/eval_coco.py
afterward for COCO area-stratified AP (small/medium/large).

Usage:
    python scripts/train_yolo.py --model yolov8s.pt --epochs 100 --seed 0
    python scripts/train_yolo.py --model yolov8m.pt --imgsz 1024 --batch 12 --name yolov8m_s0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/yolo/shiprs.yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="pretrained checkpoint to fine-tune")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024, help="ShipRSImageNet images are ~930px; ships are small")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20, help="early-stopping patience")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default=None, help="run name (default: <model-stem>_s<seed>)")
    ap.add_argument("--project", default="runs/detect", help="output dir (resolved to absolute)")
    args = ap.parse_args()

    name = args.name or f"{Path(args.model).stem}_s{args.seed}"
    # Resolve to absolute so ultralytics doesn't nest project under its own
    # runs_dir setting (which produces runs/detect/runs/detect/<name>).
    project = str(Path(args.project).resolve())

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        device=args.device,
        project=project,
        name=name,
        exist_ok=False,
        val=True,
        plots=True,
    )

    # results.box carries the final validation metrics on the val split.
    box = results.box
    save_dir = results.save_dir
    print("\n=== Final val metrics ===")
    print(f"run dir      : {save_dir}")
    print(f"mAP@0.5      : {box.map50:.4f}")
    print(f"mAP@[.5:.95] : {box.map:.4f}")
    print(f"precision    : {box.mp:.4f}")
    print(f"recall       : {box.mr:.4f}")
    print(f"best weights : {Path(save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
