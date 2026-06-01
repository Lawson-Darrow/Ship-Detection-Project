"""COCO-style evaluation for single-class ship detection on the val split.

Reports the full COCO metric suite including area-stratified AP
(AP_small / AP_medium / AP_large), which the headline ultralytics val() does not
break out. Builds a ship-only ground-truth COCO file from the dataset zip once,
then scores either a YOLO checkpoint or a precomputed COCO detections JSON. The
same ground truth + evaluator are reused across detectors (YOLO, Faster-RCNN) so
the comparison is strictly apples-to-apples.

Usage:
    python scripts/eval_coco.py --weights runs/detect/yolov8s_s0/weights/best.pt
    python scripts/eval_coco.py --dets runs/frcnn/dets_val.json   # precomputed
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

ZIP_VAL_COCO = "ShipRSImageNet_V1/COCO_Format/ShipRSImageNet_bbox_val_level_0.json"
SHIP_CLASS_NAME = "Ship"
SHIP_CAT_ID = 1  # single class, remapped


def build_ship_gt(zip_path: str, out_json: Path) -> dict:
    """Build (and cache) a ship-only single-class val ground truth in COCO format."""
    with zipfile.ZipFile(zip_path) as zf, zf.open(ZIP_VAL_COCO) as f:
        coco = json.load(f)
    ship_src_id = next(c["id"] for c in coco["categories"] if c["name"] == SHIP_CLASS_NAME)

    gt = {
        "images": [
            {"id": im["id"], "file_name": im["file_name"], "width": im["width"], "height": im["height"]}
            for im in coco["images"]
        ],
        "categories": [{"id": SHIP_CAT_ID, "name": "ship", "supercategory": "ship"}],
        "annotations": [],
    }
    next_id = 1
    for ann in coco["annotations"]:
        if ann["category_id"] != ship_src_id:
            continue
        x, y, w, h = ann["bbox"]
        gt["annotations"].append(
            {
                "id": next_id,
                "image_id": ann["image_id"],
                "category_id": SHIP_CAT_ID,
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": ann.get("iscrowd", 0),
            }
        )
        next_id += 1
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(gt))
    return gt


def yolo_detections(weights: str, gt: dict, img_dir: Path, imgsz: int) -> list:
    """Run a YOLO checkpoint over the val images, return COCO-format detections.

    Uses a very low conf threshold so the precision-recall curve is fully sampled
    (COCO AP integrates over recall; high conf thresholds truncate it).
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    name_to_id = {im["file_name"]: im["id"] for im in gt["images"]}
    dets = []
    for im in gt["images"]:
        path = img_dir / im["file_name"]
        res = model.predict(source=str(path), imgsz=imgsz, conf=0.001, iou=0.7, max_det=300, verbose=False)[0]
        image_id = name_to_id[im["file_name"]]
        for box, score in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            x1, y1, x2, y2 = box
            dets.append(
                {
                    "image_id": image_id,
                    "category_id": SHIP_CAT_ID,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return dets


def run_cocoeval(gt_json: Path, dets: list) -> None:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default="data/ShipRSImageNet_V1.zip")
    ap.add_argument("--img-dir", default="data/yolo/images/val")
    ap.add_argument("--gt-json", default="data/yolo/val_ship_coco.json")
    ap.add_argument("--weights", default=None, help="YOLO checkpoint to evaluate")
    ap.add_argument("--dets", default=None, help="precomputed COCO detections JSON (e.g. Faster-RCNN)")
    ap.add_argument("--imgsz", type=int, default=1024)
    args = ap.parse_args()

    gt_json = Path(args.gt_json)
    gt = build_ship_gt(args.zip, gt_json)
    print(f"Built ship-only val GT: {len(gt['images'])} images, {len(gt['annotations'])} ship boxes")

    if args.dets:
        dets = json.loads(Path(args.dets).read_text())
    elif args.weights:
        dets = yolo_detections(args.weights, gt, Path(args.img_dir), args.imgsz)
    else:
        raise SystemExit("Provide --weights (YOLO) or --dets (precomputed COCO detections)")

    print(f"Detections: {len(dets)}")
    run_cocoeval(gt_json, dets)


if __name__ == "__main__":
    main()
