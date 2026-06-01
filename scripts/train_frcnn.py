"""Train a Faster-RCNN (ResNet50-FPN) single-class ship detector on ShipRSImageNet.

Secondary detector for the head-to-head comparison against YOLOv8. Reads the YOLO
layout produced by prepare_shiprs.py, trains a torchvision Faster-RCNN, then runs
inference on the val split and writes detections in COCO format so eval_coco.py
scores it with the IDENTICAL ground truth + evaluator used for YOLO.

min_size is raised to 1024 because ShipRSImageNet ships are small; the default 800
loses the smallest targets (a trap Codex flagged for Faster-RCNN on this data).

Usage:
    python scripts/train_frcnn.py --epochs 26 --seed 0
    python scripts/eval_coco.py --dets runs/frcnn/frcnn_s0_dets_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the sibling eval_coco importable when run as `python scripts/train_frcnn.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T

from eval_coco import SHIP_CAT_ID, build_ship_gt, run_cocoeval

NUM_CLASSES = 2  # background + ship


class YoloShipDataset(Dataset):
    """Reads prepare_shiprs.py's YOLO layout and yields torchvision detection targets."""

    def __init__(self, root: Path, split: str, train: bool):
        self.img_dir = root / "images" / split
        self.lbl_dir = root / "labels" / split
        self.files = sorted(p.name for p in self.img_dir.glob("*.bmp"))
        tfms = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
        if train:
            tfms.insert(0, T.RandomHorizontalFlip(0.5))
        self.tfms = T.Compose(tfms)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fn = self.files[idx]
        img = Image.open(self.img_dir / fn).convert("RGB")
        w, h = img.size
        boxes, labels = [], []
        lbl = self.lbl_dir / f"{Path(fn).stem}.txt"
        for line in lbl.read_text().splitlines():
            if not line.strip():
                continue
            _, cx, cy, bw, bh = (float(v) for v in line.split())
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            boxes.append([x1, y1, x2, y2])
            labels.append(SHIP_CAT_ID)
        boxes = tv_tensors.BoundingBoxes(
            boxes if boxes else torch.zeros((0, 4)),
            format="XYXY",
            canvas_size=(h, w),
        )
        target = {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.int64)}
        img, target = self.tfms(img, target)
        return img, target, fn


def collate(batch):
    return tuple(zip(*batch))


def build_model(min_size: int) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", min_size=min_size, max_size=1333)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/yolo")
    ap.add_argument("--zip", default="data/ShipRSImageNet_V1.zip")
    ap.add_argument("--epochs", type=int, default=26)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--min-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/frcnn")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = YoloShipDataset(root, "train", train=True)
    val_ds = YoloShipDataset(root, "val", train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate)

    model = build_model(args.min_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[int(args.epochs * 0.7), int(args.epochs * 0.9)], gamma=0.1)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for imgs, targets, _ in train_dl:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item()
        sched.step()
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={running / len(train_dl):.4f}", flush=True)

    # Inference on val -> COCO detections, mapped to the shared ship-GT image ids.
    gt_json = root / "val_ship_coco.json"
    gt = build_ship_gt(args.zip, gt_json)
    name_to_id = {im["file_name"]: im["id"] for im in gt["images"]}

    model.eval()
    dets = []
    with torch.no_grad():
        for imgs, _, fns in val_dl:
            img = imgs[0].to(device)
            out_pred = model([img])[0]
            image_id = name_to_id[fns[0]]
            for box, score, label in zip(out_pred["boxes"].tolist(), out_pred["scores"].tolist(), out_pred["labels"].tolist()):
                if label != SHIP_CAT_ID:
                    continue
                x1, y1, x2, y2 = box
                dets.append({"image_id": image_id, "category_id": SHIP_CAT_ID,
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})

    dets_path = out / f"frcnn_s{args.seed}_dets_val.json"
    dets_path.write_text(json.dumps(dets))
    torch.save(model.state_dict(), out / f"frcnn_s{args.seed}.pt")
    print(f"\nWrote {len(dets)} detections -> {dets_path}")
    print("=== Faster-RCNN COCO eval on val ===")
    run_cocoeval(gt_json, dets)


if __name__ == "__main__":
    main()
