# Ship Detection & Localization in Satellite Imagery

Object detection of ships in high-resolution optical satellite imagery: predicting
**bounding boxes** (localization) for every ship in an image and evaluating with the
standard COCO detection metrics (mAP, IoU, area-stratified AP). Two detectors are
trained and compared on a common, leakage-controlled protocol:

- **YOLOv8** (single-stage) — headline detector, reported across 3 seeds.
- **Faster-RCNN-v2** (two-stage) — comparison detector, scored by the identical evaluator.

> This repository began as a binary *chip-classification* course project (ship vs.
> no-ship on 80x80 tiles). That work is preserved as **Phase 1** below. **Phase 2**
> (this top section) is the real detection/localization task: boxes, mAP, and IoU on
> a dedicated detection dataset.

## TL;DR results

Single-class ("ship") detection on **ShipRSImageNet**, evaluated on the official
validation split (550 images, 2,949 ship instances) with `pycocotools`:

| Detector | AP@[.5:.95] | AP@0.5 | AP@0.75 | AP small | AP medium | AP large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **YOLOv8s** (3-seed mean ± std) | **0.634 ± 0.004** | 0.799 ± 0.002 | 0.702 ± 0.006 | 0.203 ± 0.004 | 0.662 ± 0.005 | 0.840 ± 0.006 |
| Faster-RCNN-v2 (1 seed) | 0.580 | 0.735 | 0.670 | 0.152 | 0.597 | 0.784 |

Takeaways: YOLOv8s outperforms Faster-RCNN on this task, the headline result is
stable across seeds (std ≈ 0.004), and **both detectors degrade sharply on small
ships** (AP ≈ 0.15–0.20 vs ≈ 0.84 for large) — the dominant failure mode and the
clearest direction for future work.

## Phase 2 — Object Detection

### Dataset

[ShipRSImageNet](https://github.com/zzndream/ShipRSImageNet) V1 — 3,435 high-resolution
optical remote-sensing images (~930x930), 17,573 annotated ship instances. This project
uses the **single "ship" class** (the dataset's level-0 `Ship` category; the `Dock`
category and the fine-grained 50-class hierarchy are out of scope and treated as
background).

**Evaluation protocol (honest + benchmark-comparable):** ShipRSImageNet's official
`test` split ships images only (labels withheld for benchmarking). Following standard
practice on this dataset, models are **trained on the official `train` split (2,198
images)** and **evaluated on the official `val` split (550 images)**. The withheld
`test` images are used only for qualitative inspection.

Conversion is handled by `scripts/prepare_shiprs.py`, which extracts the dataset,
filters to the ship class, converts COCO boxes to YOLO format, validates every box
(0 clipped / 0 degenerate at build time), and emits area-stratified instance counts.

### Models & training

- **YOLOv8s**, fine-tuned from COCO-pretrained weights, `imgsz=1024` (ShipRSImageNet
  ships are small), 100 epochs with early stopping, deterministic, seeds {0, 1, 2}.
- **Faster-RCNN-v2** (ResNet50-FPN), `min_size=1024` so the smallest ships survive the
  input resize, 26 epochs.

Both detectors are scored with the same `pycocotools` evaluator against the same
ship-only validation ground truth, so the comparison is strictly apples-to-apples.

### Results

Headline detector across three seeds (ultralytics validation metrics):

| Seed | mAP@0.5 | mAP@[.5:.95] | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.869 | 0.670 | 0.874 | 0.828 |
| 1 | 0.849 | 0.663 | 0.871 | 0.812 |
| 2 | 0.867 | 0.670 | 0.878 | 0.817 |
| **mean ± std** | **0.862 ± 0.011** | **0.668 ± 0.004** | 0.874 ± 0.003 | 0.819 ± 0.008 |

COCO area-stratified AP and the YOLOv8s-vs-Faster-RCNN comparison are in the TL;DR
table above. Per-run COCO summaries are saved under `runs/coco_eval/`.

### Failure analysis

The area-stratified AP exposes a consistent weakness: small ships (COCO area < 32²)
score AP ≈ 0.20 for YOLOv8s and ≈ 0.15 for Faster-RCNN, versus ≈ 0.84 / 0.78 for large
ships. Recall on small ships is similarly low (AR ≈ 0.30). The qualitative figures
(`runs/detect/yolov8s_s0/qualitative/`) deliberately include the densest scenes and the
smallest-ship images: most misses are tiny vessels in cluttered ports and faint wakes,
not large clearly-imaged ships.

### Reproduce

```bash
# 1. Download ShipRSImageNet_V1.zip into data/ (HF mirror: insomnia7/ShipRSImageNet)
# 2. Build the YOLO dataset (extract + convert + validate)
python scripts/prepare_shiprs.py

# 3. Run the full experiment suite (YOLOv8 x3 seeds, COCO eval, Faster-RCNN, figures)
bash scripts/run_ship_experiments.sh

# Or individual steps:
python scripts/train_yolo.py --model yolov8s.pt --epochs 100 --imgsz 1024 --seed 0
python scripts/eval_coco.py  --weights runs/detect/yolov8s_s0/weights/best.pt
python scripts/train_frcnn.py --epochs 26 --seed 0
python scripts/visualize.py  --weights runs/detect/yolov8s_s0/weights/best.pt
```

Environment: Python 3.12, PyTorch 2.6 (CUDA), `ultralytics`, `pycocotools`. Trained on
a single RTX 4090.

## Phase 1 — Chip Classification (original course project)

The original work is a binary **ship vs. no-ship** classifier on 80x80 image chips from
the Kaggle *Ships in Satellite Imagery* dataset (`shipsnet.json`), comparing hand-crafted
features (RGB/HSV statistics, edges, texture, shape) under classical models against a
ResNet18 transfer-learning CNN on raw pixels.

A notable strength of this phase is its **leakage-aware evaluation**: alongside a
stratified chip-level split it implements a stricter **scene-held-out** (`GroupShuffleSplit`)
protocol so no source scene appears in both train and test. The full analysis is in the
notebook:

- `Ship_Detection_Project_2.ipynb` (Colab-ready)

On the scene-held-out split, the ResNet18 transfer-learning CNN reached F1 ≈ 0.998 /
ROC-AUC ≈ 1.0; the tuned RBF SVM reached F1 ≈ 0.95. The chip dataset is close to solved,
which is part of the motivation for moving to the harder detection task in Phase 2.

## Repository structure

```text
.
├── scripts/                      # Phase 2 detection pipeline (runnable, reproducible)
│   ├── prepare_shiprs.py         #   ShipRSImageNet -> validated YOLO dataset
│   ├── train_yolo.py             #   YOLOv8 training (seeded, deterministic)
│   ├── eval_coco.py              #   COCO eval incl. area-stratified AP
│   ├── train_frcnn.py            #   Faster-RCNN-v2 comparison detector
│   ├── visualize.py              #   qualitative GT-vs-pred figures
│   └── run_ship_experiments.sh   #   full suite driver
├── Ship_Detection_Project_2.ipynb  # Phase 1 chip-classification analysis
├── data/                         # datasets (gitignored)
└── runs/                         # training + eval outputs (gitignored)
```

## Notes

- Phase 1 originated as an individual course project for MTH/CSE 4224.
- ShipRSImageNet images and annotations are for academic use; see the dataset's terms.
- Trained model weights and raw datasets are not committed (see `.gitignore`).
