# Contributing

This is a research and education project. Issues, ideas, and PRs are welcome,
especially around the small-ship failure mode, additional detectors, and the
evaluation protocol.

## Dev setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install ultralytics pycocotools torch torchvision   # Python 3.12, CUDA build of torch
```

Then download `ShipRSImageNet_V1.zip` into `data/` and build the YOLO dataset:

```bash
python scripts/prepare_shiprs.py
bash scripts/run_ship_experiments.sh   # YOLOv8 x3 seeds, COCO eval, Faster-RCNN, figures
```

A single RTX 4090 (or any CUDA GPU) is assumed; PyTorch 2.6.

## Bar for changes

- **Keep the protocol honest.** Train on the official `train` split, evaluate on
  the official `val` split; the withheld `test` images are for qualitative
  inspection only. Do not score on the withheld test.
- **Same evaluator for both detectors.** YOLOv8 and Faster-RCNN are scored by the
  identical `pycocotools` evaluator against the same ship-only ground truth, so
  the comparison stays apples to apples.
- **Report area-stratified AP.** Small-ship AP is the headline failure mode; keep
  the area breakdown, not just the aggregate mAP.
- **Keep the seeds.** The headline detector is reported as mean over 3 seeds.
- Build-time box validation (0 clipped / 0 degenerate) should stay green after
  changes to data prep.

## Scope

Phase 1 (chip classification) is preserved in the notebook on purpose; leave it
intact. New detectors and a stronger small-object approach are welcome.
