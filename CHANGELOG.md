# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses
[SemVer](https://semver.org/) (pre-1.0: minor = breaking allowed).

## [Unreleased]

## [0.1.0] - 2026-06-05

Research-grade release. Ship detection and localization in optical satellite
imagery, plus the preserved Phase 1 chip-classification work.

### Added (Phase 2, detection)
- Single-class ship detection on ShipRSImageNet: YOLOv8s (headline, 3 seeds) and
  Faster-RCNN-v2 (comparison), scored by the same `pycocotools` evaluator.
- Honest protocol: trained on the official `train` split, evaluated on the
  official `val` split (550 images, 2,949 instances); withheld `test` used only
  for qualitative inspection.
- `scripts/prepare_shiprs.py`: extract, filter to the ship class, convert COCO to
  YOLO format, and validate every box (0 clipped / 0 degenerate at build time).
- COCO area-stratified AP and qualitative GT-vs-pred figures.

### Findings
- YOLOv8s reaches 0.634 AP@[.5:.95] / 0.799 AP@0.5 (3-seed mean), ahead of
  Faster-RCNN-v2.
- Both detectors degrade sharply on small ships (AP about 0.15 to 0.20 vs about
  0.84 for large), the dominant failure mode.

### Preserved (Phase 1, chip classification)
- The original binary ship-vs-no-ship course project, with a leakage-aware
  scene-held-out split, kept in `Ship_Detection_Project_2.ipynb`.
