#!/usr/bin/env bash
# Full ShipRSImageNet detection experiment suite (reproducible driver).
#   - YOLOv8s headline detector across 3 seeds (mean +/- std)
#   - COCO eval (incl. area-stratified AP) for every YOLO seed
#   - Faster-RCNN-v2 comparison detector (1 seed), scored by the same evaluator
#   - Qualitative failure-analysis figures
# Run from anywhere: bash scripts/run_ship_experiments.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
mkdir -p runs/coco_eval

echo "=== [1/4] YOLOv8s seeds 1,2 (seed 0 already trained) ==="
for s in 1 2; do
  "$PY" scripts/train_yolo.py --model yolov8s.pt --epochs 100 --imgsz 1024 --batch 16 --seed "$s" --name "yolov8s_s$s"
done

echo "=== [2/4] COCO eval, all YOLO seeds ==="
for s in 0 1 2; do
  "$PY" scripts/eval_coco.py --weights "runs/detect/yolov8s_s$s/weights/best.pt" > "runs/coco_eval/yolov8s_s$s.txt" 2>&1
  echo "--- yolov8s seed $s ---"; grep "Average Precision" "runs/coco_eval/yolov8s_s$s.txt"
done

echo "=== [3/4] Faster-RCNN-v2 seed 0 (26 epochs) ==="
"$PY" scripts/train_frcnn.py --epochs 26 --seed 0 --out runs/frcnn > "runs/coco_eval/frcnn_s0.txt" 2>&1
echo "--- faster-rcnn seed 0 ---"; grep "Average Precision" "runs/coco_eval/frcnn_s0.txt"

echo "=== [4/4] Qualitative figures (yolov8s seed 0) ==="
"$PY" scripts/visualize.py --weights runs/detect/yolov8s_s0/weights/best.pt

echo "ALL_SHIP_EXPERIMENTS_DONE"
