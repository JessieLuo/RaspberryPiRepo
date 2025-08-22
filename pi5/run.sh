#!/bin/bash

# download test video
#wget -nc https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/releases/download/v.2.0/test.avi
# download model
# wget https://github.com/ppogg/YOLOv5-Lite/releases/download/1.0/yolov5n-lite.onnx
# run the tracking script
python3 yolo_bytertrack.py \
  --source ./test.avi \
  --model v5lite-e-sim.onnx \
  --imgsz 320 \
  --conf 0.15 \
  --fps 30 \
  --out ./results/test_result.mp4 \
  --diag