#!/bin/bash

# download test video
#wget -nc https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/releases/download/v.2.0/test.avi
# run the tracking script
python3 yolo_bytertrack.py \
  --source ./test.avi \
  --device cpu \
  --model yolov8n.pt \
  --imgsz 384 \
  --conf 0.25 \
  --fps 30 \
  --diag