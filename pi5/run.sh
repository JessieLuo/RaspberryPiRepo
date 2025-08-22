#!/bin/bash

# download test video
#wget -nc https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/releases/download/v.2.0/test.avi
# run the tracking script
python3 track_yolo11_bytetrack.py \
  --source ./test.avi \
  --device cpu \
  --imgsz 384 \
  --conf 0.25 \
  --fps 30 \
  --diag