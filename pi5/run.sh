#!/bin/bash

# download test video
#wget -nc https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/releases/download/v.2.0/test.avi
# run the tracking script
python track_yolo11_bytetrack.py\
  --source ./test.avi \
  --out ./results/test_result.mp4