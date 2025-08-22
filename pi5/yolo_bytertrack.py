#!/usr/bin/env python3
"""
YOLOv5-Lite + ByteTrack (minimal) — CPU-friendly Pi 5 demo
- Detector: YOLOv5-Lite exported to ONNX, run with OpenCV DNN (no extra deps)
- Tracker: ByteTrack-style two-stage association with a tiny Kalman filter
- Sources: video file/URL (e.g., RTSP) or a directory of frames
- Output: MP4 with boxes + ids; optional per-frame diagnostics
Notes:
  * Export your yolov5-lite model to ONNX first (dynamic axes recommended).
    Example (from yolov5-lite repo or your training env):
      python export.py --include onnx --weights yolov5n-lite.pt --img 384 384
    Then run this script with: --model yolov5n-lite.onnx
"""
import argparse, os, sys, cv2, math, time
from pathlib import Path
import numpy as np
from typing import List, Tuple

#  Drawing ----
def draw_text(img, text, pos, font=cv2.FONT_HERSHEY_PLAIN,
              font_scale=1, font_thickness=1,
              text_color=(255,255,255), bg_color=(0,0,255)):
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    tw, th = max(tw + 16, 50), max(th + 10, 20)
    cv2.rectangle(img, (x, y), (x + tw, y + th), bg_color, -1)
    cv2.putText(img, text, (x + 8, y + th - 6), font, font_scale, text_color, font_thickness)

#  Geometry utils ----
def xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    out = np.zeros_like(xywh)
    out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return out

def xyxy2xywh(xyxy: np.ndarray) -> np.ndarray:
    out = np.zeros_like(xyxy)
    out[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
    out[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
    out[:, 2] = np.clip(xyxy[:, 2] - xyxy[:, 0], 1e-3, None)
    out[:, 3] = np.clip(xyxy[:, 3] - xyxy[:, 1], 1e-3, None)
    return out

def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: Nx4, b: Mx4 (xyxy)
    N, M = a.shape[0], b.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / np.clip(union, 1e-6, None)
    return iou.astype(np.float32)

#  Minimal Kalman (xywh, vx, vy) ---
class KalmanXYWH:
    def __init__(self, xywh: np.ndarray):
        # state: [x, y, w, h, vx, vy]
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.x[0:4, 0] = xywh.astype(np.float32)
        self.P = np.eye(6, dtype=np.float32) * 10.0
        self.F = np.eye(6, dtype=np.float32)
        self.H = np.zeros((4, 6), dtype=np.float32)
        self.Q = np.eye(6, dtype=np.float32) * 1e-2
        self.R = np.eye(4, dtype=np.float32) * 1.0
        for i in range(4):
            self.H[i, i] = 1.0

    def predict(self, dt: float = 1.0):
        self.F[0, 4] = dt
        self.F[1, 5] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z_xywh: np.ndarray):
        z = z_xywh.reshape(4, 1).astype(np.float32)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def mean_xywh(self) -> np.ndarray:
        return self.x[0:4, 0].copy()

#  ByteTrack-lite ----
class Track:
    _next_id = 0
    def __init__(self, xywh: np.ndarray, score: float, ttl: int = 30):
        self.id = Track._next_id
        Track._next_id += 1
        self.kf = KalmanXYWH(xywh)
        self.score = float(score)
        self.ttl = ttl  # frames to keep when unmatched
        self.time_since_update = 0
        self.hits = 1

    def predict(self, dt=1.0):
        self.kf.predict(dt)
        self.time_since_update += 1

    def update(self, xywh: np.ndarray, score: float):
        self.kf.update(xywh)
        self.score = float(score)
        self.time_since_update = 0
        self.hits += 1

    def xyxy(self) -> np.ndarray:
        return xywh2xyxy(self.kf.mean_xywh().reshape(1,4))[0]

class ByteTrackLite:
    """
    Two-stage association like ByteTrack:
      1) Match high-score detections with existing tracks (IOU >= iou_high)
      2) Then allow remaining low-score detections to (re)activate tracks (IOU >= iou_low)
    No appearance features; constant-velocity Kalman; CPU-only.
    """
    def __init__(self, iou_high=0.3, iou_low=0.2, conf_high=0.5, conf_low=0.1, max_ttl=30):
        self.iou_high = iou_high
        self.iou_low = iou_low
        self.conf_high = conf_high
        self.conf_low = conf_low
        self.max_ttl = max_ttl
        self.tracks: List[Track] = []

    def step(self, det_xyxy: np.ndarray, det_conf: np.ndarray, dt=1.0) -> List[Track]:
        # Predict
        for t in self.tracks:
            t.predict(dt)

        # Split dets by conf
        hi_mask = det_conf >= self.conf_high
        lo_mask = (det_conf >= self.conf_low) & (~hi_mask)

        assigned = set()
        # --- First stage: high-conf match ---
        self._associate(det_xyxy[hi_mask], det_conf[hi_mask], assigned, self.iou_high)

        # --- Second stage: low-conf match (rescue) ---
        self._associate(det_xyxy[lo_mask], det_conf[lo_mask], assigned, self.iou_low)

        # Age & prune
        alive = []
        for t in self.tracks:
            if t.time_since_update <= self.max_ttl:
                alive.append(t)
        self.tracks = alive
        return self.tracks

    def _associate(self, det_xyxy: np.ndarray, det_conf: np.ndarray, assigned: set, thr: float):
        # Build candidate tracks set (unassigned)
        unassigned_tracks = [t for t in self.tracks if t.time_since_update > 0] + [t for t in self.tracks if t.time_since_update == 0]
        if det_xyxy.shape[0] == 0:
            return
        track_boxes = np.array([t.xyxy() for t in self.tracks], dtype=np.float32) if self.tracks else np.zeros((0,4), dtype=np.float32)
        if track_boxes.shape[0] == 0:
            # Initialize new tracks from detections
            for d, s in zip(det_xyxy, det_conf):
                xywh = xyxy2xywh(d.reshape(1,4))[0]
                self.tracks.append(Track(xywh, float(s), ttl=self.max_ttl))
            return

        # IOU between tracks and detections
        M = iou(track_boxes, det_xyxy)
        # Greedy matching
        used_tracks = set()
        used_dets = set()
        while True:
            idx = np.unravel_index(np.argmax(M), M.shape)
            i, j = int(idx[0]), int(idx[1])
            if M[i, j] < thr:
                break
            if i in used_tracks or j in used_dets:
                M[i, j] = -1
                continue
            # assign
            d = det_xyxy[j]
            s = det_conf[j]
            xywh = xyxy2xywh(d.reshape(1,4))[0]
            self.tracks[i].update(xywh, float(s))
            used_tracks.add(i)
            used_dets.add(j)
            M[i, :] = -1
            M[:, j] = -1

        # Spawn new tracks from unassigned detections
        for j in range(det_xyxy.shape[0]):
            if j in used_dets:
                continue
            d = det_xyxy[j]
            s = det_conf[j]
            xywh = xyxy2xywh(d.reshape(1,4))[0]
            self.tracks.append(Track(xywh, float(s), ttl=self.max_ttl))

#  YOLOv5-Lite ONNX detector 
class YOLOv5LiteONNX:
    def __init__(self, onnx_path: str, imgsz: int = 384, conf_thresh: float = 0.1):
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        # CPU only — make sure OpenCV uses CPU backend (default). No CUDA on Pi.
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.nms_thresh = 0.45
        self.nc = None  # will infer from outputs
        self.class_names = None  # optional; can be set from file if needed

    def set_limits(self, min_box:int=0, max_det:int=0):
        self._min_box = int(min_box)
        self._max_det = int(max_det)
    
    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          boxes_xyxy: (N,4) float32 in absolute pixel coords
          conf:       (N,)   float32 object conf
          cls:        (N,)   int32  class id
        """
        h, w = frame.shape[:2]
        inp = self._preprocess(frame)
        self.net.setInput(inp)
        out = self.net.forward()
        # YOLOv5* ONNX often returns (1, N, 85); squeeze
        if out.ndim == 3:
            out = out[0]
        # out: N x (5 + nc) => [cx, cy, w, h, obj, cls...]
        if self.nc is None:
            self.nc = out.shape[1] - 5

        # --- Robust postprocessing for YOLOv5-Lite ONNX ---
        # Split raw outputs
        boxes_xywh = out[:, :4].astype(np.float32)
        obj = out[:, 4].astype(np.float32)
        cls_scores = out[:, 5:].astype(np.float32)

        # If exporter left logits (values outside [0,1]), apply sigmoid to obj and class scores
        def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
        if np.nanmax(obj) > 1.0 or np.nanmin(obj) < 0.0 or np.nanmax(cls_scores) > 1.0 or np.nanmin(cls_scores) < 0.0:
            obj = _sigmoid(obj)
            cls_scores = _sigmoid(cls_scores)

        # Some YOLOv5-Lite exports keep boxes normalized to input; scale to network input size if so
        # Heuristic: if all coords are in [0, 1.5], treat as normalized
        if np.nanmax(boxes_xywh) <= 1.5:
            boxes_xywh[:, :4] *= float(self.imgsz)

        # Class/conf extraction and keep logic
        if cls_scores.shape[1] == 0:
            # No class dimension (unlikely); treat as single-class
            cls_ids = np.zeros((boxes_xywh.shape[0],), dtype=np.int32)
            cls_conf = np.ones_like(obj, dtype=np.float32)
        else:
            cls_ids = np.argmax(cls_scores, axis=1).astype(np.int32)
            cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_ids].astype(np.float32)

        conf = (obj * cls_conf).astype(np.float32)
        keep = conf >= self.conf_thresh
        boxes_xywh = boxes_xywh[keep]
        conf = conf[keep]
        cls_ids = cls_ids[keep]

        if boxes_xywh.shape[0] == 0:
            return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        # Scale from network input back to frame size
        boxes_xyxy = self._scale_coords(boxes_xywh, (w, h))

        # Prepare [x,y,w,h] for OpenCV NMS (expects xywh integers)
        boxes_xywh_for_nms = np.zeros_like(boxes_xyxy, dtype=np.float32)
        boxes_xywh_for_nms[:, 0] = boxes_xyxy[:, 0]
        boxes_xywh_for_nms[:, 1] = boxes_xyxy[:, 1]
        boxes_xywh_for_nms[:, 2] = np.clip(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 1.0, None)
        boxes_xywh_for_nms[:, 3] = np.clip(boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 1.0, None)

        # NMS (class-agnostic)
        idxs = cv2.dnn.NMSBoxes(boxes_xywh_for_nms.tolist(), conf.tolist(),
                        score_threshold=self.conf_thresh, nms_threshold=self.nms_thresh)
        if len(idxs) == 0:
            return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        idxs = np.array(idxs, dtype=np.int32).reshape(-1)

        boxes_xyxy = boxes_xyxy[idxs].astype(np.float32)
        conf = conf[idxs].astype(np.float32)
        cls_ids = cls_ids[idxs].astype(np.int32)

        # size filter after NMS
        min_box = int(getattr(self, "_min_box", 0) or 0)
        if min_box > 0 and boxes_xyxy.shape[0] > 0:
            w = np.clip(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 0, None)
            h = np.clip(boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 0, None)
            keep_sz = (w >= min_box) & (h >= min_box)
            boxes_xyxy = boxes_xyxy[keep_sz]
            conf = conf[keep_sz]
            cls_ids = cls_ids[keep_sz]

        # cap detections per frame
        max_det = int(getattr(self, "_max_det", 0) or 0)
        if max_det and boxes_xyxy.shape[0] > max_det:
            order = np.argsort(-conf)[:max_det]
            boxes_xyxy = boxes_xyxy[order]
            conf = conf[order]
            cls_ids = cls_ids[order]

        return boxes_xyxy, conf, cls_ids

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        # Letterbox to square imgsz
        h, w = img.shape[:2]
        scale = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        top = (self.imgsz - nh) // 2
        left = (self.imgsz - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized
        blob = cv2.dnn.blobFromImage(canvas, scalefactor=1/255.0, size=(self.imgsz, self.imgsz), swapRB=True, crop=False)
        # Save for scale back
        self._last_pad = (left, top)
        self._last_scale = scale
        self._orig_shape = (w, h)
        return blob

    def _scale_coords(self, boxes_xywh: np.ndarray, orig_wh: Tuple[int, int]) -> np.ndarray:
        # Undo letterbox and scaling
        left, top = self._last_pad
        scale = self._last_scale
        w, h = self._orig_shape
        xyxy = xywh2xyxy(boxes_xywh.copy())
        xyxy[:, [0, 2]] -= left
        xyxy[:, [1, 3]] -= top
        xyxy /= scale
        # clamp
        xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0, w - 1)
        xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0, h - 1)
        return xyxy

#  Main 
def main():
    ap = argparse.ArgumentParser("YOLOv5-Lite + ByteTrack (minimal) demo")
    ap.add_argument("--source", required=True, help="Frames directory, a video file, or a URL (RTSP/HTTP)")
    ap.add_argument("--out", default="tracking_out.mp4", help="Output video filename (mp4)")
    ap.add_argument("--model", required=True, help="Path to YOLOv5-Lite ONNX weights, e.g., yolov5n-lite.onnx")
    ap.add_argument("--conf", type=float, default=0.1, help="Detection confidence low bound (ByteTrack rescues lows)")
    ap.add_argument("--nms", type=float, default=0.45, help="NMS IoU threshold (lower = fewer overlapping boxes)")
    ap.add_argument("--max_det", type=int, default=200, help="Cap number of detections per frame after NMS")
    ap.add_argument("--min_box", type=int, default=12, help="Filter boxes smaller than this (pixels) after NMS")
    ap.add_argument("--min_hits", type=int, default=2, help="Only draw tracks with at least this many updates")
    ap.add_argument("--draw_unconfirmed", action="store_true", help="Also draw tracks that haven't been updated this frame")
    ap.add_argument("--imgsz", type=int, default=384, help="Inference image size (short side). Lower = faster")
    ap.add_argument("--fps", type=int, default=30, help="Output FPS if reading frames dir or unknown input FPS")
    ap.add_argument("--diag", action="store_true", help="Print per-frame detection/tracking counts and IDs")
    ap.add_argument("--class", dest="cls", type=int, default=None, help="Restrict to a single class id (e.g., 0 for person)")
    ap.add_argument("--iou_high", type=float, default=0.3, help="High-IOU threshold for first-stage association")
    ap.add_argument("--iou_low", type=float, default=0.2, help="Low-IOU threshold for rescue-stage association")
    ap.add_argument("--conf_high", type=float, default=0.3, help="High confidence threshold for stage-1")
    ap.add_argument("--conf_low", type=float, default=0.05, help="Low confidence threshold for stage-2")
    args = ap.parse_args()

    # Detector + Tracker
    det = YOLOv5LiteONNX(args.model, imgsz=args.imgsz, conf_thresh=args.conf)
    det.nms_thresh = float(args.nms)
    det.set_limits(min_box=args.min_box, max_det=args.max_det)
    tracker = ByteTrackLite(iou_high=args.iou_high, iou_low=args.iou_low,
                            conf_high=args.conf_high, conf_low=args.conf_low, max_ttl=30)

    src = Path(args.source)
    writer = None
    fps_out = None

    if src.is_dir():
        source_str = str(src)
        fps_out = float(args.fps if args.fps and args.fps > 0 else 30.0)
        frame_paths = sorted([p for p in src.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]])
        cap = None
    else:
        source_str = str(src)
        cap = cv2.VideoCapture(source_str)
        # Probe FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1 or fps > 120:
            fps_out = float(args.fps) if args.fps and args.fps > 0 else 30.0
        else:
            fps_out = float(fps)
        if fps_out is None:
            fps_out = 30.0

    frames_written = 0
    t_start = time.time()

    def next_frame_gen():
        if src.is_dir():
            for p in frame_paths:
                img = cv2.imread(str(p))
                yield img
        else:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                yield img

    for frame in next_frame_gen():
        if frame is None:
            continue

        # Lazy-init writer
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            writer = cv2.VideoWriter(args.out, fourcc, fps_out, (w, h))

        # Detect
        boxes_xyxy, confs, clses = det.infer(frame)
        if args.cls is not None:
            mask = (clses == int(args.cls))
            boxes_xyxy = boxes_xyxy[mask]
            confs = confs[mask]
            clses = clses[mask]

        # Track
        tracks = tracker.step(boxes_xyxy, confs, dt=1.0)

        # Draw
        drawn = 0
        for t in tracks:
            # draw only stable tracks unless explicitly requested
            if not args.draw_unconfirmed and (t.time_since_update != 0 or t.hits < args.min_hits):
                continue
            x1, y1, x2, y2 = t.xyxy().astype(int).tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            draw_text(frame, f"id:{t.id} conf:{t.score:.2f}", (x1, max(0, y1 - 22)),
                    font_scale=1, font_thickness=2, bg_color=(255,0,0))
            drawn += 1

        if args.diag:
            print(f"[diag] frame={frames_written + 1} dets={boxes_xyxy.shape[0]} -> tracks_drawn={drawn}", flush=True)

        writer.write(frame)
        frames_written += 1

    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()

    wall_dur = time.time() - t_start
    out_dur = frames_written / fps_out if frames_written and fps_out else 0
    proc_fps = frames_written / wall_dur if wall_dur > 0 else 0
    print(f"Done. Wrote: {args.out}")
    print(f"  Frames processed: {frames_written}")
    print(f"  Output video FPS (playback): {fps_out:.2f}, duration={out_dur:.2f}s")
    print(f"  Inference speed (processing FPS): {proc_fps:.2f}, wall time={wall_dur:.2f}s")
    print(f"Average inference FPS: {proc_fps:.2f} over {frames_written} frames (wall {wall_dur:.2f}s)")

if __name__ == "__main__":
    main()