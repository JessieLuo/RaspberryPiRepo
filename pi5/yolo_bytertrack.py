#https://medium.com/@beam_villa/object-tracking-made-easy-with-yolov11-bytetrack-73aac16a9f4a
#
#!/usr/bin/env python3
import argparse, os, sys, cv2, glob, torch
from pathlib import Path
from ultralytics import YOLO
import time

import os

# Limit thread contention on Pi 5 (helps stability)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
try:
    import torch
    torch.set_num_threads(4)
except Exception:
    pass
try:
    cv2.setNumThreads(1)
except Exception:
    pass

def draw_text(img, text, pos, font=cv2.FONT_HERSHEY_PLAIN,
              font_scale=1, font_thickness=1,
              text_color=(255,255,255), bg_color=(0,0,255)):
    x, y = pos
    # simple padded label box
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    tw, th = max(tw + 16, 50), max(th + 10, 20)
    cv2.rectangle(img, (x, y), (x + tw, y + th), bg_color, -1)
    cv2.putText(img, text, (x + 8, y + th - 6), font, font_scale, text_color, font_thickness)

def iter_frames_from_dir(frames_dir):
    imgs = sorted(glob.glob(str(Path(frames_dir) / "*")))
    for p in imgs:
        frame = cv2.imread(p)
        if frame is None:
            continue
        yield frame

def iter_frames_from_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()

def main():
    ap = argparse.ArgumentParser("YOLO11 + ByteTrack demo")
    ap.add_argument("--source", required=True,
                    help="Path to frames directory OR a video file")
    ap.add_argument("--out", default="tracking_out.mp4",
                    help="Output video filename (mp4)")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="Ultralytics .pt weights. Fast on Pi: 'yolov8n.pt' (default) or 'yolov3-tiny.pt'. Use your own .pt as needed.")
    ap.add_argument("--conf", type=float, default=0.1,
                    help="Detection confidence (ByteTrack keeps lows in 2nd stage)")
    ap.add_argument("--imgsz", type=int, default=384,
                    help="Inference image size (short side). Lower = faster on Pi")
    ap.add_argument("--diag", action="store_true",
                    help="Print per-frame detection counts before/after filtering for debugging")
    ap.add_argument("--no-sanitize", action="store_true",
                    help="Disable pre-tracker sanitization (debug only)")
    ap.add_argument("--fps", type=int, default=30,
                    help="Output FPS (if reading frames dir)")
    ap.add_argument("--device", default=None,
                    help="Force device, e.g. 'cpu' or '0' for GPU")
    args = ap.parse_args()

    # Load model
    model = YOLO(args.model)

    # Filter invalid detections *before* the tracker runs.
    # Ultralytics triggers the tracker at the end of postprocess via this callback hook.
    def _pretracker_sanitize(predictor):
        # predictor.results is a list of Results for the current frame/batch
        res = getattr(predictor, "results", None)
        if not res:
            return
        for r in res:
            boxes = getattr(r, "boxes", None)
            data = getattr(boxes, "data", None)  # Tensor [N, >=6] with xyxy in first 4 cols
            if data is None:
                continue
            # Ensure tensor
            try:
                import torch
            except Exception:
                return
            if not isinstance(data, torch.Tensor) or data.numel() == 0:
                continue
            xyxy = data[:, :4]
            w = xyxy[:, 2] - xyxy[:, 0]
            h = xyxy[:, 3] - xyxy[:, 1]
            finite_mask = torch.isfinite(xyxy).all(dim=1)
            size_mask = (w > 0) & (h > 0)
            keep = finite_mask & size_mask
            if keep.sum() == 0:
                # No valid dets -> keep Boxes object but empty its data so tracker sees zero detections
                boxes.data = data[:0]
            elif keep.sum() < keep.shape[0]:
                boxes.data = data[keep]

    # Register sanitizer: runs just before the internal tracker update
    if not args.no_sanitize:
        model.add_callback('on_predict_postprocess_end', _pretracker_sanitize)

    # Frame iterator / source setup
    src = Path(args.source)
    writer = None
    fps_out = None

    if src.is_dir():
        # For directories, let Ultralytics stream the folder directly; set FPS from args
        source_str = str(src)
        fps_out = float(args.fps if args.fps and args.fps > 0 else 30.0)
    else:
        # Video file or URL (e.g., RTSP) — let Ultralytics handle the stream
        source_str = str(src)
        # Probe FPS to set output metadata sensibly
        tmp_cap = cv2.VideoCapture(source_str)
        fps = tmp_cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1 or fps > 120:
            fps_out = float(args.fps) if args.fps and args.fps > 0 else 30.0
        else:
            fps_out = float(fps)
        print(f"[info] using output FPS={fps_out:.2f}", file=sys.stderr)
        tmp_cap.release()

    frames_written = 0
    t_start = time.time()

    # Real-time FPS tracking (instant & EMA) for diag output
    t_prev = t_start
    ema_fps = None
    ema_alpha = 0.2  # smoothing factor for EMA FPS

    # Tracking loop (Ultralytics built-in ByteTrack) using persistent streaming on a supported source string
    results_iter = model.track(
        source=source_str,
        stream=True,
        persist=True,
        conf=args.conf,
        imgsz=args.imgsz,
        tracker="bytetrack.yaml",
        device=args.device,
        verbose=False,
    )

    for r in results_iter:
        frame = getattr(r, "orig_img", None)
        if args.diag:
            try:
                raw_n = int(getattr(getattr(r, 'boxes', None), 'data', []).shape[0])
            except Exception:
                raw_n = 0

        # Compute per-frame instantaneous and smoothed (EMA) FPS
        t_now = time.time()
        dt = t_now - t_prev
        inst_fps = (1.0 / dt) if dt > 0 else 0.0
        ema_fps = inst_fps if ema_fps is None else (ema_alpha * inst_fps + (1.0 - ema_alpha) * ema_fps)
        t_prev = t_now

        # Prepare FPS label for overlay with inst and ema FPS
        fps_label = f"FPS {ema_fps:.2f} (inst {inst_fps:.2f})"

        if frame is None:
            continue

        # Lazy-init writer on first frame
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            writer = cv2.VideoWriter(args.out, fourcc, fps_out, (w, h))

        # Handle empty results safely
        if r.boxes is None or (hasattr(r.boxes, "data") and r.boxes.data is not None and r.boxes.data.numel() == 0):
            if args.diag:
                print(f"[diag] frame={frames_written} dets_raw={raw_n} -> 0 (empty) "
                      f"(ids avail: False) | inst_fps={inst_fps:.2f} ema_fps={ema_fps:.2f}")
            cv2.putText(frame, fps_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)
            writer.write(frame)
            frames_written += 1
            continue

        boxes_xywh = getattr(r.boxes, "xywh", None)
        track_ids  = getattr(r.boxes, "id", None)
        if boxes_xywh is None:
            if args.diag:
                print(f"[diag] frame={frames_written} dets_raw={raw_n} -> 0 (no xywh) "
                      f"(ids avail: False) | inst_fps={inst_fps:.2f} ema_fps={ema_fps:.2f}")
            cv2.putText(frame, fps_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)
            writer.write(frame)
            frames_written += 1
            continue

        boxes = boxes_xywh.cpu().tolist()
        # Note: ByteTrack may produce detections without IDs on some frames (e.g., lost/re-init). We still render boxes.
        if track_ids is None:
            ids = [-1] * len(boxes)
        else:
            ids = track_ids.int().cpu().tolist()

        confs = r.boxes.conf.cpu().tolist() if hasattr(r.boxes, "conf") else [None]*len(boxes)

        import math
        for (xc, yc, w, h), tid, conf in zip(boxes, ids, confs):
            # Guard against invalid values from tracker/detector
            if not all(map(math.isfinite, (xc, yc, w, h))):
                continue
            if w <= 0 or h <= 0:
                continue
            x1 = int(xc - w/2); y1 = int(yc - h/2)
            x2 = int(xc + w/2); y2 = int(yc + h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"id:{tid}" if conf is None else f"id:{tid}  conf:{conf:.2f}"
            draw_text(frame, label, (x1, max(0, y1 - 22)), font_scale=1, font_thickness=2, bg_color=(255,0,0))

        if args.diag:
            print(f"[diag] frame={frames_written} dets_raw={raw_n} -> drawn={len(boxes)} "
                  f"(ids avail: {track_ids is not None}) "
                  f"| inst_fps={inst_fps:.2f} ema_fps={ema_fps:.2f}")

        cv2.putText(frame, fps_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        frames_written += 1
        writer.write(frame)

    writer.release()
    wall_dur = time.time() - t_start
    try:
        out_dur = frames_written / fps_out if frames_written and fps_out else 0
    except Exception:
        out_dur = 0
    proc_fps = frames_written / wall_dur if wall_dur > 0 else 0

    print(f"Done. Wrote: {args.out}")
    print(f"  Frames processed: {frames_written}")
    print(f"  Output video FPS (playback): {fps_out:.2f}, duration={out_dur:.2f}s")
    print(f"  Inference speed (processing FPS): {proc_fps:.2f}, wall time={wall_dur:.2f}s")

if __name__ == "__main__":
    main()