#https://medium.com/@beam_villa/object-tracking-made-easy-with-yolov11-bytetrack-73aac16a9f4a
#
#!/usr/bin/env python3
import argparse, os, sys, cv2, torch, math, time
from pathlib import Path
from ultralytics import YOLO

def draw_text(img, text, pos, font=cv2.FONT_HERSHEY_PLAIN,
              font_scale=1, font_thickness=1,
              text_color=(255,255,255), bg_color=(0,0,255)):
    x, y = pos
    # simple padded label box
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    tw, th = max(tw + 16, 50), max(th + 10, 20)
    cv2.rectangle(img, (x, y), (x + tw, y + th), bg_color, -1)
    cv2.putText(img, text, (x + 8, y + th - 6), font, font_scale, text_color, font_thickness)

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
    ap.add_argument("--fps", type=int, default=30,
                    help="Output FPS (if reading frames dir)")
    ap.add_argument("--device", default=None,
                    help="Force device, e.g. 'cpu' or '0' for GPU")
    ap.add_argument("--diag", action="store_true",
                    help="Print per-frame detection/tracking counts and IDs (no FPS)")
    args = ap.parse_args()

    # Load model
    model = YOLO(args.model)

    # Filter invalid detections before the tracker runs.
    # Ultralytics triggers the tracker at the end of postprocess via this callback hook.
    def _pretracker_sanitize(predictor):
        # Runs before tracker; drop NaN/Inf or non-positive boxes to avoid Kalman numeric failures
        res = getattr(predictor, "results", None)
        if not res:
            return
        for r in res:
            boxes = getattr(r, "boxes", None)
            data = getattr(boxes, "data", None)  # Tensor [N, >=6] with xyxy in first 4 cols
            if not isinstance(data, torch.Tensor) or data.numel() == 0:
                continue
            xyxy = data[:, :4]
            w = xyxy[:, 2] - xyxy[:, 0]
            h = xyxy[:, 3] - xyxy[:, 1]
            keep = torch.isfinite(xyxy).all(dim=1) & (w > 0) & (h > 0)
            boxes.data = data[keep] if keep.any() else data[:0]

    # Register sanitizer: runs just before the internal tracker update
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
                print(f"[diag] frame={frames_written + 1} dets_raw=0 -> 0 (empty) ids_avail=False", flush=True)
            writer.write(frame)
            frames_written += 1
            continue

        boxes_xywh = getattr(r.boxes, "xywh", None)
        track_ids  = getattr(r.boxes, "id", None)
        if boxes_xywh is None:
            if args.diag:
                print(f"[diag] frame={frames_written + 1} dets_raw=0 -> 0 (no xywh) ids_avail=False", flush=True)
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

        drawn_count = 0
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
            drawn_count += 1

        if args.diag:
            print(f"[diag] frame={frames_written + 1} dets_raw={len(boxes)} -> drawn={drawn_count} "
                  f"ids_avail={track_ids is not None}", flush=True)

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
    print(f"Average inference FPS: {proc_fps:.2f} over {frames_written} frames (wall {wall_dur:.2f}s)")

if __name__ == "__main__":
    main()