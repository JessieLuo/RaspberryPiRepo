#https://medium.com/@beam_villa/object-tracking-made-easy-with-yolov11-bytetrack-73aac16a9f4a
#
#!/usr/bin/env python3
import argparse, os, sys, cv2, glob, torch
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
    ap.add_argument("--model", default="yolo11n.pt",
                    help="Ultralytics model weights (e.g., yolo11n.pt)")
    ap.add_argument("--conf", type=float, default=0.1,
                    help="Detection confidence (ByteTrack keeps lows in 2nd stage)")
    ap.add_argument("--fps", type=int, default=5,
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
                # No valid dets -> make boxes empty so tracker sees none
                r.boxes = None
            elif keep.sum() < keep.shape[0]:
                boxes.data = data[keep]

    # Register sanitizer: runs just before the internal tracker update
    model.add_callback('on_predict_postprocess_end', _pretracker_sanitize)

    # Frame iterator
    src = Path(args.source)
    if src.is_dir():
        frame_iter = iter_frames_from_dir(src)
        # peek one frame to init writer
        try:
            first = next(frame_iter)
        except StopIteration:
            print("No frames found in directory.", file=sys.stderr)
            sys.exit(2)
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
        frames = [first]
        def gen():
            for f in frames:
                yield f
            for f in frame_iter:
                yield f
        frame_stream = gen()
    else:
        # video path
        tmp_cap = cv2.VideoCapture(str(src))
        if not tmp_cap.isOpened():
            print(f"Cannot open: {src}", file=sys.stderr)
            sys.exit(2)
        fps = tmp_cap.get(cv2.CAP_PROP_FPS) or 30
        w  = int(tmp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(tmp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tmp_cap.release()
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
        frame_stream = iter_frames_from_video(src)

    # Tracking loop (Ultralytics built-in ByteTrack)
    # Tracker configs: ultralytics/cfg/trackers/{bytetrack.yaml,botsort.yaml}
    for frame in frame_stream:
        results = model.track(
            frame,
            persist=True,
            conf=args.conf,
            tracker="bytetrack.yaml",
            device=args.device,
            verbose=False
        )

        # Handle empty results safely
        if not results or results[0] is None or results[0].boxes is None:
            writer.write(frame)
            continue

        r = results[0]

        # Now retrieve sanitized boxes/ids
        boxes_xywh = getattr(r.boxes, "xywh", None)
        track_ids  = getattr(r.boxes, "id", None)

        if boxes_xywh is None or track_ids is None:
            writer.write(frame)
            continue

        ids = track_ids.int().cpu().tolist()
        confs = r.boxes.conf.cpu().tolist() if hasattr(r.boxes, "conf") else [None]*len(ids)
        boxes = boxes_xywh.cpu().tolist()

        import math
        for (xc, yc, w, h), tid, conf in zip(boxes, ids, confs):
            # Guard against invalid values from tracker/detector
            # Some detections can yield NaN/Inf or zero/negative width/height.
            # Without these checks, converting to int would raise errors or produce invalid rectangles.
            if not all(map(math.isfinite, (xc, yc, w, h))):
                continue
            if w <= 0 or h <= 0:
                continue
            x1 = int(xc - w/2); y1 = int(yc - h/2)
            x2 = int(xc + w/2); y2 = int(yc + h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"id:{tid}" if conf is None else f"id:{tid}  conf:{conf:.2f}"
            draw_text(frame, label, (x1, max(0, y1 - 22)), font_scale=1, font_thickness=2, bg_color=(255,0,0))

        writer.write(frame)

    writer.release()
    print(f"Done. Wrote: {args.out}")

if __name__ == "__main__":
    main()