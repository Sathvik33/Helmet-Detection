import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

COLORS = {
    0: (0, 200, 0),
    1: (0, 0, 230),
}


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_model(model_path: Path) -> YOLO:
    """Load the fine-tuned YOLO model."""
    if not model_path.is_file():
        print(f"[ERROR] Model not found at: {model_path}")
        print("[INFO] Train the model first: python src/train.py")
        sys.exit(1)
    model = YOLO(str(model_path))
    print(f"[INFO] Loaded model: {model_path}")
    print(f"[INFO] Classes: {model.names}")
    return model


def annotate_frame(frame, results, fps_val):
    """Draw bounding boxes, labels, confidence scores, and stats on a frame."""
    annotated = frame.copy()
    helmet_count = 0
    no_helmet_count = 0

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        class_names = results[0].names

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]

            if cls_id == 0:
                helmet_count += 1
            else:
                no_helmet_count += 1

            color = COLORS.get(cls_id, (255, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} {conf:.2f}"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - lh - baseline - 4), (x1 + lw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    total = helmet_count + no_helmet_count
    hud_lines = [
        f"FPS: {fps_val:.1f}",
        f"Helmet: {helmet_count}  No Helmet: {no_helmet_count}",
    ]
    if total > 0:
        compliance = (helmet_count / total) * 100
        hud_lines.append(f"Compliance: {compliance:.0f}%")

    y = 25
    for line in hud_lines:
        cv2.putText(annotated, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 28

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Helmet Detection - Video Inference")
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    project_root = get_project_root()

    source = Path(args.source)
    if not source.is_file():
        print(f"[ERROR] Video not found: {source}")
        sys.exit(1)

    model_path = Path(args.model) if args.model else project_root / "models" / "best.pt"
    model = load_model(model_path)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {source}")
        sys.exit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Input: {source}")
    print(f"[INFO] Resolution: {width}x{height}, FPS: {fps_in:.1f}, Frames: {total_frames}")

    save_dir = Path(args.save_dir) if args.save_dir else project_root / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"detected_{source.name}"

    codecs = [("mp4v", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")]
    writer = None
    for codec, ext in codecs:
        out_file = output_path.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(out_file), fourcc, fps_in, (width, height))
        if writer.isOpened():
            output_path = out_file
            break
        writer.release()
        writer = None

    if writer is None:
        print("[ERROR] Could not initialize video writer with any codec.")
        cap.release()
        sys.exit(1)

    print(f"[INFO] Output: {output_path}")
    print("[INFO] Processing video...")

    frame_count = 0
    fps_display = 0.0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t0 = time.time()

        results = model(frame, conf=args.conf, verbose=False)

        t1 = time.time()
        fps_display = 1.0 / max(t1 - t0, 1e-6)

        annotated = annotate_frame(frame, results, fps_display)
        writer.write(annotated)

        if frame_count % 50 == 0 or frame_count == total_frames:
            elapsed = time.time() - t_start
            pct = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"  [{pct:5.1f}%] Frame {frame_count}/{total_frames} | "
                  f"FPS: {fps_display:.1f} | Elapsed: {elapsed:.1f}s")

    cap.release()
    writer.release()

    total_time = time.time() - t_start
    avg_fps = frame_count / max(total_time, 1e-6)

    print(f"\n[DONE] Processed {frame_count} frames in {total_time:.1f}s")
    print(f"[DONE] Average FPS: {avg_fps:.1f}")
    print(f"[DONE] Output saved to: {output_path}")


if __name__ == "__main__":
    main()
