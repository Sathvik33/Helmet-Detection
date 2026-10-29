import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

COLORS = {
    0: (0, 200, 0),
    1: (0, 0, 230),
}

COOLDOWN_SECONDS = 5.0


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_model(model_path: Path) -> YOLO:
    """Load the fine-tuned YOLO model."""
    if not model_path.is_file():
        print(f"[ERROR] Model not found at: {model_path}")
        sys.exit(1)
    model = YOLO(str(model_path))
    print(f"[INFO] Loaded model: {model_path}")
    return model


class ViolationLogger:
    """
    Logs helmet violations to CSV with duplicate prevention.

    Tracking IDs are TEMPORARY — they are assigned by the tracker and
    may change when a rider leaves and re-enters the frame, or when
    tracking fails.
    """

    def __init__(self, csv_path: Path, cooldown: float = COOLDOWN_SECONDS):
        self.csv_path = csv_path
        self.cooldown = cooldown
        self.last_logged = {}

        csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not csv_path.is_file():
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "track_id", "class", "confidence",
                    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                    "source", "note"
                ])

    def log_event(self, track_id: int, cls_name: str, confidence: float,
                  bbox: tuple, source: str):
        """Log a detection event if it passes the cooldown check."""
        now = time.time()

        if track_id in self.last_logged:
            elapsed = now - self.last_logged[track_id]
            if elapsed < self.cooldown:
                return False

        self.last_logged[track_id] = now
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, track_id, cls_name, f"{confidence:.4f}",
                bbox[0], bbox[1], bbox[2], bbox[3],
                source,
                "Tracking ID is temporary and may change between appearances"
            ])

        return True


def main():
    parser = argparse.ArgumentParser(description="Helmet Detection with Tracking & Violation Logging")
    parser.add_argument("--source", type=str, required=True,
                        help="Video file path or camera index (e.g., 0)")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_SECONDS,
                        help=f"Seconds between logging same track ID (default: {COOLDOWN_SECONDS})")
    parser.add_argument("--save-video", action="store_true", help="Save annotated output video")
    parser.add_argument("--log-all", action="store_true",
                        help="Log all classes, not just 'Without Helmet'")
    args = parser.parse_args()

    project_root = get_project_root()

    model_path = Path(args.model) if args.model else project_root / "models" / "best.pt"
    model = load_model(model_path)

    is_webcam = args.source.isdigit()
    if is_webcam:
        source = int(args.source)
        source_name = f"camera_{source}"
    else:
        source = args.source
        source_name = Path(source).stem
        if not Path(source).is_file():
            print(f"[ERROR] Video file not found: {source}")
            sys.exit(1)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {args.source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    results_dir = project_root / "results"
    csv_path = results_dir / "violations.csv"
    logger = ViolationLogger(csv_path, cooldown=args.cooldown)

    writer = None
    if args.save_video:
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"tracked_{source_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (width, height))
        if writer.isOpened():
            print(f"[INFO] Saving tracked video to: {out_path}")
        else:
            print("[WARN] Could not initialize video writer.")
            writer = None

    print(f"[INFO] Source: {args.source} ({width}x{height})")
    print(f"[INFO] Violation log: {csv_path}")
    print(f"[INFO] Cooldown: {args.cooldown}s per track ID")
    if is_webcam:
        print("[INFO] Press Q to quit.")

    frame_count = 0
    violations_logged = 0
    fps_display = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    time.sleep(0.1)
                    continue
                break

            frame_count += 1
            t0 = time.time()

            try:
                results = model.track(
                    frame,
                    conf=args.conf,
                    persist=True,
                    verbose=False,
                )
            except Exception as e:
                if frame_count == 1:
                    print(f"[WARN] Tracking failed ({e}). Falling back to detection-only mode.")
                results = model(frame, conf=args.conf, verbose=False)

            t1 = time.time()
            fps_display = 1.0 / max(t1 - t0, 1e-6)

            annotated = frame.copy()
            helmet_count = 0
            no_helmet_count = 0

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                class_names = results[0].names
                has_ids = boxes.id is not None

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = class_names[cls_id]
                    track_id = int(boxes.id[i]) if has_ids else -1

                    if cls_id == 0:
                        helmet_count += 1
                    else:
                        no_helmet_count += 1

                    if cls_id == 1 or args.log_all:
                        if track_id >= 0:
                            logged = logger.log_event(
                                track_id, cls_name, conf,
                                (x1, y1, x2, y2), source_name
                            )
                            if logged:
                                violations_logged += 1

                    color = COLORS.get(cls_id, (255, 255, 0))
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    if track_id >= 0:
                        label = f"ID:{track_id} {cls_name} {conf:.2f}"
                    else:
                        label = f"{cls_name} {conf:.2f}"

                    (lw, lh), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(annotated, (x1, y1 - lh - baseline - 4),
                                  (x1 + lw, y1), color, -1)
                    cv2.putText(annotated, label, (x1, y1 - baseline - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            total = helmet_count + no_helmet_count
            hud_y = 25
            cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            hud_y += 28
            cv2.putText(annotated, f"Helmet: {helmet_count}  No Helmet: {no_helmet_count}",
                        (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            hud_y += 28
            if total > 0:
                compliance = (helmet_count / total) * 100
                cv2.putText(annotated, f"Compliance: {compliance:.0f}%", (10, hud_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                hud_y += 28
            cv2.putText(annotated, f"Violations logged: {violations_logged}", (10, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 230), 2)

            if writer:
                writer.write(annotated)

            if is_webcam:
                cv2.imshow("Helmet Tracker", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break

            if not is_webcam and frame_count % 100 == 0:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                pct = (frame_count / max(total_frames, 1)) * 100
                print(f"  [{pct:.1f}%] Frame {frame_count} | FPS: {fps_display:.1f} | "
                      f"Violations: {violations_logged}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print(f"\n[DONE] Processed {frame_count} frames.")
    print(f"[DONE] Total violations logged: {violations_logged}")
    print(f"[DONE] Violation log saved to: {csv_path}")
    print("\n[NOTE] Tracking IDs are temporary. The same rider may receive")
    print("       different IDs across separate appearances. This system")
    print("       cannot reliably re-identify individuals.")


if __name__ == "__main__":
    main()
