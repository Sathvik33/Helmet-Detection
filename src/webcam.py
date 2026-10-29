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


def main():
    parser = argparse.ArgumentParser(description="Helmet Detection - Real-Time Webcam")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    args = parser.parse_args()

    project_root = get_project_root()

    model_path = Path(args.model) if args.model else project_root / "models" / "best.pt"
    model = load_model(model_path)

    print(f"[INFO] Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        print("[INFO] Check that a webcam is connected and not in use by another app.")
        print("[INFO] Try a different camera index with: --camera 1")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Webcam resolution: {actual_w}x{actual_h}")
    print(f"[INFO] Press 'Q' to quit.")

    fps_display = 0.0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame. Retrying...")
                time.sleep(0.1)
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Cannot read from webcam. Exiting.")
                    break

            frame_count += 1
            t0 = time.time()

            results = model(frame, conf=args.conf, verbose=False)

            t1 = time.time()
            fps_display = 1.0 / max(t1 - t0, 1e-6)

            annotated = frame.copy()
            helmet_count = 0
            no_helmet_count = 0

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = results[0].names[cls_id]

                    if cls_id == 0:
                        helmet_count += 1
                    else:
                        no_helmet_count += 1

                    color = COLORS.get(cls_id, (255, 255, 0))
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

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

            cv2.putText(annotated, f"Helmet: {helmet_count}", (10, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            hud_y += 28

            cv2.putText(annotated, f"No Helmet: {no_helmet_count}", (10, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 230), 2)
            hud_y += 28

            if total > 0:
                compliance = (helmet_count / total) * 100
                comp_color = (0, 200, 0) if compliance >= 80 else (0, 165, 255) if compliance >= 50 else (0, 0, 230)
                cv2.putText(annotated, f"Compliance: {compliance:.0f}%", (10, hud_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, comp_color, 2)

            cv2.putText(annotated, "Press Q to quit", (10, actual_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Helmet Compliance Checker", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("[INFO] User pressed Q. Exiting.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Processed {frame_count} frames. Webcam released.")


if __name__ == "__main__":
    main()
