import argparse
import sys
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


def annotate_image(image, results):
    """Draw bounding boxes, labels, and confidence scores on the image."""
    annotated = image.copy()

    if results[0].boxes is None or len(results[0].boxes) == 0:
        cv2.putText(annotated, "No detections", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        return annotated

    boxes = results[0].boxes
    class_names = results[0].names
    helmet_count = 0
    no_helmet_count = 0

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
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(annotated, (x1, y1 - label_h - baseline - 5),
                      (x1 + label_w, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    total = helmet_count + no_helmet_count
    summary_lines = [
        f"Total Detections: {total}",
        f"With Helmet: {helmet_count}",
        f"Without Helmet: {no_helmet_count}",
    ]
    if total > 0:
        compliance = (helmet_count / total) * 100
        summary_lines.append(f"Compliance: {compliance:.0f}%")

    y_offset = 30
    for line in summary_lines:
        cv2.putText(annotated, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 28

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Helmet Detection - Image Inference")
    parser.add_argument("--source", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory (default: results/)")
    parser.add_argument("--no-display", action="store_true", help="Don't display the output image")
    args = parser.parse_args()

    project_root = get_project_root()

    source = Path(args.source)
    if not source.is_file():
        print(f"[ERROR] Image not found: {source}")
        sys.exit(1)

    model_path = Path(args.model) if args.model else project_root / "models" / "best.pt"
    model = load_model(model_path)

    image = cv2.imread(str(source))
    if image is None:
        print(f"[ERROR] Could not read image: {source}")
        sys.exit(1)

    print(f"[INFO] Input image: {source} ({image.shape[1]}x{image.shape[0]})")

    results = model(image, conf=args.conf, verbose=False)
    annotated = annotate_image(image, results)

    save_dir = Path(args.save_dir) if args.save_dir else project_root / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"detected_{source.name}"
    cv2.imwrite(str(output_path), annotated)
    print(f"[INFO] Saved annotated image to: {output_path}")

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        print(f"\n  Detections:")
        for box in results[0].boxes:
            cls_name = results[0].names[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f"    - {cls_name}: {conf:.3f}")

    if not args.no_display:
        try:
            cv2.imshow("Helmet Detection", annotated)
            print("\n[INFO] Press any key to close the window.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            print("[INFO] Display not available. Image saved to disk.")


if __name__ == "__main__":
    main()
