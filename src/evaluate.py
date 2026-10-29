import argparse
import sys
import time
from pathlib import Path

from ultralytics import YOLO


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Evaluate Helmet Detection Model")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    args = parser.parse_args()

    project_root = get_project_root()

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = project_root / "models" / "best.pt"

    if not model_path.is_file():
        print(f"[ERROR] Model not found at: {model_path}")
        print("[INFO] Train the model first with: python src/train.py")
        sys.exit(1)

    data_yaml = project_root / "Data" / "data.yaml"
    if not data_yaml.is_file():
        print(f"[ERROR] data.yaml not found at: {data_yaml}")
        sys.exit(1)

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Helmet Detection - Model Evaluation")
    print("=" * 60)
    print(f"  Model    : {model_path}")
    print(f"  Dataset  : {data_yaml}")
    print(f"  Conf     : {args.conf}")
    print(f"  Img Size : {args.imgsz}")
    print("=" * 60)

    model = YOLO(str(model_path))

    class_names = model.names
    print(f"\n[INFO] Model classes: {class_names}")
    if class_names and (0 in class_names):
        if "Helmet" not in class_names[0] and "helmet" not in class_names[0]:
            print("[WARN] This model does not appear to be fine-tuned for helmet detection.")
            print(f"       Class 0 = '{class_names[0]}' (expected 'With Helmet')")
            print("[WARN] You may be using the wrong checkpoint.")

    print("\n" + "=" * 60)
    print("  Evaluating on TEST set (split=test)")
    print("=" * 60)

    start_time = time.time()

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        conf=args.conf,
        project=str(project_root / "runs" / "detect"),
        name="helmet_eval_test",
        exist_ok=True,
        verbose=True,
    )

    eval_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS (Test Set)")
    print("=" * 60)

    print(f"\n  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*25} {'-'*10}")
    print(f"  {'Precision':<25} {metrics.box.mp:>10.4f}")
    print(f"  {'Recall':<25} {metrics.box.mr:>10.4f}")
    print(f"  {'mAP@0.5':<25} {metrics.box.map50:>10.4f}")
    print(f"  {'mAP@0.5:0.95':<25} {metrics.box.map:>10.4f}")

    print(f"\n  Per-Class Performance:")
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*14}")

    for i, name in class_names.items():
        if i < len(metrics.box.p):
            print(f"  {name:<20} {metrics.box.p[i]:>10.4f} {metrics.box.r[i]:>10.4f} "
                  f"{metrics.box.ap50[i]:>10.4f} {metrics.box.ap[i]:>14.4f}")

    speed = metrics.speed
    preprocess_ms = speed.get("preprocess", 0)
    inference_ms = speed.get("inference", 0)
    postprocess_ms = speed.get("postprocess", 0)
    total_ms = preprocess_ms + inference_ms + postprocess_ms

    print(f"\n  Inference Timing:")
    print(f"  {'Preprocess':<25} {preprocess_ms:>10.2f} ms")
    print(f"  {'Inference':<25} {inference_ms:>10.2f} ms")
    print(f"  {'Postprocess':<25} {postprocess_ms:>10.2f} ms")
    print(f"  {'Total per image':<25} {total_ms:>10.2f} ms")
    if total_ms > 0:
        fps = 1000.0 / total_ms
        print(f"  {'Estimated FPS':<25} {fps:>10.1f}")

    print(f"\n  Total evaluation time: {eval_time:.1f}s")

    print("\n" + "=" * 60)
    print("  IMPORTANT NOTES")
    print("=" * 60)
    print("  - Check the confusion matrix in runs/detect/helmet_eval_test/")
    print("    for false positive analysis, especially for 'Without Helmet'.")
    print("  - This is an educational prototype. Do NOT use these metrics")
    print("    to claim enforcement-grade accuracy.")
    print("  - Evaluation was performed on the held-out test set only.")
    print("=" * 60)


if __name__ == "__main__":
    main()
