import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO


def get_project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def find_pretrained_weights(project_root: Path) -> str:
    """Locate the pretrained YOLOv8-Nano weights file."""
    candidates = [
        project_root / "model" / "yolov8n.pt",
        project_root / "models" / "yolov8n.pt",
        project_root / "yolov8n.pt",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)

    print("[INFO] yolov8n.pt not found locally. Ultralytics will download it automatically.")
    return "yolov8n.pt"


def find_data_yaml(project_root: Path) -> str:
    """Locate the dataset configuration file."""
    data_yaml = project_root / "Data" / "data.yaml"
    if not data_yaml.is_file():
        print(f"[ERROR] data.yaml not found at {data_yaml}")
        sys.exit(1)
    return str(data_yaml)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-Nano for Helmet Detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()

    project_root = get_project_root()
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    best_pt = models_dir / "best.pt"

    if best_pt.is_file() and not args.resume:
        print(f"[INFO] Trained checkpoint already exists at: {best_pt}")
        print("[INFO] To retrain, delete models/best.pt or use --resume.")
        print("[INFO] Skipping training.")
        return

    weights = find_pretrained_weights(project_root)
    data_yaml = find_data_yaml(project_root)

    print("=" * 60)
    print("  Helmet Detection - YOLOv8-Nano Training")
    print("=" * 60)
    print(f"  Weights  : {weights}")
    print(f"  Dataset  : {data_yaml}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  Img Size : {args.imgsz}")
    print(f"  Patience : {args.patience}")
    print("=" * 60)

    if args.resume:
        last_pt = project_root / "runs" / "detect" / "helmet_train" / "weights" / "last.pt"
        if last_pt.is_file():
            print(f"[INFO] Resuming training from: {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print("[WARN] No last.pt found for resume. Starting fresh training.")
            model = YOLO(weights)
    else:
        model = YOLO(weights)

    try:
        results = model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            project=str(project_root / "runs" / "detect"),
            name="helmet_train",
            exist_ok=True,
            pretrained=True,
            verbose=True,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"\n[ERROR] GPU memory error: {e}")
            print("[INFO] Retrying with batch=8...")
            model = YOLO(weights)
            results = model.train(
                data=data_yaml,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=8,
                patience=args.patience,
                project=str(project_root / "runs" / "detect"),
                name="helmet_train",
                exist_ok=True,
                pretrained=True,
                verbose=True,
            )
        else:
            raise

    train_best = project_root / "runs" / "detect" / "helmet_train" / "weights" / "best.pt"
    if train_best.is_file():
        shutil.copy2(str(train_best), str(best_pt))
        print(f"\n[SUCCESS] Best checkpoint saved to: {best_pt}")
    else:
        print("[WARN] best.pt not found in training output. Check runs/detect/helmet_train/")

    train_last = project_root / "runs" / "detect" / "helmet_train" / "weights" / "last.pt"
    last_dest = models_dir / "last.pt"
    if train_last.is_file():
        shutil.copy2(str(train_last), str(last_dest))
        print(f"[INFO] Last checkpoint saved to: {last_dest}")

    print("\n[INFO] Training complete. Artifacts saved in runs/detect/helmet_train/")


if __name__ == "__main__":
    main()