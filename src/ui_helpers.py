import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

COLORS = {
    0: (46, 204, 113),  # Green for With Helmet (BGR: cv2 format: (113, 204, 46))
    1: (231, 76, 60),   # Red for Without Helmet (BGR: (60, 76, 231))
}

COLOR_BGR = {
    0: (46, 204, 113),   # With Helmet
    1: (60, 76, 231),   # Without Helmet
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_available_models() -> Dict[str, Path]:
    root = get_project_root()
    models = {}
    
    best_path = root / "models" / "best.pt"
    if best_path.is_file():
        models["Fine-Tuned YOLOv8n (Best)"] = best_path
        
    last_path = root / "models" / "last.pt"
    if last_path.is_file():
        models["Fine-Tuned YOLOv8n (Last)"] = last_path
        
    base_path = root / "model" / "yolov8n.pt"
    if base_path.is_file():
        models["Baseline YOLOv8n (Pretrained)"] = base_path
        
    return models


def load_model(model_path: Path) -> YOLO:
    return YOLO(str(model_path))


def annotate_frame(
    image: np.ndarray,
    results,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_hud: bool = True,
    fps: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, any], List[Dict[str, any]]]:
    """
    Annotate image with bounding boxes, labels, and statistics.
    Returns:
        annotated_image (BGR)
        summary_stats (dict)
        detections_list (list of dicts)
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    helmet_count = 0
    no_helmet_count = 0
    detections = []

    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        class_names = results[0].names
        has_ids = boxes.id is not None

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = class_names.get(cls_id, str(cls_id))
            track_id = int(boxes.id[i]) if has_ids else None

            if cls_id == 0:
                helmet_count += 1
            else:
                no_helmet_count += 1

            detections.append({
                "Track ID": track_id if track_id is not None else "-",
                "Class": cls_name,
                "Confidence": f"{conf:.2%}",
                "Conf_Value": conf,
                "BBox": f"[{x1}, {y1}, {x2}, {y2}]",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "Class_ID": cls_id,
            })

            if show_boxes:
                color = COLOR_BGR.get(cls_id, (0, 255, 255))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                if show_labels:
                    if track_id is not None and track_id >= 0:
                        label = f"#{track_id} {cls_name} {conf:.2f}"
                    else:
                        label = f"{cls_name} {conf:.2f}"

                    (lw, lh), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                    )
                    top = max(y1, lh + baseline + 4)
                    cv2.rectangle(
                        annotated,
                        (x1, top - lh - baseline - 4),
                        (x1 + lw, top),
                        color,
                        -1,
                    )
                    cv2.putText(
                        annotated,
                        label,
                        (x1, top - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

    total = helmet_count + no_helmet_count
    compliance_rate = (helmet_count / total * 100) if total > 0 else 0.0

    if show_hud:
        # Draw sleek semi-transparent HUD banner in top-left
        hud_lines = []
        if fps is not None:
            hud_lines.append(f"FPS: {fps:.1f}")
        hud_lines.append(f"Total: {total}")
        hud_lines.append(f"Helmet: {helmet_count}")
        hud_lines.append(f"No Helmet: {no_helmet_count}")
        if total > 0:
            hud_lines.append(f"Compliance: {compliance_rate:.0f}%")

        box_height = len(hud_lines) * 24 + 14
        box_width = 190
        overlay = annotated.copy()
        cv2.rectangle(overlay, (8, 8), (8 + box_width, 8 + box_height), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

        y_text = 28
        for line in hud_lines:
            cv2.putText(
                annotated,
                line,
                (16, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            y_text += 24

    stats = {
        "total": total,
        "helmet_count": helmet_count,
        "no_helmet_count": no_helmet_count,
        "compliance_rate": compliance_rate,
    }

    return annotated, stats, detections


def convert_video_to_h264(input_path: str, output_path: str) -> bool:
    """Transcode video to H.264 mp4 using ffmpeg for HTML5 browser compatibility."""
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "23",
            output_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except Exception:
        return False
