# 🏍️ Real-Time Automated Bicycle & Scooter Helmet Compliance Checker

A computer vision application that detects whether riders are wearing helmets, using **YOLOv8-Nano** fine-tuned on the [Roboflow Bicycle Helmet Detection](https://universe.roboflow.com/bicycle-helemt/bike-helmet-detection-2vdjo-qkthu/dataset/1) dataset.

> **⚠️ Educational Prototype** — This is a student project for learning computer vision. It is NOT an automated law-enforcement system and should NOT be used for enforcement decisions.

---

## ✨ Features

| Feature | Status |
|---|---|
| Streamlit interactive web frontend (Photo & Video Live/Upload) | ✅ Implemented |
| YOLOv8-Nano fine-tuning on custom helmet dataset | ✅ Implemented |
| Single image detection with annotated output | ✅ Implemented |
| Recorded video detection with FPS overlay | ✅ Implemented |
| Real-time webcam detection | ✅ Implemented |
| Compliance statistics (helmet/no-helmet counts, %) | ✅ Implemented |
| Rider tracking with BoT-SORT / ByteTrack | ✅ Implemented |
| Violation CSV logging with deduplication | ✅ Implemented |
| GPU + CPU support | ✅ Implemented |

---

## 🏗️ Architecture & Workflow

```
┌──────────────┐    ┌─────────────┐    ┌──────────────────┐
│  Input Source │───▶│ YOLOv8-Nano │───▶│ Detection Output │
│  (img/vid/cam)│    │ (fine-tuned)│    │ (boxes + labels) │
└──────────────┘    └─────────────┘    └──────────────────┘
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                        ┌──────────┐   ┌──────────┐   ┌──────────┐
                        │ Annotated│   │ Compliance│   │ Violation│
                        │ Display  │   │   Stats   │   │  CSV Log │
                        └──────────┘   └──────────┘   └──────────┘
```

### Classes
| Class ID | Label | Description |
|---|---|---|
| 0 | With Helmet | Rider wearing a helmet |
| 1 | Without Helmet | Rider NOT wearing a helmet |

---

## 📁 Project Structure

```
Helmet_detection/
├── app.py                      # Streamlit interactive web application
├── Data/                       # Dataset (Roboflow export)
│   ├── data.yaml               # Dataset config (abs paths)
│   ├── train/images/ & labels/ # 3,548 training images
│   ├── valid/images/ & labels/ # 127 validation images
│   └── test/images/ & labels/  # 64 test images
├── models/
│   └── best.pt                 # Fine-tuned checkpoint
├── src/
│   ├── ui_helpers.py           # Streamlit UI annotation & video conversion helpers
│   ├── train.py                # Training script
│   ├── evaluate.py             # Test-set evaluation
│   ├── detect_image.py         # Single image inference
│   ├── detect_video.py         # Video file inference
│   ├── webcam.py               # Real-time webcam detection
│   └── tracker.py              # Tracking + violation logging
├── runs/detect/                # Ultralytics training artifacts
├── results/                    # Output images, videos, CSVs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📦 Dataset

- **Source**: [Roboflow - Bike Helmet Detection v1](https://universe.roboflow.com/bicycle-helemt/bike-helmet-detection-2vdjo-qkthu/dataset/1)
- **Total Images**: 3,739 (augmented from ~1,371 originals)
- **Split**: Train 3,548 / Valid 127 / Test 64
- **Format**: YOLOv8 (normalized bounding boxes in `.txt` files)
- **Resolution**: 640×640 (stretched)
- **Annotations**: 9,842 total (6,530 With Helmet, 3,312 Without Helmet)
- **License**: Public Domain

---

## 🛠️ Installation (Windows)

### Prerequisites
- Python 3.8+ (tested with 3.10+)
- NVIDIA GPU with CUDA (recommended) or CPU

### Steps

```bash
# 1. Clone or navigate to the project
cd Helmet_detection

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 0. Launch the Interactive Web App (Streamlit Frontend)

```bash
streamlit run app.py
```

The web application opens at `http://localhost:8501` featuring:
- 🖼️ **Upload Photo**: Upload custom photos or select from test dataset samples. Displays bounding boxes, confidence scores, compliance percentage, detections breakdown table, and download option.
- 📸 **Live Photo (Camera)**: Capture instant snapshots with browser camera/webcam and run detection immediately.
- 🎞️ **Upload Video**: Process MP4/AVI/MOV/MKV videos with optional ByteTrack object tracking, frame stride speedup, real-time preview, browser video player, and downloadable violation CSV logs.
- 📹 **Live Video (Webcam)**: Real-time live camera feed with on-screen HUD, FPS counter, compliance metrics, and interactive start/stop toggle.
- ⚙️ **Customizable Controls**: Switch between fine-tuned checkpoints, adjust confidence/IoU sliders, and customize visual overlays on the fly.

### 1. Train the Model

```bash
python src/train.py
```

Options:
```bash
python src/train.py --epochs 100 --batch 8 --imgsz 640 --patience 15
python src/train.py --resume    # Resume from last checkpoint
```

The best checkpoint is saved to `models/best.pt`.
Training artifacts (plots, metrics) are in `runs/detect/helmet_train/`.

### 2. Evaluate on Test Set

```bash
python src/evaluate.py
```

Options:
```bash
python src/evaluate.py --model models/best.pt --conf 0.25
```

### 3. Detect in Images

```bash
python src/detect_image.py --source path/to/image.jpg
python src/detect_image.py --source photo.png --conf 0.5 --no-display
```

Output saved to `results/detected_<filename>`.

### 4. Detect in Videos

```bash
python src/detect_video.py --source input.mp4
python src/detect_video.py --source clip.avi --conf 0.4
```

Output saved to `results/detected_<filename>`.

### 5. Real-Time Webcam

```bash
python src/webcam.py
python src/webcam.py --conf 0.4 --camera 0
```

Press **Q** to quit. Shows live FPS, counts, and compliance %.

### 6. Tracking + Violation Logging

```bash
# From video file
python src/tracker.py --source input.mp4 --save-video

# From webcam
python src/tracker.py --source 0
```

Violations are logged to `results/violations.csv` with deduplication.

---

## ☁️ Google Colab Training

To train in Colab:

```python
# 1. Install ultralytics
!pip install ultralytics

# 2. Upload or mount your dataset
# Option A: Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Update data.yaml path to match Colab filesystem
# Change: path: /content/drive/MyDrive/Helmet_detection/Data

# 4. Train
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model.train(
    data="/content/drive/MyDrive/Helmet_detection/Data/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
)

# 5. Download the best weights
# The checkpoint will be at: runs/detect/train/weights/best.pt
```

> **Note**: Update the `path` field in `data.yaml` to point to the Colab filesystem location of your `Data/` folder.

---

## 📊 Output Structure

After training and running detection:

```
results/
├── detected_sample.jpg         # Annotated image output
├── detected_input.mp4          # Annotated video output
├── tracked_input.mp4           # Tracked video output
└── violations.csv              # Violation event log

runs/detect/
├── helmet_train/               # Training artifacts
│   ├── weights/best.pt
│   ├── weights/last.pt
│   ├── results.png             # Training curves
│   ├── confusion_matrix.png
│   └── ...
└── helmet_eval_test/           # Evaluation artifacts
    ├── confusion_matrix.png
    └── ...
```

---

## ⚠️ Limitations & Ethical Considerations

### Technical Limitations
- **Small test set** (64 images) — evaluation metrics may not generalize well
- **Augmented dataset** — 3× augmentation from ~1,371 originals means limited real-world diversity
- **No rider-helmet association** — the model detects helmets and no-helmets independently, not as attributes of a specific person
- **Tracking IDs are temporary** — BoT-SORT assigns IDs per appearance; re-entering riders get new IDs
- **Resolution-dependent** — performance may vary with image quality, distance, and angle
- **Single-class confusion** — may confuse hats, visors, or partially visible helmets

### Ethical Considerations
- **NOT for enforcement** — do not use to issue fines, penalties, or legal actions
- **Privacy** — recording and analyzing individuals raises privacy concerns
- **Bias** — the model's accuracy may vary across different demographics, helmet types, and lighting conditions
- **False positives/negatives** — incorrect detections can have real consequences if misused
- **Consent** — ensure proper consent and legal compliance before deploying camera-based monitoring

---

## 📝 License

- **Dataset**: Public Domain (via Roboflow)
- **YOLOv8**: AGPL-3.0 (Ultralytics)
- **Project Code**: For educational use
