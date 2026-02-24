# 🩻 AI X-Ray Segmentation Project

## 📌 Project Overview

This project builds an **end-to-end medical image segmentation pipeline** using:

* **YOLOv8 (Segmentation)** for automatic detection
* **Label Studio** for manual annotation
* **Docker** for annotation environment
* **WSL (Ubuntu)** for development

The goal is to:

1. Take an X-ray image as input
2. Detect lung regions
3. Detect infection areas
4. Generate segmentation masks + bounding boxes
5. Train a custom YOLO segmentation model
6. Prepare for future correction workflow with SAM

---

# 🧱 Project Architecture

```
ai_model_deployment/
│
├── data/
│   ├── raw/                     # Original dataset (Kaggle)
│   ├── seg/
│   │   ├── raw/images/          # Images selected for annotation
│   │   ├── exports/             # Label Studio JSON exports
│   │   └── yolo/                # Converted YOLO segmentation dataset
│
├── scripts/
│   ├── yolo_segment.py
│   └── convert_labelstudio_to_yolo_seg.py
│
├── src/
│   ├── models/
│   └── utils/
│
└── README.md
```

---

# 🚀 Phase Progress

## ✅ Level 1 – Environment Setup

* WSL Ubuntu installed
* Python virtual environment created
* YOLOv8 installed
* Project structure initialized

Activate environment:

```bash
source venv/bin/activate
```

---

## ✅ Level 2 – Model Loading & Testing

* YOLO segmentation model loaded
* Inference tested on sample image
* Overlay, masks, JSON outputs generated

Run inference:

```bash
PYTHONPATH=. python scripts/yolo_segment.py \
  --image data/raw/test.jpg \
  --weights yolov8n-seg.pt
```

Outputs:

* Segmentation masks
* Bounding boxes
* JSON summary
* Overlay image

---

## ✅ Level 3 – Custom Segmentation Training (CURRENT LEVEL)

### Step 1 — Dataset Download

Downloaded medical X-ray dataset from Kaggle:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/raw/
```

---

### Step 2 — Image Selection

Copied 30 images for segmentation work:

```bash
mkdir -p data/seg/raw/images
```

Images used for annotation:

* Normal lungs
* Pneumonia lungs

---

### Step 3 — Annotation with Label Studio (Docker)

Run Label Studio:

```bash
docker run -it -p 8080:8080 \
  -v $PWD/data/seg:/label-studio/data \
  --name labelstudio \
  heartexlabs/label-studio:latest
```

Access:

```
http://localhost:8080
```

Created segmentation project with labels:

* `Poumon`
* `Infection`

Annotated images using Polygon tool.

---

### Step 4 — Export Annotations

Exported JSON from Label Studio:

```
data/seg/exports/labelstudio_export.json
```

---

### Step 5 — Convert to YOLO Segmentation Format

```bash
PYTHONPATH=. python scripts/convert_labelstudio_to_yolo_seg.py \
  --export data/seg/exports/labelstudio_export.json \
  --images-root data/seg/raw/images \
  --out data/seg/yolo \
  --classes Poumon,Infection
```

Generated structure:

```
data/seg/yolo/
├── images/train
├── images/val
├── labels/train
├── labels/val
└── data.yaml
```

---

### Step 6 — Train YOLO Segmentation Model

```bash
yolo segment train \
  model=yolov8n-seg.pt \
  data=data/seg/yolo/data.yaml \
  epochs=20 \
  imgsz=640 \
  batch=4
```

Output:

* Trained weights
* Metrics
* Predictions preview

---

# 🎯 Current Status

✔ Dataset downloaded
✔ Images selected
✔ Images annotated
✔ JSON exported
✔ Converted to YOLO format
🔄 Ready for full training & evaluation

---

# 🧠 What the Model Learns

The model learns to:

* Segment lung regions (Poumon)
* Segment infection areas (Infection)
* Predict masks for new X-ray images

It outputs:

* Pixel masks
* Bounding boxes
* Confidence scores

---

# 🔮 Next Phase (Upcoming)

* Integrate SAM for correction
* Build correction loop
* Auto-update YOLO with corrected masks
* Create simple UI pipeline

---

# 🛠 Tech Stack

* Python 3.10+
* YOLOv8 (Ultralytics)
* Label Studio
* Docker
* WSL Ubuntu
* OpenCV
* NumPy

---

# 📈 Future Improvements

* Increase annotated dataset size
* Add validation metrics tracking
* Add experiment tracking
* Build minimal web interface
* Deploy model via API

---

# 👤 Author

Ngahi
AI Model Deployment Project
First end-to-end computer vision project 🚀

---

# ⭐ Project Goal

Build a complete medical AI segmentation pipeline from scratch:

Dataset → Annotation → Conversion → Training → Inference → Improvement Loop


## ✅ Level 4 – Industrial KPI: Void Rate Computation

### 🎯 Objective

Transform segmentation outputs into a measurable industrial indicator.

We compute:
void_rate = total_void_area / component_area

This converts raw pixel segmentation into a meaningful KPI.

---

### 📊 Method

1. Generate segmentation masks using YOLO.
2. Extract binary masks:
   - chip.png (component / lung)
   - holes.png (voids / infection)
3. Compute:
   - component_area_px
   - void_area_px
4. Calculate:

void_rate = void_area_px / component_area_px

---

### 🧮 Run Void Rate Computation

```bash
PYTHONPATH=. python scripts/compute_void_rate.py --masks-dir outputs/yolo/masks
