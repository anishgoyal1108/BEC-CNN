# CVAT YOLO Training + Web Inference App

This project trains a YOLO model from CVAT YOLO 1.1 annotation exports and serves a web app on port `80`.

## 1) Install dependencies

Use Python `3.11` or `3.12` (recommended). Python `3.14` currently causes package resolution/wheel issues with `ultralytics` dependencies.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Build dataset from CVAT YOLO zips

By default, `organize_data.py` reads all `*.zip` files in this folder and images from `data/`.

```bash
python3 organize_data.py \
  --images-dir data \
  --output-dir dataset \
  --val-split 0.2
```

Outputs:
- `dataset/images/train`, `dataset/images/val`
- `dataset/labels/train`, `dataset/labels/val`
- `dataset/dataset.yaml`
- `dataset/classes.json` (class names + consistent class colors)

## 3) Train model

```bash
python3 train.py \
  --data dataset/dataset.yaml \
  --model yolov8n.pt \
  --epochs 60 \
  --imgsz 640
```

Trained model is copied to `models/best.pt`.

## 4) Run web app on port 80

```bash
sudo .venv/bin/python app.py
```

Open `http://<server-ip>/` in your browser.

## Behavior

- Upload an image.
- Output image contains bounding boxes only (no labels, no confidence text).
- Box color is class-consistent by label.
- Legend shows each label and its color.
