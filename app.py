#!/usr/bin/env python3
"""Flask app for image upload and class-colored bounding box visualization."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template_string, request
from PIL import Image
from ultralytics import YOLO

APP_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Double-target readout classifier</title>
  <style>
    :root {
      --bg: #0d1117;
      --panel: #161b22;
      --text: #e6edf3;
      --muted: #9da7b3;
      --accent: #2f81f7;
      --border: #30363d;
    }
    body {
      margin: 0;
      font-family: "Segoe UI", "Noto Sans", sans-serif;
      background: radial-gradient(circle at 20% 20%, #1f2733 0%, var(--bg) 60%);
      color: var(--text);
    }
    .wrap {
      max-width: 1100px;
      margin: 24px auto;
      padding: 0 16px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 14px;
    }
    h1 {
      margin: 0 0 12px;
      font-size: 1.4rem;
    }
    p {
      color: var(--muted);
    }
    input[type=file] {
      color: var(--text);
    }
    button {
      background: var(--accent);
      border: 0;
      color: white;
      padding: 10px 14px;
      border-radius: 8px;
      cursor: pointer;
      margin-left: 8px;
    }
    .row {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 14px;
    }
    img {
      max-width: 100%;
      height: auto;
      border-radius: 10px;
      border: 1px solid var(--border);
    }
    .legend-item {
      display: flex;
      align-items: center;
      margin-bottom: 8px;
      font-size: 0.95rem;
    }
    .swatch {
      width: 16px;
      height: 16px;
      border-radius: 4px;
      margin-right: 10px;
      border: 1px solid #ffffff33;
    }
    @media (max-width: 800px) {
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1 style="text-align:center;">Double-target readout classifier</h1>
      <p>Upload an image to classify transfer/no transfer in double-target readouts. The image may contain an arbitrary number of double-targets, of any size, in any order.</p>
      {% if error %}<p style="color:#ff7b72">{{ error }}</p>{% endif %}
      <form method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required />
        <button type="submit">Run Detection</button>
      </form>
    </div>

    <div class="row">
      <div class="card">
        {% if output_image %}
          <img src="data:image/png;base64,{{ output_image }}" alt="Detection result" />
        {% else %}
          <p>No output yet.</p>
        {% endif %}
      </div>
      <div class="card">
        <h1>Label Legend</h1>
        {% for item in legend %}
          <div class="legend-item">
            <span class="swatch" style="background: {{ item.color }}"></span>
            <span>{{ item.name }}</span>
          </div>
        {% endfor %}
      </div>
    </div>
  </div>
</body>
</html>
"""

MODEL_PATH = Path("models/best.pt")
CLASSES_META = Path("dataset/classes.json")
CONF_THRESHOLD = 0.50

app = Flask(__name__)
_model = YOLO(str(MODEL_PATH)) if MODEL_PATH.exists() else None


def load_class_meta() -> tuple[list[str], dict[str, str], dict[str, list[int]]]:
    if CLASSES_META.exists():
        payload = json.loads(CLASSES_META.read_text(encoding="utf-8"))
        return (
            payload["names"],
            payload["colors_hex_by_name"],
            payload["colors_bgr_by_name"],
        )

    if _model is None:
        return [], {}, {}
    names_map = _model.names if isinstance(_model.names, dict) else {}
    names = [names_map[i] for i in sorted(names_map.keys())]
    fallback_hex = {name: "#00ccff" for name in names}
    fallback_bgr = {name: [255, 204, 0] for name in names}
    return names, fallback_hex, fallback_bgr


def draw_boxes(
    image_bgr: np.ndarray, result, names: list[str], bgr_by_name: dict[str, list[int]]
) -> np.ndarray:
    boxes = result.boxes
    if boxes is None:
        return image_bgr

    for box in boxes:
        conf = float(box.conf.item()) if box.conf is not None else 0.0
        if conf < CONF_THRESHOLD:
            continue
        cls_id = int(box.cls.item())
        if cls_id < 0 or cls_id >= len(names):
            continue
        label_name = names[cls_id]
        color = bgr_by_name.get(label_name, [0, 204, 255])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(image_bgr, p1, p2, color, 3)
    return image_bgr


def to_base64_png(image_bgr: np.ndarray) -> str:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
    names, hex_by_name, bgr_by_name = load_class_meta()
    legend = [
        {"name": name, "color": hex_by_name.get(name, "#00ccff")} for name in names
    ]
    output_image = None
    error = None

    if request.method == "POST":
        if _model is None:
            error = f"Model not found: {MODEL_PATH}. Train first, then retry."
            return render_template_string(
                APP_HTML, output_image=output_image, legend=legend, error=error
            )

        uploaded = request.files.get("image")
        if uploaded is None or uploaded.filename == "":
            error = "No file uploaded."
            return render_template_string(
                APP_HTML, output_image=output_image, legend=legend, error=error
            )

        raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            error = "Unable to decode image file."
            return render_template_string(
                APP_HTML, output_image=output_image, legend=legend, error=error
            )

        results = _model.predict(image, conf=CONF_THRESHOLD, verbose=False)
        boxed = draw_boxes(image.copy(), results[0], names, bgr_by_name)
        output_image = to_base64_png(boxed)

    return render_template_string(
        APP_HTML, output_image=output_image, legend=legend, error=error
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False)
