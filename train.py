#!/usr/bin/env python3
"""Train a YOLO model and save it for the web app."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/dataset.yaml", help="Path to YOLO dataset yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="cvat_train")
    parser.add_argument("--device", default=None, help="cuda device or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Training finished but best weights not found: {best}")

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / "best.pt"
    shutil.copy2(best, target)
    print(f"Saved trained model to: {target}")


if __name__ == "__main__":
    main()
