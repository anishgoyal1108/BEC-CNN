#!/usr/bin/env python3
"""Prepare a YOLO dataset from CVAT YOLO 1.1 exports and local images."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
import zipfile
from colorsys import hsv_to_rgb
from pathlib import Path

CVAT_LABEL_COLORS = {
    "transfer-target": "#6366f1",
    "notransfer-target": "#ff6b6b",
    "transfer-ring": "#5eea66",
    "notransfer-ring": "#e65ce8",
}


def hex_to_bgr(color_hex: str) -> list[int]:
    color_hex = color_hex.lstrip("#")
    r = int(color_hex[0:2], 16)
    g = int(color_hex[2:4], 16)
    b = int(color_hex[4:6], 16)
    return [b, g, r]


def label_color(label_name: str) -> str:
    if label_name in CVAT_LABEL_COLORS:
        return CVAT_LABEL_COLORS[label_name]
    digest = hashlib.md5(label_name.encode("utf-8")).hexdigest()
    hue = (int(digest[:8], 16) % 360) / 360.0
    sat = 0.75
    val = 0.95
    r, g, b = hsv_to_rgb(hue, sat, val)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zips",
        nargs="+",
        default=[str(p) for p in Path(".").glob("*.zip")],
        help="CVAT YOLO 1.1 zip files",
    )
    parser.add_argument(
        "--images-dir",
        default="data",
        help="Directory containing source images referenced by labels",
    )
    parser.add_argument("--output-dir", default="dataset", help="Dataset output directory")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    return parser.parse_args()


def load_classes(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        raw = zf.read("obj.names").decode("utf-8").strip().splitlines()
    return [x.strip() for x in raw if x.strip()]


def collect_annotations(zip_paths: list[Path], class_count: int) -> dict[str, list[str]]:
    annotations: dict[str, list[str]] = {}
    for zip_path in zip_paths:
        with tempfile.TemporaryDirectory(prefix="cvat_") as td:
            tmp = Path(td)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp)
            for txt_path in (tmp / "obj_train_data").glob("*.txt"):
                image_stem = txt_path.stem
                lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
                valid_lines = []
                for ln in lines:
                    parts = ln.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    if cls_id < 0 or cls_id >= class_count:
                        continue
                    valid_lines.append(" ".join(parts))
                if valid_lines:
                    annotations[image_stem] = valid_lines
    return annotations


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    zip_paths = [Path(p) for p in args.zips if Path(p).exists()]
    if not zip_paths:
        raise SystemExit("No zip files found. Pass --zips explicitly.")

    class_lists = [load_classes(zp) for zp in zip_paths]
    first_classes = class_lists[0]
    for idx, classes in enumerate(class_lists[1:], start=1):
        if classes != first_classes:
            raise SystemExit(
                f"Class mismatch between zip[0] and zip[{idx}]. "
                "All annotation zips must use identical class order."
            )

    class_names = first_classes
    class_count = len(class_names)

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    annotations = collect_annotations(zip_paths, class_count)
    stems = sorted(annotations.keys())
    random.seed(args.seed)
    random.shuffle(stems)

    val_count = int(len(stems) * args.val_split)
    val_stems = set(stems[:val_count])

    copied = 0
    missing = []
    for stem in stems:
        image_path = find_image(images_dir, stem)
        if image_path is None:
            missing.append(stem)
            continue
        split = "val" if stem in val_stems else "train"
        dst_image = output_dir / "images" / split / image_path.name
        dst_label = output_dir / "labels" / split / f"{stem}.txt"
        shutil.copy2(image_path, dst_image)
        dst_label.write_text("\n".join(annotations[stem]) + "\n", encoding="utf-8")
        copied += 1

    class_colors = {name: label_color(name) for name in class_names}

    names_yaml = ", ".join([f"'{name}'" for name in class_names])
    dataset_yaml_text = (
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {class_count}\n"
        f"names: [{names_yaml}]\n"
    )
    (output_dir / "dataset.yaml").write_text(dataset_yaml_text, encoding="utf-8")
    (output_dir / "classes.json").write_text(
        json.dumps(
            {
                "names": class_names,
                "colors_hex_by_name": class_colors,
                "colors_bgr_by_name": {k: hex_to_bgr(v) for k, v in class_colors.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "zip_files": [str(zp) for zp in zip_paths],
        "images_dir": str(images_dir),
        "output_dir": str(output_dir),
        "total_annotations": len(stems),
        "copied_images": copied,
        "missing_images": missing,
        "classes": class_names,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Prepared dataset at: {output_dir}")
    print(f"Classes ({class_count}): {', '.join(class_names)}")
    print(f"Copied images: {copied}/{len(stems)}")
    if missing:
        print(f"Missing images: {len(missing)} (see {output_dir / 'summary.json'})")


if __name__ == "__main__":
    main()
