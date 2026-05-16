import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import torch

ARCHIVE_DIR = "archive"


def convert_visdrone_to_yolo(src_images, src_annotations, dst_images, dst_labels):
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    img_files = sorted(os.listdir(src_images))
    skipped = 0

    for fname in img_files:
        if not fname.endswith(".jpg"):
            continue
        src_img = os.path.join(src_images, fname)
        dst_img = os.path.join(dst_images, fname)
        if not os.path.exists(dst_img):
            shutil.copy2(os.path.abspath(src_img), dst_img)
        ann_name = fname.replace(".jpg", ".txt")
        ann_path = os.path.join(src_annotations, ann_name)
        if not os.path.exists(ann_path):
            skipped += 1
            continue

        img = cv2.imread(src_img)
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        yolo_lines = []
        with open(ann_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue

                bbox_left = float(parts[0])
                bbox_top = float(parts[1])
                bbox_w = float(parts[2])
                bbox_h = float(parts[3])
                category = int(parts[5])

                if bbox_w <= 0 or bbox_h <= 0 or category < 1 or category > 10:
                    continue

                yolo_cls = category - 1

                x_center = (bbox_left + bbox_w / 2.0) / w
                y_center = (bbox_top + bbox_h / 2.0) / h
                norm_w = bbox_w / w
                norm_h = bbox_h / h

                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                norm_w = min(1, norm_w)
                norm_h = min(1, norm_h)

                yolo_lines.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        dst_label_path = os.path.join(dst_labels, ann_name)
        with open(dst_label_path, "w") as f:
            f.write("\n".join(yolo_lines))

    print(f"转换完成: {src_annotations} -> {dst_labels}")
    print(f"  处理 {len(img_files)} 张图像, 跳过 {skipped} 张")
    return len(img_files) - skipped


def prepare_yolo_dataset():
    base = "visdrone_yolo"
    subsets = {
        "train": {
            "src_img": os.path.join(ARCHIVE_DIR, "VisDrone2019-DET-train",
                                    "VisDrone2019-DET-train", "images"),
            "src_ann": os.path.join(ARCHIVE_DIR, "VisDrone2019-DET-train",
                                    "VisDrone2019-DET-train", "annotations"),
        },
        "val": {
            "src_img": os.path.join(ARCHIVE_DIR, "VisDrone2019-DET-val",
                                    "VisDrone2019-DET-val", "images"),
            "src_ann": os.path.join(ARCHIVE_DIR, "VisDrone2019-DET-val",
                                    "VisDrone2019-DET-val", "annotations"),
        },
    }

    for name, paths in subsets.items():
        dst_img = os.path.join(base, name, "images")
        dst_lbl = os.path.join(base, name, "labels")
        convert_visdrone_to_yolo(paths["src_img"], paths["src_ann"], dst_img, dst_lbl)

    yaml_path = os.path.join(base, "dataset.yaml")
    abs_base = os.path.abspath(base)
    yaml_content = f"""# VisDrone YOLO format dataset
path: {abs_base}
train: train/images
val: val/images

nc: 10
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\ndataset.yaml 已生成: {yaml_path}")
    print(f"数据集路径: {abs_base}")
    return yaml_path


def train_model(epochs, batch, patience, lr):
    yaml_path = prepare_yolo_dataset()

    model = YOLO("yolov8n.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        name="visdrone_yolov8",
        patience=patience,
        lr0=lr,
        device=device,
        workers=8,
        amp=False,
    )
    return results


if __name__ == "__main__":
    EPOCHS = 50
    BATCH = 32
    PATIENCE = 10
    LR = 0.01
    train_model(epochs=EPOCHS, batch=BATCH, patience=PATIENCE, lr=LR)