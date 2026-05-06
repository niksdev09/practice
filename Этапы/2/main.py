from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

model = YOLO("yolov8n.pt")

base_dir = Path(__file__).resolve().parent
image_path = base_dir / "image.jpg"
results = model(image_path)

boxes = results[0].boxes

print(f"{'class':15} {'conf':10} {'x_min':8} {'y_min':8} {'x_max':8} {'y_max':8}")
for box in boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    name = model.names[cls]
    print(f"{name:15} {conf:<10.2f} {x1:<8} {y1:<8} {x2:<8} {y2:<8}")
