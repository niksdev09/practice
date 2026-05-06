from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

model = YOLO("yolov8n.pt")

base_dir = Path(__file__).resolve().parent
image_path = base_dir / "image.jpg"
results = model(image_path)
results[0].show()
