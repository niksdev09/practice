from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")
model = YOLO("yolov8n.pt")

thresholds = [0.1, 0.3, 0.5, 0.7]

for conf_thr in thresholds:
    result = model("image.jpg", conf=conf_thr, verbose=False)[0]
    print(f"conf={conf_thr}: detections={len(result.boxes)}")
    result.show()
