from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")
model = YOLO("yolov8n.pt")

results = model("image.jpg")

for box in results[0].boxes:
    print(box.cls)
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    coords = box.xyxy[0].tolist()

    # print(model.names[cls], conf, coords)
