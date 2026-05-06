from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame)[0]
    counts = {}

    for box in result.boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        counts[name] = counts.get(name, 0) + 1

    text = ", ".join([f"{k}: {v}" for k, v in counts.items()])
    frame = result.plot()
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Yolo8", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()