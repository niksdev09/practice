from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

target_names = {"person"}
target_ids = [cls_id for cls_id, name in model.names.items() if name in target_names]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    x1_roi, y1_roi = w // 4, h // 4
    x2_roi, y2_roi = 3 * w // 4, 3 * h // 4

    result = model(frame, classes=target_ids, conf=0.5)[0]
    count_person = 0

    for box in result.boxes:
        cls = int(box.cls[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if x1_roi <= cx <= x2_roi and y1_roi <= cy <= y2_roi:
            count_person += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 0, 0), 2)
    cv2.putText(frame, f"person in ROI: {count_person}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("ROI Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
