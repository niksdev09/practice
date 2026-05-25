from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    x1_roi, y1_roi = w // 4, h // 4
    x2_roi, y2_roi = 3 * w // 4, 3 * h // 4

    result = model(frame, conf=0.5)[0]
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
    cv2.putText(frame, f"objects in ROI: {count_person}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("ROI", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
