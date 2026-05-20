from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("video.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("video_result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame)[0]
    annotated = result.plot()
    out.write(annotated)

cap.release()
out.release()
