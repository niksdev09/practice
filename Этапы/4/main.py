from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")

target_class = "person"
image_name = "image.jpg"


# img = cv2.imread(image_name)
# result = model(img)[0]
target_id = [k for k, v in model.names.items() if v == target_class][0]

# for box in result.boxes:
#     cls = int(box.cls[0])
#     if cls != target_id:
#         continue
#     x1, y1, x2, y2 = map(int, box.xyxy[0])
#     conf = float(box.conf[0])
#     label = f"{model.names[cls]} {conf:.2f}"
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# output_path = "image_result.jpg"
# cv2.imwrite(output_path, img)


###### 2 

target_names = {"person"}
target_ids = [cls_id for cls_id, name in model.names.items() if name in target_names]

results = model(image_name, classes=target_ids)

results[0].show()