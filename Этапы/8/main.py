from ultralytics import YOLO
import os
import xml.etree.ElementTree as ET

model = YOLO("yolov8n.pt")

dataset = "dataset"
classes = {"apple", "banana", "orange"}

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    return inter / (area_a + area_b - inter) if inter > 0 else 0


def read_xml(path):
    root = ET.parse(path).getroot()
    boxes = []

    for obj in root.findall("object"):
        name = obj.find("name").text
        box = obj.find("bndbox")

        boxes.append([
            name,
            [
                int(box.find("xmin").text),
                int(box.find("ymin").text),
                int(box.find("xmax").text),
                int(box.find("ymax").text)
            ]
        ])

    return boxes


tp = 0
fp = 0
fn = 0

for file in os.listdir(dataset):
    if not file.endswith(".jpg"):
        continue

    img_path = os.path.join(dataset, file)
    xml_path = os.path.join(dataset, file.replace(".jpg", ".xml"))

    gt_boxes = [b for b in read_xml(xml_path) if b[0] in classes]

    result = model(img_path, conf=0.3)[0]
    # result.show()

    pred_boxes = []

    for box in result.boxes:
        cls = model.names[int(box.cls[0])]
        if cls in classes:
            pred_boxes.append([cls, list(map(int, box.xyxy[0]))])

    matched = set()

    for pred_cls, pred_box in pred_boxes:
        best_iou = 0
        best_gt = -1

        for i, (gt_cls, gt_box) in enumerate(gt_boxes):
            if gt_cls != pred_cls:
                continue

            score = iou(pred_box, gt_box)

            if score > best_iou:
                best_iou = score
                best_gt = i

        if best_iou > 0.5 and best_gt not in matched:
            tp += 1
            matched.add(best_gt)
        else:
            fp += 1

    fn += len(gt_boxes) - len(matched)


precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1:", round(f1, 3))