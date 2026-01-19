import os
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# =========================
# CONFIG
# =========================
IMAGE_DIR = "data/test"
ANN_FILE = "data/test/_annotations.coco.json"
WEIGHT_PATH = "outputs/faster-rcnn/faster-rcnn.pth"
SAVE_DIR = "outputs/faster-rcnn/test"

NUM_CLASSES = 8
SCORE_THRESH = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# MODEL
# =========================
model = fasterrcnn_resnet50_fpn(weights=None, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================
# DATA
# =========================
coco_gt = COCO(ANN_FILE)
img_ids = coco_gt.getImgIds()
tf = transforms.Compose([transforms.ToTensor()])

# =========================
# UTILS
# =========================
def draw_pred_boxes(img, outputs):
    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score < SCORE_THRESH:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"P:{label}:{score:.2f}"
        cv2.putText(img, text, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

def draw_gt_boxes(img, anns):
    for ann in anns:
        x, y, w, h = ann["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

# =========================
# INFERENCE + VIS + COCO EVAL
# =========================
results = []

with torch.no_grad():
    for img_id in tqdm(img_ids, desc="Inference"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(IMAGE_DIR, img_info["file_name"])

        img = cv2.imread(img_path)
        img_tensor = tf(img).to(DEVICE)

        output = model([img_tensor])[0]

        # ---------- collect results for COCO ----------
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score < SCORE_THRESH:
                continue
            x1, y1, x2, y2 = box
            results.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score)
            })

        # ---------- visualization ----------
        draw_pred_boxes(img, output)
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        draw_gt_boxes(img, anns)

        save_path = os.path.join(SAVE_DIR, img_info["file_name"])
        cv2.imwrite(save_path, img)

# =========================
# COCO EVAL (PR + mAP@0.5 + mAP@0.5:0.95)
# =========================
if len(results) == 0:
    raise RuntimeError("No detection results, check model or score threshold!")

coco_dt = coco_gt.loadRes(results)
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# 取指标
mAP50 = coco_eval.stats[0]  # AP @ IoU=0.50
mAP5095 = coco_eval.stats[1]  # AP @ IoU=0.50:0.95
precision = coco_eval.stats[7]  # Precision (averaged over IoU=0.50:0.95)
recall = coco_eval.stats[8]     # Recall (averaged over IoU=0.50:0.95)

print(f"\n✅ mAP@0.5 = {mAP50:.4f}")
print(f"✅ mAP@0.5:0.95 = {mAP5095:.4f}")
print(f"📌 Average Precision = {precision:.4f}")
print(f"📌 Average Recall = {recall:.4f}")
print(f"📁 Visualization saved to: {SAVE_DIR}")
