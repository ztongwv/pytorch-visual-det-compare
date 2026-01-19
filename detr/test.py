import os
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from transformers import DetrImageProcessor, DetrForObjectDetection
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")  # 忽略 FutureWarning

# =========================
# CONFIG
# =========================
CFG = {
    "test_img_dir": "data/val",
    "test_ann_file": "data/val/_annotations.coco.json",
    "batch_size": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "outputs/detr/detr_epoch4.pth",  # 加载你训练好的权重
    "save_dir": "outputs/detr/test"
}
os.makedirs(CFG["save_dir"], exist_ok=True)

# =========================
# DATASET
# =========================
class CocoDataset(Dataset):
    def __init__(self, img_dir, ann_file):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.label2cat = {v: k for k, v in self.cat2label.items()}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img_info["width"], img_info["height"]

        target = {"image_id": img_id, "annotations": []}
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            x, y, w, h = ann["bbox"]
            target["annotations"].append({
                "bbox": [x / img_width, y / img_height, w / img_width, h / img_height],
                "category_id": ann["category_id"],
                "area": ann.get("area", w * h),
                "iscrowd": ann.get("iscrowd", 0)
            })
        return img, target

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)

# =========================
# MODEL
# =========================
def build_model(num_classes, pretrained=False):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    if pretrained:
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_classes
        )
    else:
        from transformers import DetrConfig
        config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
        config.num_labels = num_classes
        model = DetrForObjectDetection(config)
    return model, processor

# =========================
# DRAW DETECTIONS
# =========================
def draw_predictions(image: Image.Image, boxes, labels, scores, dataset: CocoDataset):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cat_id = dataset.label2cat[label.item()]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{cat_id}:{score:.2f}", fill="yellow", font=font)
    return image

# =========================
# EVALUATION
# =========================
def evaluate(model, processor, loader, coco_gt, dataset, device, score_thresh=0.01):
    model.eval()
    results = []

    for images, targets in tqdm(loader, desc="Evaluating", ncols=120):
        encoding = processor(images=images, return_tensors="pt")
        pixel_values = encoding["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        logits = outputs.logits  # [B, num_queries, num_classes]
        pred_boxes = outputs.pred_boxes  # [B, num_queries, 4], normalized

        probs = logits.softmax(-1)
        scores, labels = probs[..., :-1].max(-1)  # 忽略最后背景类

        for i in range(len(images)):
            img = images[i]
            img_id = targets[i]["image_id"]
            boxes = pred_boxes[i]
            lbls = labels[i]
            scrs = scores[i]

            # 过滤低置信度
            mask = scrs > score_thresh
            boxes_filtered = boxes[mask].cpu().numpy()
            lbls_filtered = lbls[mask].cpu().numpy()
            scrs_filtered = scrs[mask].cpu().numpy()

            # 转为像素坐标
            w, h = img.size
            boxes_px = []
            for b in boxes_filtered:
                x_c, y_c, bw, bh = b
                x1 = (x_c - bw / 2) * w
                y1 = (y_c - bh / 2) * h
                x2 = (x_c + bw / 2) * w
                y2 = (y_c + bh / 2) * h
                boxes_px.append([x1, y1, x2, y2])

            # 保存检测图片
            img_draw = draw_predictions(img.copy(), boxes_px, lbls_filtered, scrs_filtered, dataset)
            img_draw.save(os.path.join(CFG["save_dir"], f"{img_id}.jpg"))

            # 转 COCO 结果格式
            for box, label, score in zip(boxes_filtered, lbls_filtered, scrs_filtered):
                results.append({
                    "image_id": img_id,
                    "category_id": int(dataset.label2cat[label]),
                    "bbox": [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                    "score": float(score)
                })

    # 写入 JSON
    import json
    with open(os.path.join(CFG["save_dir"], "results.json"), "w") as f:
        json.dump(results, f)

    # 计算 COCO mAP
    from pycocotools.cocoeval import COCOeval
    coco_dt = coco_gt.loadRes(os.path.join(CFG["save_dir"], "results.json"))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()  # 输出 P, R, mAP@50, mAP@50-95

# =========================
# MAIN
# =========================
def main():
    device = torch.device(CFG["device"])
    test_dataset = CocoDataset(CFG["test_img_dir"], CFG["test_ann_file"])
    test_loader = DataLoader(test_dataset, batch_size=CFG["batch_size"],
                             shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"Test dataset has {len(test_dataset)} images, {len(test_dataset.cat_ids)} classes: {test_dataset.cat_ids}")

    model, processor = build_model(len(test_dataset.cat_ids), pretrained=False)
    model.load_state_dict(torch.load(CFG["model_path"], map_location=device))
    model.to(device)

    coco_gt = COCO(CFG["test_ann_file"])
    evaluate(model, processor, test_loader, coco_gt, test_dataset, device)

if __name__ == "__main__":
    main()
