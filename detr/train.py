# train_detr.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrConfig
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.optim as optim

# =========================
# CONFIG
# =========================
CFG = {
    "img_dir": "data/train",
    "ann_file": "data/train/_annotations.coco.json",
    "epochs": 5,
    "batch_size": 2,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "outputs/detr/detr.pth",
    "pretrained": False  # 是否使用预训练权重
}
os.makedirs(os.path.dirname(CFG["save_path"]), exist_ok=True)

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

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        width, height = img_info["width"], img_info["height"]

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            # 坐标归一化
            x1 = x / width
            y1 = y / height
            x2 = (x + w) / width
            y2 = (y + h) / height
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat2label[ann["category_id"]])

        # 处理空目标图片
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "class_labels": labels
        }

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

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
        config = DetrConfig(
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        model = DetrForObjectDetection(config)

    return model, processor

# =========================
# TRAIN LOOP
# =========================
def train_one_epoch(model, processor, optimizer, loader, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=120)

    for images, targets in pbar:
        # 构建 processor 需要的 batch_annotations
        batch_annotations = []
        for i, t in enumerate(targets):
            ann_list = []
            for box, label in zip(t["boxes"], t["class_labels"]):
                x1, y1, x2, y2 = box.tolist()
                width = x2 - x1
                height = y2 - y1
                ann_list.append({
                    "bbox": [x1, y1, width, height],
                    "category_id": label.item(),
                    "area": width * height  # 关键修复
                })
            batch_annotations.append({
                "image_id": i,
                "annotations": ann_list
            })


        # 使用 processor
        encoding = processor(images=images, annotations=batch_annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in encoding["labels"]]

        # forward
        outputs = model(pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader)


# =========================
# MAIN
# =========================
def main():
    device = torch.device(CFG["device"])

    # 构建数据集
    dataset = CocoDataset(CFG["img_dir"], CFG["ann_file"])
    CFG["num_classes"] = len(dataset.cat_ids)
    print(f"Dataset has {CFG['num_classes']} classes: {dataset.cat_ids}")

    loader = DataLoader(dataset, batch_size=CFG["batch_size"],
                        shuffle=True, collate_fn=collate_fn, num_workers=4)

    # 构建模型
    model, processor = build_model(CFG["num_classes"], pretrained=CFG["pretrained"])
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

    for epoch in range(CFG["epochs"]):
        loss = train_one_epoch(model, processor, optimizer, loader, device, epoch)
        print(f"[Epoch {epoch}] Loss: {loss:.4f}")

    torch.save(model.state_dict(), CFG["save_path"])
    print("Model saved to:", CFG["save_path"])


if __name__ == "__main__":
    main()
