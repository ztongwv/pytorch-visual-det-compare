import os
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")  # 忽略 FutureWarning

# =========================
# CONFIG
# =========================
CFG = {
    "train_img_dir": "data/train",
    "train_ann_file": "data/train/_annotations.coco.json",
    "val_img_dir": "data/val",
    "val_ann_file": "data/val/_annotations.coco.json",
    "epochs": 5,
    "batch_size": 2,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "outputs/fasterrcnn"
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
        self.label2cat = {i: cat_id for cat_id, i in self.cat2label.items()}
        self.transform = transforms.ToTensor()  # PIL -> Tensor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat2label[ann["category_id"]])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }
        return img, target

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)

# =========================
# MODEL
# =========================
def build_model(num_classes, pretrained=True):
    # 加载官方 Faster R-CNN
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    # 替换预测器头，适配自定义类别
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# =========================
# TRAIN/VAL LOOP
# =========================
def train_one_epoch(model, optimizer, loader, device, epoch, val_loader=None):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=120)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        pbar.set_postfix(loss=losses.item())

    avg_loss = total_loss / len(loader)

    # 验证
    val_loss = None
    if val_loader is not None:
        model.eval()
        val_total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_total += losses.item()
        val_loss = val_total / len(val_loader)

    return avg_loss, val_loss

# =========================
# MAIN
# =========================
def main():
    device = torch.device(CFG["device"])

    train_dataset = CocoDataset(CFG["train_img_dir"], CFG["train_ann_file"])
    val_dataset = CocoDataset(CFG["val_img_dir"], CFG["val_ann_file"])
    CFG["num_classes"] = len(train_dataset.cat_ids)
    print(f"Dataset has {len(train_dataset)} train images, {len(val_dataset)} val images, "
          f"{CFG['num_classes']} classes: {train_dataset.cat_ids}")

    train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CFG["batch_size"],
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = build_model(CFG["num_classes"], pretrained=False)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

    train_losses = []
    val_losses = []

    for epoch in range(CFG["epochs"]):
        train_loss, val_loss = train_one_epoch(model, optimizer, train_loader,
                                               device, epoch, val_loader)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 保存模型
        torch.save(model.state_dict(), os.path.join(CFG["save_dir"], f"fasterrcnn_epoch{epoch}.pth"))

    # 绘制训练曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(CFG["epochs"]), train_losses, label="Train Loss")
    plt.plot(range(CFG["epochs"]), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CFG["save_dir"], "loss_curve.png"))
    print("Training complete. Model and loss curve saved.")

if __name__ == "__main__":
    main()
