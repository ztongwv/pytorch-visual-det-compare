import os
import time
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm

# =========================
# CONFIG
# =========================
CFG = {
    "img_dir": "data/train",
    "ann_file": "data/train/_annotations.coco.json",
    "num_classes": 8,
    "epochs": 3,
    "batch_size": 4,
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# =========================
# DATASET (official style)
# =========================
class CocoDetection(Dataset):
    def __init__(self, img_dir, ann_file):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        area = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            area.append(ann["area"])
            iscrowd.append(ann["iscrowd"])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.tensor(area, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }

        img = F.to_tensor(img)
        return img, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return tuple(zip(*batch))


# =========================
# MODEL
# =========================
def build_model(num_classes):
    # model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )
    return model


# =========================
# TRAIN / EVAL
# =========================
def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    losses_total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=120)

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        losses_total += losses.item()
        pbar.set_postfix(loss=losses.item())

    return losses_total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses_total = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses_total += losses.item()

    return losses_total / len(loader)


# =========================
# MAIN
# =========================
def main():
    device = torch.device(CFG["device"])

    dataset = CocoDetection(CFG["img_dir"], CFG["ann_file"])
    loader = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    model = build_model(CFG["num_classes"])
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=CFG["lr"],
        momentum=CFG["momentum"],
        weight_decay=CFG["weight_decay"],
    )

    for epoch in range(CFG["epochs"]):
        loss = train_one_epoch(model, optimizer, loader, device, epoch)
        print(f"[Epoch {epoch}] Loss: {loss:.4f}")

    # ✅ 检查输出目录是否存在，不存在就创建
    save_dir = "outputs/faster-rcnn"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "faster-rcnn.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")


if __name__ == "__main__":
    main()
