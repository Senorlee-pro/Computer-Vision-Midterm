import os
import copy

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import wandb

from task3_unet import UNet
from task3_loss import DiceLoss, CombinedLoss

class OxfordPetDataset(Dataset):
    TRIMAP_MAPPING = {1: 1, 2: 0, 3: 2}  # -> 0:bg, 1:pet, 2:boundary

    def __init__(self, root: str, split: str = "train", size: int = 256):
        self.root = root
        self.size = size

        img_dir = os.path.join(root, "images")
        all_images = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith(".jpg")
        ])

        # train/val/test 划分: 70% / 15% / 15%
        n = len(all_images)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        if split == "train":
            self.images = all_images[:train_end]
        elif split == "val":
            self.images = all_images[train_end:val_end]
        else:
            self.images = all_images[val_end:]

        self.img_dir = img_dir
        self.trimap_dir = os.path.join(root, "annotations", "trimaps")
        self.img_transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        trimap_name = name.replace(".jpg", ".png")
        trimap = Image.open(os.path.join(self.trimap_dir, trimap_name))

        img = TF.resize(img, (self.size, self.size), interpolation=TF.InterpolationMode.BILINEAR)
        trimap = TF.resize(trimap, (self.size, self.size), interpolation=TF.InterpolationMode.NEAREST)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])

        trimap_np = np.array(trimap, dtype=np.int64)
        label = np.zeros_like(trimap_np, dtype=np.int64)
        for src, dst in self.TRIMAP_MAPPING.items():
            label[trimap_np == src] = dst

        label = torch.from_numpy(label).long()

        return img, label

def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                 ignore_index: int = 255) -> dict:
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)

    ious = {}
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union > 0:
            ious[cls] = (intersection / union).item()
        else:
            ious[cls] = float("nan")

    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    miou = np.mean(valid_ious) if valid_ious else 0.0

    return {"per_class": ious, "miou": miou}

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, num_classes, device):
    model.eval()
    total_loss = 0.0
    all_mious = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)

        metrics = compute_miou(outputs, labels, num_classes)
        all_mious.append(metrics["miou"])

    avg_loss = total_loss / len(loader.dataset)
    avg_miou = np.mean(all_mious)

    return avg_loss, avg_miou


def run_experiment(loss_name: str, criterion, train_loader, val_loader, test_loader,
                   device, num_classes=3, epochs=30, lr=1e-3, weight_decay=1e-4):

    wandb.init(project="cv-task3-pet-segmentation", name=f"{loss_name}_wd", reinit=True,
               config={"loss": loss_name, "lr": lr, "epochs": epochs})

    model = UNet(n_channels=3, n_classes=num_classes, bilinear=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_miou = 0.0
    best_weights = None
    history = {"train_loss": [], "val_loss": [], "val_miou": []}

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_miou = validate(model, val_loader, criterion, num_classes, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mIoU": val_miou,
        })

        if val_miou > best_miou:
            best_miou = val_miou
            best_weights = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val mIoU: {val_miou:.4f}")

    model.load_state_dict(best_weights)

    test_loss, test_miou = validate(model, test_loader, criterion, num_classes, device)

    wandb.log({"best_val_mIoU": best_miou, "test_loss": test_loss, "test_mIoU": test_miou})
    wandb.finish()
    print(f"最佳 Val mIoU ({loss_name}): {best_miou:.4f}")
    print(f"测试集 mIoU  ({loss_name}): {test_miou:.4f}")

    return model, best_miou, history, test_miou


if __name__ == "__main__":
    DATASET_ROOT = "./oxford-iiit-pet"
    IMG_SIZE = 256
    BATCH_SIZE = 16
    NUM_CLASSES = 3
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_set = OxfordPetDataset(DATASET_ROOT, split="train", size=IMG_SIZE)
    val_set = OxfordPetDataset(DATASET_ROOT, split="val", size=IMG_SIZE)
    test_set = OxfordPetDataset(DATASET_ROOT, split="test", size=IMG_SIZE)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=8, pin_memory=True)

    print(f"训练集: {len(train_set)} 张 | 验证集: {len(val_set)} 张 | 测试集: {len(test_set)} 张")

    loss_configs = {
        "CE": nn.CrossEntropyLoss(),
        "Dice": DiceLoss(smooth=1e-5),
        "CE+Dice": CombinedLoss(alpha=0.5, smooth=1e-5),
    }

    histories = {}
    val_results = {}
    test_results = {}

    for name, criterion in loss_configs.items():
        model, best_val_miou, history, test_miou = run_experiment(
            loss_name=name,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=DEVICE,
            num_classes=NUM_CLASSES,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        histories[name] = history
        val_results[name] = best_val_miou
        test_results[name] = test_miou

        torch.save(model.state_dict(), f"unet_best_{name.replace('+', '_')}.pth")

    print(f"{'='*60}")
    print(f"  {'Loss':10s}  {'Val mIoU':>10s}  {'Test mIoU':>10s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}")
    for name in loss_configs:
        print(f"  {name:10s}  {val_results[name]:10.4f}  {test_results[name]:10.4f}")

