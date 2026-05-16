import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import OxfordIIITPet
from sklearn.metrics import accuracy_score
import timm
import copy
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


trainval_dataset = OxfordIIITPet(
    root=".",
    split="trainval",
    download=True,
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
)

test_dataset = OxfordIIITPet(
    root=".",
    split="test",
    download=True,
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
)

train_size = int(0.8 * len(trainval_dataset))
val_size = len(trainval_dataset) - train_size
train_dataset, val_dataset = random_split(
    trainval_dataset, [train_size, val_size]
)

val_dataset.dataset.transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")


def build_model(pretrained=True, num_classes=37):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def build_vit_tiny(pretrained=True, num_classes=37):
    model = timm.create_model(
        "vit_tiny_patch16_224.augreg_in21k",
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def create_optimizer(model, fc_lr=1e-3, backbone_lr=1e-4, head_name="fc",
                     weight_decay=0.0):
    head = getattr(model, head_name)
    head_params = list(head.parameters())
    backbone_params = [
        p for n, p in model.named_parameters() if head_name not in n
    ]
    return optim.Adam([
        {"params": head_params, "lr": fc_lr},
        {"params": backbone_params, "lr": backbone_lr}
    ], weight_decay=weight_decay)


def run_experiment(model, train_loader, val_loader, test_loader,
                   fc_lr=1e-3, backbone_lr=1e-4, epochs=10,
                   label="experiment", device="cpu", head_name="fc",
                   weight_decay=0.0, use_scheduler=False,
                   scheduler_type="cosine"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, fc_lr, backbone_lr, head_name,
                                 weight_decay=weight_decay)

    scheduler = None
    scheduler_after_step = False
    if use_scheduler:
        if scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
        elif scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=3
            )
            scheduler_after_step = True
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    wandb.init(project="cv-task1-pet-classification", name=label, reinit=True,
               config={
                   "fc_lr": fc_lr, "backbone_lr": backbone_lr, "epochs": epochs,
                   "weight_decay": weight_decay, "use_scheduler": use_scheduler,
                   "scheduler_type": scheduler_type,
               })

    best_val_acc = 0.0
    best_model_state = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            if scheduler_after_step:
                scheduler.step(val_acc)
            else:
                scheduler.step()

        current_lrs = [pg["lr"] for pg in optimizer.param_groups]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "head_lr": current_lrs[0],
            "backbone_lr": current_lrs[1],
        })

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {current_lrs[0]:.1e}/{current_lrs[1]:.1e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())


    model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    wandb.log({"test_acc": test_acc, "test_loss": test_loss, "best_val_acc": best_val_acc})
    wandb.finish()

    print(f"[{label}] Best Val Acc: {best_val_acc:.4f} | "
          f"Test Acc: {test_acc:.4f}")
    print("-" * 60)

    return {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "history": history,
        "model_state": best_model_state,
    }

if __name__ == "__main__":
    BATCH_SIZE = 64
    FC_LR = 1e-3
    BACKBONE_LR = 1e-4
    EPOCHS = 15
    PRETRAINED = True
    LABEL = "Baseline"
    WEIGHT_DECAY = 1e-4
    USE_SCHEDULER = True
    WEIGHT_NAME = "best"

    MODEL = "resnet" # or "vit"

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    if MODEL == "resnet":
        model = build_model(pretrained=PRETRAINED)
        result = run_experiment(
            model, train_loader, val_loader, test_loader,
            fc_lr=FC_LR, backbone_lr=BACKBONE_LR, epochs=EPOCHS,
            label=LABEL, device=device,
            weight_decay=WEIGHT_DECAY, use_scheduler=USE_SCHEDULER, scheduler_type="cosine",
        )
    elif MODEL == "vit":
        model_vit = build_vit_tiny(pretrained=PRETRAINED)
        result = run_experiment(
            model_vit, train_loader, val_loader, test_loader,
            fc_lr=FC_LR, backbone_lr=BACKBONE_LR, epochs=EPOCHS,
            label=LABEL, device=device, head_name="head",
            weight_decay=WEIGHT_DECAY, use_scheduler=USE_SCHEDULER, scheduler_type="cosine",
        )
    else:
        raise NotImplementedError
    
    torch.save(result["model_state"], f"{WEIGHT_NAME}.pth")


