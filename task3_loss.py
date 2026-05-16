import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(1)
        pred_softmax = F.softmax(pred, dim=1)

        target_one_hot = F.one_hot(target, num_classes=n_classes).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            mask = (target != self.ignore_index).unsqueeze(1).float()
            pred_softmax = pred_softmax * mask
            target_one_hot = target_one_hot * mask

        intersection = (pred_softmax * target_one_hot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_per_class.mean()

        return dice_loss


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target)
        return (1 - self.alpha) * ce_loss + self.alpha * dice_loss