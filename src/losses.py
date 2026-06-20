import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class Dice loss for semantic segmentation."""

    def __init__(self, num_classes: int = 19, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probabilities * targets_one_hot, dims)
        cardinality = torch.sum(probabilities + targets_one_hot, dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()


class CombinedSegmentationLoss(nn.Module):
    """Combination of cross-entropy loss and Dice loss."""

    def __init__(self, num_classes: int = 19, dice_weight: float = 0.5):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.cross_entropy(logits, targets)
        dice_loss = self.dice(logits, targets)
        return ce_loss + self.dice_weight * dice_loss
