import torch


def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 255) -> float:
    """Compute pixel accuracy over valid pixels only."""
    predictions = torch.argmax(logits, dim=1)
    valid = targets != ignore_index

    if valid.sum().item() == 0:
        return 0.0

    correct = (predictions[valid] == targets[valid]).sum().item()
    total = valid.sum().item()
    return correct / total


def mean_iou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 19, ignore_index: int = 255) -> float:
    """Compute mean Intersection over Union over valid semantic classes."""
    predictions = torch.argmax(logits, dim=1)
    valid = targets != ignore_index
    iou_values = []

    for class_id in range(num_classes):
        pred_class = (predictions == class_id) & valid
        target_class = (targets == class_id) & valid

        intersection = torch.logical_and(pred_class, target_class).sum().item()
        union = torch.logical_or(pred_class, target_class).sum().item()

        if union > 0:
            iou_values.append(intersection / union)

    if not iou_values:
        return 0.0

    return sum(iou_values) / len(iou_values)


def dice_score(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 19, ignore_index: int = 255) -> float:
    """Compute mean Dice score over valid semantic classes."""
    predictions = torch.argmax(logits, dim=1)
    valid = targets != ignore_index
    dice_values = []

    for class_id in range(num_classes):
        pred_class = (predictions == class_id) & valid
        target_class = (targets == class_id) & valid

        intersection = torch.logical_and(pred_class, target_class).sum().item()
        denominator = pred_class.sum().item() + target_class.sum().item()

        if denominator > 0:
            dice_values.append((2.0 * intersection) / denominator)

    if not dice_values:
        return 0.0

    return sum(dice_values) / len(dice_values)
