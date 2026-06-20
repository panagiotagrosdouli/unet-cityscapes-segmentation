import torch


def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute pixel accuracy."""
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total


def mean_iou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 19) -> float:
    """Compute mean Intersection over Union."""
    predictions = torch.argmax(logits, dim=1)
    iou_values = []

    for class_id in range(num_classes):
        pred_class = predictions == class_id
        target_class = targets == class_id

        intersection = torch.logical_and(pred_class, target_class).sum().item()
        union = torch.logical_or(pred_class, target_class).sum().item()

        if union > 0:
            iou_values.append(intersection / union)

    if not iou_values:
        return 0.0

    return sum(iou_values) / len(iou_values)
