from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


CITYSCAPES_COLORS = np.array([
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32],    # bicycle
], dtype=np.uint8)


def decode_segmentation_mask(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """Convert a train-ID mask to an RGB color image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in enumerate(CITYSCAPES_COLORS):
        color_mask[mask == class_id] = color

    color_mask[mask == ignore_index] = np.array([0, 0, 0], dtype=np.uint8)
    return color_mask


def save_prediction_grid(
    image: torch.Tensor,
    target: torch.Tensor,
    logits: torch.Tensor,
    output_path: str,
    ignore_index: int = 255,
    title: Optional[str] = None,
) -> None:
    """Save a qualitative comparison of image, ground truth and prediction."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    image_np = image.detach().cpu().numpy().transpose(1, 2, 0)
    image_np = np.clip(image_np, 0.0, 1.0)

    target_np = target.detach().cpu().numpy()
    prediction_np = torch.argmax(logits, dim=0).detach().cpu().numpy()

    target_rgb = decode_segmentation_mask(target_np, ignore_index=ignore_index)
    prediction_rgb = decode_segmentation_mask(prediction_np, ignore_index=ignore_index)

    plt.figure(figsize=(12, 4))
    if title:
        plt.suptitle(title)

    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Input image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(target_rgb)
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction_rgb)
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
