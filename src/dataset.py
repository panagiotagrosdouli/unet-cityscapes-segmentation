from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


CITYSCAPES_ID_TO_TRAIN_ID = {
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]


def convert_label_ids_to_train_ids(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """Convert official Cityscapes label IDs to 19 train IDs.

    Cityscapes annotation files contain label IDs that are not contiguous.
    CrossEntropyLoss requires class indices in [0, num_classes - 1], therefore
    ignored/unlabeled pixels are assigned to ignore_index.
    """
    converted = np.full(mask.shape, ignore_index, dtype=np.uint8)
    for label_id, train_id in CITYSCAPES_ID_TO_TRAIN_ID.items():
        converted[mask == label_id] = train_id
    return converted


def apply_training_augmentation(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply lightweight augmentations suitable for semantic segmentation.

    Geometric transforms are applied identically to the image and mask, while
    photometric transforms are applied only to the image to preserve labels.
    """
    if np.random.rand() < 0.5:
        image = np.ascontiguousarray(np.flip(image, axis=1))
        mask = np.ascontiguousarray(np.flip(mask, axis=1))

    if np.random.rand() < 0.5:
        alpha = np.random.uniform(0.85, 1.15)  # contrast
        beta = np.random.uniform(-15.0, 15.0)  # brightness
        image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    return image, mask


class CityscapesDataset(Dataset):
    """Dataset loader for Cityscapes semantic segmentation.

    Expected structure:
    data_root/leftImg8bit/split/city/*_leftImg8bit.png
    data_root/gtFine/split/city/*_gtFine_labelIds.png
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 256,
        ignore_index: int = 255,
        augment: bool = False,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.ignore_index = ignore_index
        self.augment = augment

        image_dir = self.data_root / "leftImg8bit" / split
        self.image_paths = sorted(image_dir.glob("*/*_leftImg8bit.png"))

        if not self.image_paths:
            raise FileNotFoundError(
                f"No Cityscapes images found in {image_dir}. Check the dataset path."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _mask_path_from_image_path(self, image_path: Path) -> Path:
        city = image_path.parent.name
        filename = image_path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        return self.data_root / "gtFine" / self.split / city / filename

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        mask_path = self._mask_path_from_image_path(image_path)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        if self.augment:
            image, mask = apply_training_augmentation(image, mask)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = convert_label_ids_to_train_ids(mask, ignore_index=self.ignore_index)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        image_tensor = torch.tensor(image, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.long)

        return image_tensor, mask_tensor
