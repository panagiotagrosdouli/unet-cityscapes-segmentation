from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CityscapesDataset(Dataset):
    """Dataset loader for Cityscapes images and semantic masks.

    It expects the standard Cityscapes folder structure:
    data_root/leftImg8bit/split/city/*_leftImg8bit.png
    data_root/gtFine/split/city/*_gtFine_labelIds.png
    """

    def __init__(self, data_root: str, split: str = "train", image_size: int = 256):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        image_tensor = torch.tensor(image, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.long)

        return image_tensor, mask_tensor
