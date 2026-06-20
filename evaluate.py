import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CityscapesDataset
from src.metrics import mean_iou, pixel_accuracy
from src.unet import UNet


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CityscapesDataset(args.data_root, split="val", image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNet(in_channels=3, num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    total_accuracy = 0.0
    total_miou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            total_accuracy += pixel_accuracy(logits, masks)
            total_miou += mean_iou(logits, masks, num_classes=args.num_classes)

    total_accuracy /= len(loader)
    total_miou /= len(loader)

    print(f"Pixel Accuracy: {total_accuracy:.4f}")
    print(f"Mean IoU: {total_miou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate U-Net on Cityscapes validation set.")
    parser.add_argument("--data-root", type=str, required=True, help="Path to Cityscapes dataset root.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=19)
    args = parser.parse_args()

    evaluate(args)
