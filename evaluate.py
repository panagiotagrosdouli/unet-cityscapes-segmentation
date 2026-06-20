import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CityscapesDataset
from src.metrics import dice_score, mean_iou, pixel_accuracy
from src.unet import UNet
from src.visualization import save_prediction_grid


def load_checkpoint(model: UNet, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CityscapesDataset(args.data_root, split=args.split, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNet(in_channels=3, num_classes=args.num_classes).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_accuracy = 0.0
    total_miou = 0.0
    total_dice = 0.0
    saved = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            total_accuracy += pixel_accuracy(logits, masks, ignore_index=args.ignore_index)
            total_miou += mean_iou(logits, masks, num_classes=args.num_classes, ignore_index=args.ignore_index)
            total_dice += dice_score(logits, masks, num_classes=args.num_classes, ignore_index=args.ignore_index)

            if saved < args.num_visualizations:
                save_prediction_grid(
                    images[0],
                    masks[0],
                    logits[0],
                    str(output_dir / f"prediction_{saved + 1:03d}.png"),
                    ignore_index=args.ignore_index,
                    title=f"{args.split} sample {saved + 1}",
                )
                saved += 1

    total_accuracy /= len(loader)
    total_miou /= len(loader)
    total_dice /= len(loader)

    print(f"Pixel Accuracy: {total_accuracy:.4f}")
    print(f"Mean IoU: {total_miou:.4f}")
    print(f"Dice Score: {total_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate U-Net on Cityscapes semantic segmentation.")
    parser.add_argument("--data-root", type=str, required=True, help="Path to Cityscapes dataset root.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    parser.add_argument("--num-visualizations", type=int, default=5)
    args = parser.parse_args()

    evaluate(args)
