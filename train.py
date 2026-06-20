import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CityscapesDataset
from src.losses import CombinedSegmentationLoss
from src.metrics import mean_iou, pixel_accuracy
from src.unet import UNet


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = CityscapesDataset(args.data_root, split="train", image_size=args.image_size)
    val_dataset = CityscapesDataset(args.data_root, split="val", image_size=args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNet(in_channels=3, num_classes=args.num_classes).to(device)
    criterion = CombinedSegmentationLoss(num_classes=args.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    best_miou = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        model.eval()
        val_accuracy = 0.0
        val_miou = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                val_accuracy += pixel_accuracy(logits, masks)
                val_miou += mean_iou(logits, masks, num_classes=args.num_classes)

        train_loss /= len(train_loader)
        val_accuracy /= len(val_loader)
        val_miou /= len(val_loader)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_pixel_acc={val_accuracy:.4f}, "
            f"val_mIoU={val_miou:.4f}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            print("Saved best model checkpoint.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net on Cityscapes semantic segmentation.")
    parser.add_argument("--data-root", type=str, required=True, help="Path to Cityscapes dataset root.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=19)
    args = parser.parse_args()

    train(args)
