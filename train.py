import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CityscapesDataset
from src.losses import CombinedSegmentationLoss
from src.metrics import dice_score, mean_iou, pixel_accuracy
from src.unet import UNet
from src.visualization import save_prediction_grid


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = CityscapesDataset(args.data_root, split="train", image_size=args.image_size)
    val_dataset = CityscapesDataset(args.data_root, split="val", image_size=args.image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(in_channels=3, num_classes=args.num_classes).to(device)
    criterion = CombinedSegmentationLoss(num_classes=args.num_classes, ignore_index=args.ignore_index)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_miou = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        model.eval()
        val_accuracy = 0.0
        val_miou = 0.0
        val_dice = 0.0
        first_batch_saved = False

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                val_accuracy += pixel_accuracy(logits, masks, ignore_index=args.ignore_index)
                val_miou += mean_iou(logits, masks, num_classes=args.num_classes, ignore_index=args.ignore_index)
                val_dice += dice_score(logits, masks, num_classes=args.num_classes, ignore_index=args.ignore_index)

                if args.save_predictions and not first_batch_saved:
                    save_prediction_grid(
                        images[0],
                        masks[0],
                        logits[0],
                        str(output_dir / f"epoch_{epoch + 1:03d}_prediction.png"),
                        ignore_index=args.ignore_index,
                        title=f"Epoch {epoch + 1}",
                    )
                    first_batch_saved = True

        train_loss /= len(train_loader)
        val_accuracy /= len(val_loader)
        val_miou /= len(val_loader)
        val_dice /= len(val_loader)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_pixel_acc={val_accuracy:.4f}, "
            f"val_mIoU={val_miou:.4f}, "
            f"val_dice={val_dice:.4f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou": best_miou,
            "args": vars(args),
        }
        torch.save(checkpoint, checkpoint_dir / "last_model.pth")

        if val_miou > best_miou:
            best_miou = val_miou
            checkpoint["best_miou"] = best_miou
            torch.save(checkpoint, checkpoint_dir / "best_model.pth")
            print("Saved best model checkpoint.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net on Cityscapes semantic segmentation.")
    parser.add_argument("--data-root", type=str, required=True, help="Path to Cityscapes dataset root.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    train(args)
