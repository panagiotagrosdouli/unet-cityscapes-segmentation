# U-Net for Cityscapes Semantic Segmentation

This project implements a U-Net based deep learning pipeline for semantic segmentation of urban street scenes. The target dataset is **Cityscapes**, a benchmark dataset for pixel-level urban scene understanding.

## Objective

The goal is to classify each pixel of an input road-scene image into a semantic class such as road, sidewalk, building, vehicle, pedestrian, vegetation, and sky. This task is important in computer vision applications such as autonomous driving and intelligent transportation systems.

## Methodology

The project uses a **U-Net encoder-decoder architecture**. The encoder extracts high-level contextual features, while the decoder progressively restores spatial resolution. Skip connections transfer fine-grained spatial information from the encoder to the decoder, improving boundary localization.

## Project Structure

```text
unet-cityscapes-segmentation/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   └── unet.py
├── train.py
└── evaluate.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Dataset

Download the Cityscapes dataset from the official website after registration:

- leftImg8bit images
- gtFine annotations

Expected folder format:

```text
data/cityscapes/
├── leftImg8bit/
│   ├── train/
│   └── val/
└── gtFine/
    ├── train/
    └── val/
```

## Training

```bash
python train.py --data-root data/cityscapes --epochs 20 --batch-size 2 --image-size 256
```

## Evaluation

```bash
python evaluate.py --data-root data/cityscapes --checkpoint checkpoints/best_model.pth --image-size 256
```

## Metrics

The model is evaluated using:

- Pixel Accuracy
- Mean Intersection over Union (mIoU)
- Dice Score

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- OpenCV
- Matplotlib

## Notes

Cityscapes images are high resolution, so resizing or cropping is recommended when training on limited hardware.
