# U-Net for Cityscapes Semantic Segmentation

A research-oriented PyTorch implementation of **U-Net for semantic segmentation of urban street scenes** using the Cityscapes dataset. The project is designed as a complete academic pipeline: dataset preprocessing, label remapping, model definition, reproducible training, quantitative evaluation, and qualitative visualization.

## Research Objective

Semantic segmentation is a dense prediction task where every pixel of an input image is assigned to a semantic category. In autonomous driving and intelligent transportation systems, this enables road-scene understanding: road, sidewalk, buildings, vehicles, pedestrians, traffic signs, vegetation, and sky.

This project studies the following research question:

> How effectively can a U-Net encoder-decoder architecture segment urban driving scenes when trained with a combined cross-entropy and Dice objective?

## Core Contributions

- U-Net implementation in PyTorch.
- Cityscapes loader with correct conversion from official label IDs to 19 train IDs.
- Ignore-index handling for unlabeled pixels.
- Combined Cross-Entropy + Dice loss.
- Pixel Accuracy, mIoU, and Dice Score evaluation.
- Reproducible training with fixed seeds and checkpoint metadata.
- Qualitative prediction visualization.
- Academic methodology and report templates.

## Project Structure

```text
unet-cityscapes-segmentation/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── methodology.md
│   └── report_template.md
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── unet.py
│   └── visualization.py
├── train.py
└── evaluate.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Download Cityscapes after registration and organize it as:

```text
data/cityscapes/
├── leftImg8bit/
│   ├── train/
│   └── val/
└── gtFine/
    ├── train/
    └── val/
```

The project uses `*_gtFine_labelIds.png` annotation files and converts them internally to the 19 semantic train IDs used for training.

## Training

```bash
python train.py \
  --data-root data/cityscapes \
  --epochs 50 \
  --batch-size 2 \
  --image-size 256 \
  --save-predictions
```

The training script saves:

- `checkpoints/best_model.pth`
- `checkpoints/last_model.pth`
- qualitative predictions in `outputs/`

## Evaluation

```bash
python evaluate.py \
  --data-root data/cityscapes \
  --checkpoint checkpoints/best_model.pth \
  --image-size 256 \
  --num-visualizations 5
```

## Metrics

The model is evaluated using:

| Metric | Meaning |
|---|---|
| Pixel Accuracy | Percentage of correctly classified valid pixels |
| mIoU | Mean Intersection over Union across semantic classes |
| Dice Score | Region-overlap metric useful for class imbalance |

## Suggested Experiments

| Experiment | Image Size | Loss | Purpose |
|---|---:|---|---|
| Baseline | 256 | CE + Dice | Main reproducible result |
| Higher Resolution | 512 | CE + Dice | Better boundaries if GPU memory allows |
| Ablation | 256 | CE only | Measure contribution of Dice loss |
| Longer Training | 256 | CE + Dice | Test convergence behavior |

## Academic Notes

For a strong university-level report, include:

1. Problem definition and mathematical formulation.
2. Dataset description and preprocessing.
3. Architecture explanation with encoder, decoder, bottleneck, and skip connections.
4. Loss function derivation.
5. Quantitative results table.
6. Qualitative predictions.
7. Discussion of class imbalance and failure cases.

## Limitations

- Full-resolution Cityscapes training is computationally expensive.
- Small classes such as poles, traffic signs, riders, and motorcycles are harder to segment.
- U-Net is a strong baseline but newer architectures such as DeepLabV3+, HRNet, and SegFormer may achieve better benchmark performance.

## Future Work

- Add data augmentation.
- Add class-weighted cross-entropy.
- Compare U-Net with DeepLabV3+ or SegFormer.
- Add experiment tracking.
- Add hyperparameter search.
- Train with larger image crops.
