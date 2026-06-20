# Experimental Protocol

## Aim

The aim is to evaluate a U-Net semantic segmentation model on urban street-scene understanding using the Cityscapes dataset.

## Primary Hypothesis

A U-Net model trained with a combined cross-entropy and Dice objective will produce better region overlap than a model trained with cross-entropy alone, especially under class imbalance.

## Dataset Split

Use the official Cityscapes split:

- Training: `leftImg8bit/train` and `gtFine/train`
- Validation: `leftImg8bit/val` and `gtFine/val`

The test set should not be used for model selection.

## Preprocessing

1. Resize RGB images to the selected square input size.
2. Resize masks with nearest-neighbor interpolation.
3. Convert official Cityscapes label IDs to 19 train IDs.
4. Assign unlabeled or ignored pixels to `ignore_index = 255`.

## Baseline Experiment

```bash
python train.py --data-root data/cityscapes --epochs 50 --batch-size 2 --image-size 256 --save-predictions
```

## Evaluation

```bash
python evaluate.py --data-root data/cityscapes --checkpoint checkpoints/best_model.pth --image-size 256 --num-visualizations 5
```

## Metrics

Report:

- Pixel Accuracy
- Mean Intersection over Union
- Dice Score

mIoU should be treated as the primary metric.

## Ablation Studies

Recommended ablations:

1. Cross-entropy only.
2. Cross-entropy plus Dice loss.
3. Different image sizes, such as 256 and 512.
4. Different learning rates, such as 1e-3 and 1e-4.

## Qualitative Analysis

For each experiment, include visual examples:

- input image
- ground-truth semantic mask
- predicted semantic mask

Discuss common failure cases, especially for small classes such as pole, traffic sign, rider, motorcycle, and bicycle.

## Reporting Table

| Experiment | Image Size | Epochs | Loss | Pixel Accuracy | mIoU | Dice Score |
|---|---:|---:|---|---:|---:|---:|
| Baseline | 256 | 50 | CE + Dice | TBD | TBD | TBD |
| CE only | 256 | 50 | CE | TBD | TBD | TBD |
| Higher resolution | 512 | 50 | CE + Dice | TBD | TBD | TBD |

## Reproducibility Checklist

- Fixed random seed.
- Dataset split stated clearly.
- Image size reported.
- Loss function reported.
- Optimizer and learning rate reported.
- Number of epochs reported.
- Checkpoint selection criterion reported.
- Quantitative and qualitative results included.
