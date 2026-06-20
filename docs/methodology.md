# Methodology

## Problem Definition

Semantic segmentation is formulated as dense pixel-wise classification. Given an input RGB image `x` with spatial dimensions `H x W`, the objective is to estimate a label map `y` where each pixel belongs to one of `C = 19` semantic classes used in the Cityscapes benchmark.

The model learns a function:

```text
f_theta: R^(3 x H x W) -> R^(C x H x W)
```

where `theta` are the trainable parameters and the output is a class-logit tensor for every pixel.

## Dataset

The project targets the Cityscapes dataset, which contains high-resolution street-scene images and pixel-level semantic annotations. The training protocol uses the official train/validation split.

Important preprocessing detail: Cityscapes annotation IDs are not contiguous class indices. Therefore, labels are converted from official label IDs to the 19 train IDs before being passed to the loss function.

## Architecture

The implemented model is a U-Net. It consists of:

1. Encoder path: repeated convolutional blocks and max pooling for hierarchical feature extraction.
2. Bottleneck: deepest representation with the largest receptive field.
3. Decoder path: transposed convolutions for spatial upsampling.
4. Skip connections: concatenation of encoder features with decoder features to preserve boundary information.

This architecture is suitable for semantic segmentation because it combines global semantic context with local spatial precision.

## Optimization Objective

The training objective combines cross-entropy loss and Dice loss:

```text
L = L_CE + lambda * L_Dice
```

Cross-entropy improves pixel-wise classification, while Dice loss addresses class imbalance by directly optimizing region overlap.

## Evaluation Metrics

The project reports:

- Pixel Accuracy
- Mean Intersection over Union (mIoU)
- Dice Score

mIoU is the most important metric because it penalizes both false positives and false negatives at the region level.

## Reproducibility

The training script includes:

- fixed random seed
- deterministic CuDNN configuration
- checkpoint saving
- stored training arguments
- qualitative prediction visualizations

## Suggested Experimental Protocol

A strong academic experiment should include at least three runs:

1. Baseline U-Net with image size 256.
2. U-Net with larger image size, for example 512, if hardware allows it.
3. Ablation without Dice loss or with different Dice weights.

The final report should compare quantitative metrics and include qualitative segmentation examples.
