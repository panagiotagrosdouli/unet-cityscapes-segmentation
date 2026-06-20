# U-Net for Semantic Segmentation of Urban Street Scenes

## Abstract

This project investigates semantic segmentation of urban driving scenes using a U-Net convolutional neural network. The model is trained and evaluated on the Cityscapes dataset, targeting 19 semantic classes. The study focuses on dense pixel-wise classification, class imbalance, and reproducible evaluation using pixel accuracy, mean Intersection over Union, and Dice score.

## 1. Introduction

Semantic segmentation is a central problem in computer vision, especially in autonomous driving, robotics, and intelligent transportation systems. Unlike image classification, semantic segmentation requires assigning a semantic label to every pixel of an image. This enables scene-level understanding of road layout, vehicles, pedestrians, traffic signs, and surrounding infrastructure.

## 2. Related Work

Fully convolutional networks introduced the idea of replacing dense classification layers with convolutional operations for spatial prediction. U-Net extended encoder-decoder segmentation by introducing skip connections, which help preserve spatial detail. Cityscapes is widely used as a benchmark for urban scene understanding because it provides high-quality pixel-level annotations.

## 3. Dataset

The Cityscapes dataset contains urban street scenes with semantic labels. The official label IDs are mapped to 19 training classes. Pixels that do not belong to these semantic classes are ignored during training and evaluation.

## 4. Method

The proposed system uses a U-Net architecture composed of an encoder, bottleneck, decoder, and skip connections. The model receives RGB images and outputs a dense tensor of class logits. The final semantic prediction is obtained by applying argmax across the class dimension.

## 5. Loss Function

The optimization objective combines cross-entropy loss and Dice loss. Cross-entropy provides stable pixel-wise supervision, while Dice loss improves overlap-based optimization and helps with class imbalance.

## 6. Evaluation

The model is evaluated using pixel accuracy, mean Intersection over Union, and Dice score. mIoU is treated as the primary metric because it captures segmentation quality across all semantic classes.

## 7. Results

This section should include:

| Experiment | Image Size | Epochs | Pixel Accuracy | mIoU | Dice Score |
|---|---:|---:|---:|---:|---:|
| Baseline U-Net | 256 | 50 | TBD | TBD | TBD |
| Larger Input | 512 | 50 | TBD | TBD | TBD |
| No Dice Loss | 256 | 50 | TBD | TBD | TBD |

## 8. Discussion

Discuss which classes were segmented accurately and which classes remained difficult. Small or rare objects such as traffic signs, poles, riders, and motorcycles are expected to be more challenging than large regions such as road, building, sky, and vegetation.

## 9. Conclusion

The project demonstrates a complete semantic segmentation pipeline using U-Net and Cityscapes. It includes preprocessing, training, evaluation, quantitative metrics, qualitative visualization, and reproducibility mechanisms.

## Future Work

- Add data augmentation.
- Compare U-Net with DeepLabV3+ or SegFormer.
- Train at higher resolution.
- Use class-weighted loss.
- Add experiment tracking.
