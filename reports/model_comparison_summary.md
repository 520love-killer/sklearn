# CIFAR-100 Model Comparison Summary

This report records the local 50-epoch CIFAR-100 experiment so the important result can be committed without uploading large `runs/` artifacts or model checkpoints.

## Setup

- Dataset: CIFAR-100
- Image size: `32x32`
- Classes: `100`
- Epochs: `50`
- Batch size: `128`
- Device used locally: NVIDIA GeForce RTX 4060 Laptop GPU
- Script: `train_cifar100_multi_models.py`

## Results

| Model | Best Epoch | Best Test Accuracy | Best Test Loss | Final Train Accuracy | Final Test Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-Tony | 45 | 58.18% | 1.6412 | 77.58% | 58.05% |
| ResNet-18 | 47 | 76.33% | 0.9570 | 99.86% | 76.23% |
| EfficientNetV2-S | 49 | 55.57% | 1.5905 | 60.27% | 55.48% |

## Interpretation

ResNet-18 performed best in this run. ViT-Tony reached about 58% test accuracy, which is a reasonable baseline for a small ViT trained from scratch on CIFAR-100, but it is weaker than ResNet-18 under the current augmentation and training setup.

EfficientNetV2-S did not outperform ViT-Tony in this run. Because it is a larger architecture, it may need different hyperparameters, longer training, or pretrained weights to show its usual strength.

## Generated Local Artifacts

The following files were generated locally under `runs/` and are intentionally ignored by Git:

- `runs/vit_tony/best_model.pth`
- `runs/resnet18/best_model.pth`
- `runs/efficientnetv2_s/best_model.pth`
- `runs/*/training_log.csv`
- `runs/*/loss_curve.png`
- `runs/*/accuracy_curve.png`
- `runs/model_comparison/*.png`
- `runs/model_comparison/summary.csv`
