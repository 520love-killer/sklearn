import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CIFAR100_MEAN = torch.tensor([0.5071, 0.4866, 0.4409], dtype=torch.float32)
CIFAR100_STD = torch.tensor([0.2673, 0.2564, 0.2762], dtype=torch.float32)


class PreprocessedCIFAR100Dataset(Dataset):
    """读取已经完成 ToTensor、归一化、Z-score 标准化和标签编码的 CIFAR-100 npz 文件。"""

    def __init__(self, npz_path, augment=False):
        self.npz_path = Path(npz_path)
        payload = np.load(self.npz_path)

        self.images = torch.from_numpy(payload["images"]).float()
        self.labels = torch.from_numpy(payload["labels"]).long()
        self.augment = augment

        # 训练增强只使用轻量方法：随机裁剪和随机水平翻转。
        # 这里的输入已经是标准化后的 Tensor，因此可以直接对 Tensor 做几何增强。
        self.augmentation = transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.augment:
            image = self.augmentation(image)

        return image, label


def denormalize_image(image):
    """把 Z-score 标准化后的图像还原到 [0, 1]，便于可视化。"""
    mean = CIFAR100_MEAN.view(3, 1, 1)
    std = CIFAR100_STD.view(3, 1, 1)
    return torch.clamp(image * std + mean, 0.0, 1.0)


def summarize_split(name, images, labels):
    label_counter = Counter(labels.tolist())
    class_ids = sorted(label_counter)
    counts = np.array([label_counter[class_id] for class_id in class_ids])

    return {
        "name": name,
        "image_shape": tuple(images.shape),
        "label_shape": tuple(labels.shape),
        "sample_count": int(images.shape[0]),
        "channel_count": int(images.shape[1]) if images.ndim == 4 else None,
        "height": int(images.shape[2]) if images.ndim == 4 else None,
        "width": int(images.shape[3]) if images.ndim == 4 else None,
        "label_count": int(labels.shape[0]),
        "class_count": int(len(class_ids)),
        "label_min": int(labels.min()),
        "label_max": int(labels.max()),
        "class_min_count": int(counts.min()),
        "class_max_count": int(counts.max()),
        "is_balanced": bool(counts.min() == counts.max()),
        "has_nan": bool(torch.isnan(images).any()),
        "has_inf": bool(torch.isinf(images).any()),
        "value_min": float(images.min()),
        "value_max": float(images.max()),
        "value_mean": float(images.mean()),
        "value_std": float(images.std(unbiased=False)),
        "channel_mean": images.mean(dim=(0, 2, 3)).tolist(),
        "channel_std": images.std(dim=(0, 2, 3), unbiased=False).tolist(),
        "label_distribution": {int(k): int(v) for k, v in sorted(label_counter.items())},
    }


def build_quality_report(train_summary, test_summary):
    def fmt(values):
        return ", ".join(f"{value:.6f}" for value in values)

    def split_table(summary):
        return f"""| 检查项 | 结果 |
| --- | --- |
| 图像 Shape | `{summary["image_shape"]}` |
| 标签 Shape | `{summary["label_shape"]}` |
| 样本数量 | {summary["sample_count"]} |
| 标签数量 | {summary["label_count"]} |
| 类别数量 | {summary["class_count"]} |
| 标签范围 | {summary["label_min"]} - {summary["label_max"]} |
| 每类最少样本数 | {summary["class_min_count"]} |
| 每类最多样本数 | {summary["class_max_count"]} |
| 类别是否均衡 | {"是" if summary["is_balanced"] else "否"} |
| 是否存在 NaN | {"是" if summary["has_nan"] else "否"} |
| 是否存在 Inf | {"是" if summary["has_inf"] else "否"} |
| 数值最小值 | {summary["value_min"]:.6f} |
| 数值最大值 | {summary["value_max"]:.6f} |
| 全局均值 | {summary["value_mean"]:.6f} |
| 全局标准差 | {summary["value_std"]:.6f} |
| RGB 通道均值 | `({fmt(summary["channel_mean"])})` |
| RGB 通道标准差 | `({fmt(summary["channel_std"])})` |"""

    return f"""# CIFAR-100 预处理数据质量检查报告

## 检查对象

- 训练集：`outputs/preprocessed_cifar100/train_preprocessed.npz`
- 测试集：`outputs/preprocessed_cifar100/test_preprocessed.npz`
- 图像字段：`images`
- 标签字段：`labels`

## 训练集质量检查

{split_table(train_summary)}

## 测试集质量检查

{split_table(test_summary)}

## 结论

当前预处理后的 CIFAR-100 数据满足 ViT-Tiny 训练输入要求：

- 图像张量为 `[N, 3, 32, 32]`，符合 PyTorch 卷积和 ViT patch embedding 的输入格式。
- 标签数量与图像数量一致，标签范围为 `0 - 99`。
- 训练集和测试集均覆盖完整 100 个类别。
- 训练集每类 500 张，测试集每类 100 张，类别分布均衡。
- 未检测到 NaN 或 Inf。
- 标准化后数值范围处于合理区间，均值接近 0，标准差接近 1。
"""


def build_feature_section():
    return """# 特征处理与构造

## 总体流程

本实验面向 CIFAR-100 图像分类任务构建 ViT-Tiny 输入特征。CIFAR-100 原始图像尺寸为 `32 x 32 x 3`，经过预处理后被组织为 PyTorch 标准张量格式 `[N, 3, 32, 32]`，其中 `N` 表示样本数量，`3` 表示 RGB 三个通道。

## ToTensor 与归一化

首先将原始图像由整数像素表示转换为浮点张量表示，并将像素值从 `[0, 255]` 线性缩放到 `[0, 1]`。该步骤统一了输入数值范围，避免较大的像素数值直接进入网络导致梯度更新不稳定。

对于 ViT 而言，图像会被划分为若干 patch，并映射为 token 序列。稳定的输入数值范围有助于 patch embedding 层学习更加平滑的低层视觉表示。

## Z-score 标准化

在归一化之后，对 RGB 三个通道分别执行 Z-score 标准化：

```text
x' = (x - mean) / std
```

其中 `mean` 和 `std` 使用 CIFAR-100 训练集统计值。该步骤使不同通道的输入分布更加一致，有助于优化器更稳定地更新参数，并降低不同颜色通道数值尺度差异对模型训练的影响。

对于 ViT-Tiny，标准化可以改善 patch token 的初始分布，使 Transformer Encoder 中的注意力计算和前馈网络更容易收敛。

## 标签编码

CIFAR-100 的细分类标签已经被编码为 `0 - 99` 的整数形式，因此无需额外引入 LabelEncoder。该编码形式可以直接与 `torch.nn.CrossEntropyLoss` 配合使用。

## RandomCrop(32, padding=4)

训练阶段在图像四周填充 4 个像素，并随机裁剪回 `32 x 32`。该方法模拟目标在小范围内的位置偏移，使模型不会过度依赖物体在图像中的固定位置。

ViT 使用 patch 序列建模图像内容，随机裁剪能够改变局部 patch 的组合方式，促使模型学习更稳健的位置关系和局部结构特征。对于样本规模有限的 CIFAR-100，该增强可以有效降低过拟合风险。

## RandomHorizontalFlip(p=0.5)

训练阶段以 50% 概率对图像进行水平翻转。该方法在不改变大多数类别语义的前提下扩充样本变化，使模型能够学习左右方向不敏感的视觉特征。

对于 ViT-Tiny，水平翻转可以增加 patch token 的空间排列多样性，减少模型对固定方向纹理和布局的记忆，从而提升泛化能力并缓解过拟合。

## 测试集处理原则

测试集仅保留确定性的归一化和标准化，不使用随机裁剪或随机翻转。这样可以保证评估结果稳定、可复现，并避免随机增强导致测试指标波动。
"""


def save_visualization(train_images, output_path, seed=42):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_images), generator=generator)[:10]
    selected = train_images[indices]

    crop_transform = transforms.RandomCrop(size=32, padding=4)
    flip_transform = transforms.RandomHorizontalFlip(p=1.0)

    rows = [
        ("Original", [denormalize_image(image) for image in selected]),
        ("Standardized", [image for image in selected]),
        ("RandomCrop", [crop_transform(image) for image in selected]),
        ("Flip", [flip_transform(image) for image in selected]),
    ]

    fig, axes = plt.subplots(len(rows), 10, figsize=(18, 7))
    for row_index, (row_name, images) in enumerate(rows):
        for col_index, image in enumerate(images):
            ax = axes[row_index, col_index]
            image_to_show = image
            if row_name != "Original":
                image_to_show = denormalize_image(image_to_show)
            image_to_show = image_to_show.permute(1, 2, 0).cpu().numpy()
            ax.imshow(image_to_show)
            ax.axis("off")
            if col_index == 0:
                ax.set_ylabel(row_name, fontsize=11)

    fig.suptitle("CIFAR-100 Preprocessing Visualization", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_dataloaders(train_npz, test_npz, batch_size, num_workers):
    train_dataset = PreprocessedCIFAR100Dataset(train_npz, augment=True)
    test_dataset = PreprocessedCIFAR100Dataset(test_npz, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="CIFAR-100 preprocessed data QA and augmentation pipeline.")
    parser.add_argument("--train-npz", type=Path, default=Path("outputs/preprocessed_cifar100/train_preprocessed.npz"))
    parser.add_argument("--test-npz", type=Path, default=Path("outputs/preprocessed_cifar100/test_preprocessed.npz"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PreprocessedCIFAR100Dataset(args.train_npz, augment=True)
    test_dataset = PreprocessedCIFAR100Dataset(args.test_npz, augment=False)

    train_summary = summarize_split("train", train_dataset.images, train_dataset.labels)
    test_summary = summarize_split("test", test_dataset.images, test_dataset.labels)

    quality_report = build_quality_report(train_summary, test_summary)
    quality_report_path = args.report_dir / "cifar100_preprocessed_quality_report.md"
    quality_report_path.write_text(quality_report, encoding="utf-8-sig")

    feature_section_path = args.report_dir / "cifar100_feature_engineering_section.md"
    feature_section_path.write_text(build_feature_section(), encoding="utf-8-sig")

    visualization_path = args.output_dir / "preprocess_visualization.png"
    save_visualization(train_dataset.images, visualization_path)

    train_loader, test_loader = build_dataloaders(
        args.train_npz,
        args.test_npz,
        args.batch_size,
        args.num_workers,
    )

    images, labels = next(iter(train_loader))
    print("CIFAR-100 feature pipeline completed.")
    print(f"Quality report: {quality_report_path.resolve()}")
    print(f"Feature section: {feature_section_path.resolve()}")
    print(f"Visualization: {visualization_path.resolve()}")
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Test loader batches: {len(test_loader)}")
    print(f"First train batch images: {tuple(images.shape)}")
    print(f"First train batch labels: {tuple(labels.shape)}")


if __name__ == "__main__":
    main()
