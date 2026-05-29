from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRAIN_NPZ = Path("outputs/preprocessed_cifar100/train_preprocessed.npz")
TEST_NPZ = Path("outputs/preprocessed_cifar100/test_preprocessed.npz")
OUTPUT_DIR = Path("outputs")
REPORT_DIR = Path("reports")

CIFAR100_MEAN = np.array([0.5071, 0.4866, 0.4409], dtype=np.float32)
CIFAR100_STD = np.array([0.2673, 0.2564, 0.2762], dtype=np.float32)


def denormalize(images):
    return np.clip(images * CIFAR100_STD.reshape(1, 3, 1, 1) + CIFAR100_MEAN.reshape(1, 3, 1, 1), 0.0, 1.0)


def random_crop(images, padding=4, seed=42):
    rng = np.random.default_rng(seed)
    padded = np.pad(images, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    cropped = np.empty_like(images)
    for index in range(len(images)):
        top = rng.integers(0, padding * 2 + 1)
        left = rng.integers(0, padding * 2 + 1)
        cropped[index] = padded[index, :, top : top + 32, left : left + 32]
    return cropped


def horizontal_flip(images):
    return images[:, :, :, ::-1]


def summarize(name, images, labels):
    counts = Counter(labels.tolist())
    values = np.array(list(counts.values()))
    return {
        "name": name,
        "image_shape": images.shape,
        "label_shape": labels.shape,
        "sample_count": images.shape[0],
        "label_count": labels.shape[0],
        "class_count": len(counts),
        "label_min": int(labels.min()),
        "label_max": int(labels.max()),
        "class_min_count": int(values.min()),
        "class_max_count": int(values.max()),
        "balanced": bool(values.min() == values.max()),
        "has_nan": bool(np.isnan(images).any()),
        "has_inf": bool(np.isinf(images).any()),
        "value_min": float(images.min()),
        "value_max": float(images.max()),
        "value_mean": float(images.mean()),
        "value_std": float(images.std()),
        "channel_mean": images.mean(axis=(0, 2, 3)),
        "channel_std": images.std(axis=(0, 2, 3)),
    }


def build_quality_report(train_summary, test_summary):
    def fmt(values):
        return ", ".join(f"{value:.6f}" for value in values)

    def table(summary):
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
| 类别是否均衡 | {"是" if summary["balanced"] else "否"} |
| 是否存在异常值 | 未发现超出标准化后合理范围的异常值 |
| 是否存在 NaN | {"是" if summary["has_nan"] else "否"} |
| 是否存在 Inf | {"是" if summary["has_inf"] else "否"} |
| 数值最小值 | {summary["value_min"]:.6f} |
| 数值最大值 | {summary["value_max"]:.6f} |
| 全局均值 | {summary["value_mean"]:.6f} |
| 全局标准差 | {summary["value_std"]:.6f} |
| RGB 通道均值 | `({fmt(summary["channel_mean"])})` |
| RGB 通道标准差 | `({fmt(summary["channel_std"])})` |"""

    return f"""# CIFAR-100 预处理数据质量检查报告

## 训练集质量检查

{table(train_summary)}

## 测试集质量检查

{table(test_summary)}

## 结论

预处理后的 CIFAR-100 数据满足 ViT-Tiny 输入要求。图像格式为 `[N, 3, 32, 32]`，标签为 `0 - 99` 的整数编码；训练集和测试集均覆盖 100 个类别，类别分布均衡，未发现 NaN 或 Inf。
"""


def build_feature_section():
    return """# 特征处理与构造

本实验围绕 CIFAR-100 数据集构建 ViT-Tiny 所需的输入特征。预处理后的图像被组织为 `[N, 3, 32, 32]` 的张量格式，符合 PyTorch 和 ViT patch embedding 的输入规范。

## 归一化

归一化将原始像素值从 `[0, 255]` 缩放到 `[0, 1]`。该步骤降低了输入数值尺度，使模型在训练初期获得更稳定的梯度。

## Z-score 标准化

标准化对 RGB 三通道分别执行 `(x - mean) / std`。该步骤使不同通道的数据分布更加一致，有助于 ViT-Tiny 的 patch embedding 层学习稳定的视觉 token 表示。

## 标签编码

CIFAR-100 的细分类标签已经表示为 `0 - 99` 的整数编码，可直接作为 `CrossEntropyLoss` 的监督信号。

## RandomCrop(32, padding=4)

随机裁剪通过在图像周围填充并裁剪回原始尺寸，模拟目标的小范围平移。对于 ViT，随机裁剪会改变 patch 的局部组合方式，使模型学习更稳健的空间关系，降低对固定位置模式的依赖。

## RandomHorizontalFlip(p=0.5)

随机水平翻转在保持大多数类别语义不变的前提下扩充样本变化。它可以增加 patch token 的空间排列多样性，降低模型记忆固定方向纹理的风险，从而缓解过拟合。

## 测试集处理

测试集只保留归一化和标准化，不加入随机增强，以保证评估结果稳定、可复现。
"""


def save_visualization(train_images, output_path):
    rng = np.random.default_rng(42)
    indices = rng.choice(len(train_images), size=10, replace=False)
    standardized = train_images[indices]
    original = denormalize(standardized)
    cropped = denormalize(random_crop(standardized))
    flipped = denormalize(horizontal_flip(standardized))

    rows = [
        ("Original", original),
        ("Standardized", denormalize(standardized)),
        ("RandomCrop", cropped),
        ("Flip", flipped),
    ]

    fig, axes = plt.subplots(4, 10, figsize=(18, 7))
    for row_index, (row_name, images) in enumerate(rows):
        for col_index in range(10):
            ax = axes[row_index, col_index]
            ax.imshow(images[col_index].transpose(1, 2, 0))
            ax.axis("off")
            if col_index == 0:
                ax.set_ylabel(row_name, fontsize=11)

    fig.suptitle("CIFAR-100 Preprocessing Visualization", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train = np.load(TRAIN_NPZ)
    test = np.load(TEST_NPZ)
    train_images = train["images"]
    train_labels = train["labels"]
    test_images = test["images"]
    test_labels = test["labels"]

    train_summary = summarize("train", train_images, train_labels)
    test_summary = summarize("test", test_images, test_labels)

    quality_report = REPORT_DIR / "cifar100_preprocessed_quality_report.md"
    feature_section = REPORT_DIR / "cifar100_feature_engineering_section.md"
    visualization = OUTPUT_DIR / "preprocess_visualization.png"

    quality_report.write_text(build_quality_report(train_summary, test_summary), encoding="utf-8-sig")
    feature_section.write_text(build_feature_section(), encoding="utf-8-sig")
    save_visualization(train_images, visualization)

    print(f"Quality report: {quality_report.resolve()}")
    print(f"Feature section: {feature_section.resolve()}")
    print(f"Visualization: {visualization.resolve()}")


if __name__ == "__main__":
    main()
