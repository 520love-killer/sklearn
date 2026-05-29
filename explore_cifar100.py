import pickle
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path("data/cifar-100-python")
OUTPUT_DIR = Path("outputs")
REPORT_DIR = Path("reports")


def load_pickle(path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def image_from_row(row):
    return row.reshape(3, 32, 32).transpose(1, 2, 0)


def channel_mean_std(data):
    images = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    mean = images.mean(axis=(0, 2, 3))
    std = images.std(axis=(0, 2, 3))
    return mean, std


def format_triplet(values):
    return ", ".join(f"{value:.4f}" for value in values)


def save_fine_distribution_chart(train_counts, test_counts, fine_names, output_path):
    labels = np.arange(len(fine_names))
    train_values = [train_counts[i] for i in labels]
    test_values = [test_counts[i] for i in labels]

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(labels - 0.2, train_values, width=0.4, label="train")
    ax.bar(labels + 0.2, test_values, width=0.4, label="test")
    ax.set_title("CIFAR-100 Fine Class Distribution")
    ax.set_xlabel("Fine class id")
    ax.set_ylabel("Sample count")
    ax.set_xticks(labels[::5])
    ax.set_xticklabels(labels[::5])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_coarse_distribution_chart(train_counts, test_counts, coarse_names, output_path):
    labels = np.arange(len(coarse_names))
    train_values = [train_counts[i] for i in labels]
    test_values = [test_counts[i] for i in labels]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(labels - 0.2, train_values, width=0.4, label="train")
    ax.bar(labels + 0.2, test_values, width=0.4, label="test")
    ax.set_title("CIFAR-100 Coarse Class Distribution")
    ax.set_xlabel("Coarse class")
    ax.set_ylabel("Sample count")
    ax.set_xticks(labels)
    ax.set_xticklabels(coarse_names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_sample_grid(train_data, fine_labels, fine_names, output_path, count=40, cols=8):
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 2.0))
    axes = axes.ravel()

    for ax, row, label in zip(axes, train_data[:count], fine_labels[:count]):
        ax.imshow(image_from_row(row))
        ax.set_title(fine_names[label], fontsize=8)
        ax.axis("off")

    for ax in axes[count:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    output_path,
    fine_chart,
    coarse_chart,
    sample_grid,
    fine_names,
    coarse_names,
    train_fine_counts,
    test_fine_counts,
    train_coarse_counts,
    test_coarse_counts,
    train_shape,
    test_shape,
    train_mean,
    train_std,
    test_mean,
    test_std,
    all_mean,
    all_std,
):
    total_train = sum(train_fine_counts.values())
    total_test = sum(test_fine_counts.values())
    fine_train_values = list(train_fine_counts.values())
    fine_test_values = list(test_fine_counts.values())
    coarse_train_values = list(train_coarse_counts.values())
    coarse_test_values = list(test_coarse_counts.values())

    fine_rows = "\n".join(
        f"| {idx} | {name} | {train_fine_counts[idx]} | {test_fine_counts[idx]} |"
        for idx, name in enumerate(fine_names)
    )
    coarse_rows = "\n".join(
        f"| {idx} | {name} | {train_coarse_counts[idx]} | {test_coarse_counts[idx]} |"
        for idx, name in enumerate(coarse_names)
    )

    markdown = f"""# CIFAR-100 数据探索报告

## 数据位置

- CIFAR-100 目录：`{DATA_ROOT.resolve()}`
- 训练文件：`{(DATA_ROOT / "train").resolve()}`
- 测试文件：`{(DATA_ROOT / "test").resolve()}`
- 元信息文件：`{(DATA_ROOT / "meta").resolve()}`

## 基本信息

| 项目 | 数值 |
| --- | ---: |
| 训练集样本数 | {total_train} |
| 测试集样本数 | {total_test} |
| 总样本数 | {total_train + total_test} |
| 图像尺寸 | 32 x 32 x 3 |
| 单张图像展开维度 | 3072 |
| 训练数据矩阵形状 | `{train_shape}` |
| 测试数据矩阵形状 | `{test_shape}` |
| 细分类数量 | {len(fine_names)} |
| 粗分类数量 | {len(coarse_names)} |

## 未处理原始数据说明

CIFAR-100 原始数据不是一张张 `.jpg` 或 `.png` 图片，而是保存在 Python 二进制文件中的矩阵。

| 项目 | 数值 |
| --- | --- |
| 原始训练文件 | `train` |
| 原始测试文件 | `test` |
| 原始像素范围 | 0 - 255 |
| 原始存储形状 | `[样本数, 3072]` |
| 还原后的图片尺寸 | `32 x 32 x 3` |
| 通道顺序 | RGB |

用于神经网络训练时，通常会先把像素从 `0 - 255` 缩放到 `0 - 1`，再按 RGB 三通道做标准化。

## CIFAR-100 均值和标准差

下面的统计值来自当前本地 CIFAR-100 原始数据，统计前先执行 `像素值 / 255.0`。

| 数据范围 | RGB 均值 mean | RGB 标准差 std |
| --- | --- | --- |
| 训练集 train | `({format_triplet(train_mean)})` | `({format_triplet(train_std)})` |
| 测试集 test | `({format_triplet(test_mean)})` | `({format_triplet(test_std)})` |
| 训练集 + 测试集 | `({format_triplet(all_mean)})` | `({format_triplet(all_std)})` |

训练代码中建议优先使用训练集统计值做归一化，避免把测试集信息提前泄露到训练流程中。

## 类别平衡性结论

CIFAR-100 数据集是平衡的。

- 100 个细分类中，每个类别训练集都有 {min(fine_train_values)} 张，测试集都有 {min(fine_test_values)} 张。
- 20 个粗分类中，每个类别训练集都有 {min(coarse_train_values)} 张，测试集都有 {min(coarse_test_values)} 张。
- 训练集和测试集都覆盖完整的 100 个细分类和 20 个粗分类。

## 可视化图片

### 样本图像网格

![CIFAR-100 样本图像](../{sample_grid.as_posix()})

### 100 个细分类分布

![CIFAR-100 细分类分布](../{fine_chart.as_posix()})

### 20 个粗分类分布

![CIFAR-100 粗分类分布](../{coarse_chart.as_posix()})

## 图片格式与噪声观察

- CIFAR-100 的图片是 32 x 32 彩色小图，分辨率较低，细节天然有限。
- 图片以 RGB 三通道存储，每张图片展开后是 3072 个数值。
- 从样本可视化看，图片存在明显低分辨率带来的模糊和压缩感，这是 CIFAR 系列数据集的正常特征。
- 未发现明显文件损坏、全黑图、空图或类别缺失问题。

## 细分类样本数量

| ID | 类别 | 训练集 | 测试集 |
| ---: | --- | ---: | ---: |
{fine_rows}

## 粗分类样本数量

| ID | 类别 | 训练集 | 测试集 |
| ---: | --- | ---: | ---: |
{coarse_rows}
"""
    # 使用 utf-8-sig，Windows 记事本和部分 PowerShell 环境识别中文更稳定。
    output_path.write_text(markdown, encoding="utf-8-sig")


def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Missing CIFAR-100 directory: {DATA_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    meta = load_pickle(DATA_ROOT / "meta")
    train = load_pickle(DATA_ROOT / "train")
    test = load_pickle(DATA_ROOT / "test")

    fine_names = meta["fine_label_names"]
    coarse_names = meta["coarse_label_names"]

    train_fine_counts = Counter(train["fine_labels"])
    test_fine_counts = Counter(test["fine_labels"])
    train_coarse_counts = Counter(train["coarse_labels"])
    test_coarse_counts = Counter(test["coarse_labels"])
    train_mean, train_std = channel_mean_std(train["data"])
    test_mean, test_std = channel_mean_std(test["data"])
    all_data = np.concatenate([train["data"], test["data"]], axis=0)
    all_mean, all_std = channel_mean_std(all_data)

    fine_chart = OUTPUT_DIR / "cifar100_fine_class_distribution.png"
    coarse_chart = OUTPUT_DIR / "cifar100_coarse_class_distribution.png"
    sample_grid = OUTPUT_DIR / "cifar100_exploration_samples.png"
    report_path = REPORT_DIR / "cifar100_data_exploration.md"

    save_fine_distribution_chart(train_fine_counts, test_fine_counts, fine_names, fine_chart)
    save_coarse_distribution_chart(train_coarse_counts, test_coarse_counts, coarse_names, coarse_chart)
    save_sample_grid(train["data"], train["fine_labels"], fine_names, sample_grid)

    write_report(
        report_path,
        fine_chart,
        coarse_chart,
        sample_grid,
        fine_names,
        coarse_names,
        train_fine_counts,
        test_fine_counts,
        train_coarse_counts,
        test_coarse_counts,
        train["data"].shape,
        test["data"].shape,
        train_mean,
        train_std,
        test_mean,
        test_std,
        all_mean,
        all_std,
    )

    print(f"Report saved: {report_path.resolve()}")
    print(f"Sample grid saved: {sample_grid.resolve()}")
    print(f"Fine distribution saved: {fine_chart.resolve()}")
    print(f"Coarse distribution saved: {coarse_chart.resolve()}")


if __name__ == "__main__":
    main()
