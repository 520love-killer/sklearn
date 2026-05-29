from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-100 训练集 RGB 三通道统计值。
# 这些值用于 Z-score 标准化，让每个通道的数据分布更接近均值 0、标准差 1。
CIFAR100_MEAN = (0.5071, 0.4866, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def build_cifar100_transforms():
    """构建 CIFAR-100 的训练集和测试集图像预处理流程。"""

    train_transform = transforms.Compose(
        [
            # 1. 随机裁剪（Random Crop）
            # 先在 32x32 图片四周各补 4 个像素，再随机裁剪回 32x32。
            # 作用：让模型看到同一张图片的轻微平移版本，提升泛化能力。
            transforms.RandomCrop(size=32, padding=4),

            # 2. 随机水平翻转（Random Horizontal Flip）
            # 以 50% 概率把图片左右翻转。
            # 作用：增加训练样本的变化，减少模型只记住固定方向特征的风险。
            transforms.RandomHorizontalFlip(p=0.5),

            # 3. 归一化（Normalization）
            # ToTensor 会把 PIL 图片或 numpy 图片转换成 PyTorch Tensor。
            # 同时会把像素值从整数范围 [0, 255] 缩放到浮点范围 [0, 1]。
            # 输出形状也会从 [H, W, C] 变成 [C, H, W]，即 [3, 32, 32]。
            transforms.ToTensor(),

            # 4. 标准化（Standardization, Z-score）
            # 对每个 RGB 通道执行：(x - mean) / std。
            # 作用：让输入分布更稳定，通常可以让神经网络更容易训练。
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
        ]
    )

    test_transform = transforms.Compose(
        [
            # 测试集不做随机裁剪、随机翻转等数据增强。
            # 原因：评估阶段应该保持输入确定，保证测试指标稳定、可复现。

            # 1. 归一化（Normalization）
            # 把原始像素 [0, 255] 转换成 [0, 1] 的 Tensor。
            transforms.ToTensor(),

            # 2. 标准化（Standardization, Z-score）
            # 测试集必须使用训练集的 mean/std，而不是单独使用测试集统计值。
            # 原因：避免把测试集信息泄露到训练流程中。
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
        ]
    )

    return train_transform, test_transform


def build_cifar100_datasets(data_dir="data", download=False):
    """创建 CIFAR-100 训练集和测试集对象。"""

    train_transform, test_transform = build_cifar100_transforms()
    data_dir = Path(data_dir)

    train_dataset = datasets.CIFAR100(
        root=str(data_dir),
        train=True,
        download=download,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR100(
        root=str(data_dir),
        train=False,
        download=download,
        transform=test_transform,
    )

    # 5. 标签编码（Label Encoding）
    # torchvision.datasets.CIFAR100 已经把类别标签编码成整数：
    # - train_dataset.targets 是训练集标签列表
    # - test_dataset.targets 是测试集标签列表
    # - 标签范围是 0 到 99，一共 100 个类别
    # 因此这里不需要再手动做 LabelEncoder。
    return train_dataset, test_dataset


def build_cifar100_dataloaders(
    data_dir="data",
    batch_size=128,
    num_workers=2,
    download=False,
    pin_memory=True,
):
    """生成 ViT-Tiny 训练 CIFAR-100 所需的 train_loader 和 test_loader。"""

    train_dataset, test_dataset = build_cifar100_datasets(
        data_dir=data_dir,
        download=download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = build_cifar100_dataloaders(
        data_dir="data",
        batch_size=128,
        num_workers=2,
        download=False,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Class count: {len(train_loader.dataset.classes)}")
