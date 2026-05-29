import pickle
from pathlib import Path

import numpy as np


DATA_ROOT = Path("data/cifar-100-python")
OUTPUT_DIR = Path("outputs/preprocessed_cifar100")

# 使用训练集统计值，和 cifar100_preprocessing.py 中的 PyTorch Normalize 保持一致。
CIFAR100_MEAN = np.array([0.5071, 0.4866, 0.4409], dtype=np.float32)
CIFAR100_STD = np.array([0.2673, 0.2564, 0.2762], dtype=np.float32)


def load_pickle(path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def normalize_to_unit_range(data):
    """归一化：把原始像素从 [0, 255] 转换到 [0, 1]。"""
    images = data.reshape(-1, 3, 32, 32).astype(np.float32)
    return images / 255.0


def standardize_z_score(images):
    """标准化：对 RGB 每个通道执行 z-score，即 (x - mean) / std。"""
    mean = CIFAR100_MEAN.reshape(1, 3, 1, 1)
    std = CIFAR100_STD.reshape(1, 3, 1, 1)
    return (images - mean) / std


def preprocess_split(split_name, payload):
    """处理 train/test 划分，并保存一个可复用的 .npz 文件。"""
    normalized_images = normalize_to_unit_range(payload["data"])
    standardized_images = standardize_z_score(normalized_images)

    # CIFAR-100 的 fine_labels 已经是 0-99 的整数编码，不需要额外 LabelEncoder。
    labels = np.array(payload["fine_labels"], dtype=np.int64)
    coarse_labels = np.array(payload["coarse_labels"], dtype=np.int64)

    output_path = OUTPUT_DIR / f"{split_name}_preprocessed.npz"
    np.savez_compressed(
        output_path,
        images=standardized_images,
        labels=labels,
        coarse_labels=coarse_labels,
    )
    return output_path, standardized_images, labels


def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Missing CIFAR-100 directory: {DATA_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_pickle(DATA_ROOT / "train")
    test = load_pickle(DATA_ROOT / "test")

    train_path, train_images, train_labels = preprocess_split("train", train)
    test_path, test_images, test_labels = preprocess_split("test", test)

    print("CIFAR-100 feature preprocessing completed.")
    print(f"Train output: {train_path.resolve()}")
    print(f"Test output: {test_path.resolve()}")
    print(f"Train images shape: {train_images.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"Label range: {train_labels.min()} - {train_labels.max()}")
    print("First train batch check:")
    first_batch = train_images[:128]
    print(f"  batch shape: {first_batch.shape}")
    print(f"  channel mean: {first_batch.mean(axis=(0, 2, 3))}")
    print(f"  channel std: {first_batch.std(axis=(0, 2, 3))}")


if __name__ == "__main__":
    main()
