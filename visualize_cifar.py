import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def unpickle(path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def load_cifar10(root, count):
    data_dir = root / "cifar-10-batches-py"
    meta = unpickle(data_dir / "batches.meta")
    names = meta["label_names"]
    batch = unpickle(data_dir / "data_batch_1")
    images = batch["data"][:count].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = [names[label] for label in batch["labels"][:count]]
    return images, labels


def load_cifar100(root, count):
    data_dir = root / "cifar-100-python"
    meta = unpickle(data_dir / "meta")
    names = meta["fine_label_names"]
    batch = unpickle(data_dir / "train")
    images = batch["data"][:count].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = [names[label] for label in batch["fine_labels"][:count]]
    return images, labels


def save_grid(images, labels, output, cols=8):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9))
    axes = np.atleast_1d(axes).ravel()

    for ax, image, label in zip(axes, images, labels):
        ax.imshow(image)
        ax.set_title(label, fontsize=8)
        ax.axis("off")

    for ax in axes[len(images) :]:
        ax.axis("off")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Save sample image grids for CIFAR-10 or CIFAR-100.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.dataset == "cifar10":
        images, labels = load_cifar10(args.data_dir, args.count)
        output = args.output or Path("outputs/cifar10_samples.png")
    else:
        images, labels = load_cifar100(args.data_dir, args.count)
        output = args.output or Path("outputs/cifar100_samples.png")

    save_grid(images, labels, output)
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
