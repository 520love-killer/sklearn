import argparse
import csv
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class CIFAR100NPZDataset(Dataset):
    """读取已经预处理完成的 CIFAR-100 npz 文件。"""

    def __init__(self, npz_path, train=False):
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Cannot find dataset file: {self.npz_path}")

        payload = np.load(self.npz_path)
        self.images = torch.from_numpy(payload["images"]).float()
        self.labels = torch.from_numpy(payload["labels"]).long()
        self.train = train

        if self.images.ndim != 4 or self.images.shape[1:] != (3, 32, 32):
            raise ValueError(f"Expected images shape [N, 3, 32, 32], got {tuple(self.images.shape)}")
        if self.labels.ndim != 1 or len(self.labels) != len(self.images):
            raise ValueError("Labels must be one-dimensional and match image count.")

        # 训练集只补充轻量几何增强。
        # 注意：npz 中 images 已经完成 ToTensor、归一化和 Z-score 标准化，
        # 因此这里不能再次 Normalize，否则会造成二次标准化。
        self.train_transform = transforms.Compose(
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

        # RandomCrop 和 RandomHorizontalFlip 是训练期增强。
        # 它们能增加 patch 排列变化，降低 ViT-Tony 对固定空间位置的记忆。
        if self.train:
            image = self.train_transform(image)

        return image, label


class PatchEmbedding(nn.Module):
    """把 32x32 图像切成 patch，并映射成 Transformer token。"""

    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.num_patches = (image_size // patch_size) ** 2

        # 用卷积实现 patch embedding：
        # 输入 [B, 3, 32, 32] -> 输出 [B, embed_dim, 8, 8]。
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        # [B, embed_dim, 8, 8] -> [B, 64, embed_dim]
        return x.flatten(2).transpose(1, 2)


class ViTTony(nn.Module):
    """轻量化 ViT-Tony，适合 CIFAR-100 课程实验。"""

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=100,
        embed_dim=192,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        # CLS Token 用来聚合整张图片的信息，最后送入分类头。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position Embedding 为每个 patch token 提供位置信息。
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def set_seed(seed):
    """固定随机种子，提升实验可复现性。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_dataloaders(args):
    train_dataset = CIFAR100NPZDataset(args.train_npz, train=True)
    test_dataset = CIFAR100NPZDataset(args.test_npz, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    return train_loader, test_loader


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).sum().item()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=f"Epoch {epoch} Train", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy_from_logits(logits, labels)
        total_samples += batch_size

        progress.set_postfix(
            loss=total_loss / total_samples,
            acc=total_correct / total_samples,
        )

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=f"Epoch {epoch} Test", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy_from_logits(logits, labels)
        total_samples += batch_size

        progress.set_postfix(
            loss=total_loss / total_samples,
            acc=total_correct / total_samples,
        )

    return total_loss / total_samples, total_correct / total_samples


def save_training_log(log_path, history):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc"],
        )
        writer.writeheader()
        writer.writerows(history)


def annotate_best_point(ax, epochs, values, mode):
    if mode == "min":
        best_index = int(np.argmin(values))
        label = f"best {values[best_index]:.4f}"
    else:
        best_index = int(np.argmax(values))
        label = f"best {values[best_index]:.4f}"

    ax.scatter(epochs[best_index], values[best_index], s=55, zorder=5)
    ax.annotate(
        label,
        xy=(epochs[best_index], values[best_index]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )


def plot_curves(output_dir, history):
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    test_loss = [item["test_loss"] for item in history]
    train_acc = [item["train_acc"] for item in history]
    test_acc = [item["test_acc"] for item in history]

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(epochs, train_loss, color="#1f77b4", linewidth=2, label="Train Loss")
    ax.plot(epochs, test_loss, color="#d62728", linewidth=2, label="Test Loss")
    annotate_best_point(ax, epochs, test_loss, mode="min")
    ax.set_title("ViT-Tony CIFAR-100 Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(epochs, train_acc, color="#2ca02c", linewidth=2, label="Train Accuracy")
    ax.plot(epochs, test_acc, color="#9467bd", linewidth=2, label="Test Accuracy")
    annotate_best_point(ax, epochs, test_acc, mode="max")
    ax.set_title("ViT-Tony CIFAR-100 Accuracy Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT-Tony on preprocessed CIFAR-100 npz files.")
    parser.add_argument("--train-npz", type=Path, default=Path("outputs/preprocessed_cifar100/train_preprocessed.npz"))
    parser.add_argument("--test-npz", type=Path, default=Path("outputs/preprocessed_cifar100/test_preprocessed.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vit_tony_cifar100"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pin-memory", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.pin_memory = not args.no_pin_memory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Train npz: {args.train_npz}")
    print(f"Test npz: {args.test_npz}")

    train_loader, test_loader = build_dataloaders(args)

    model = ViTTony(
        image_size=32,
        patch_size=4,
        num_classes=100,
        embed_dim=192,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_test_acc = -math.inf
    best_model_path = args.output_dir / "best_vit_tony_cifar100.pth"
    log_path = args.output_dir / "training_log.csv"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
        )
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device,
            epoch,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(row)
        save_training_log(log_path, history)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "test_acc": test_acc,
                    "args": vars(args),
                },
                best_model_path,
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"best_test_acc={best_test_acc:.4f}"
        )

    plot_curves(args.output_dir, history)
    print(f"Training finished.")
    print(f"Best model: {best_model_path}")
    print(f"Training log: {log_path}")
    print(f"Loss curve: {args.output_dir / 'loss_curve.png'}")
    print(f"Accuracy curve: {args.output_dir / 'accuracy_curve.png'}")


if __name__ == "__main__":
    main()
