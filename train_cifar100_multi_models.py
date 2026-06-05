import argparse
import csv
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]
MODEL_NAMES = ["vit_tony", "resnet18", "efficientnetv2_s"]


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class ViTTony(nn.Module):
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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
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
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_transforms(use_randaugment):
    train_steps = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if use_randaugment:
        train_steps.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    test_steps = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ]
    return transforms.Compose(train_steps), transforms.Compose(test_steps)


def build_dataloaders(args):
    train_transform, test_transform = build_transforms(args.randaugment)
    train_set = datasets.CIFAR100(
        root=str(args.data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = datasets.CIFAR100(
        root=str(args.data_dir),
        train=False,
        download=True,
        transform=test_transform,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, test_loader


def build_model(model_name, num_classes=100):
    if model_name == "vit_tony":
        return ViTTony(num_classes=num_classes)
    if model_name == "resnet18":
        model = models.resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    if model_name == "efficientnetv2_s":
        model = models.efficientnet_v2_s(weights=None, num_classes=num_classes)
        first_conv = model.features[0][0]
        if isinstance(first_conv, nn.Conv2d):
            model.features[0][0] = nn.Conv2d(
                first_conv.in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=1,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )
        return model
    raise ValueError(f"Unsupported model_name: {model_name}")


def one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).float()


def rand_bbox(width, height, lam, device):
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = torch.randint(width, (1,), device=device).item()
    cy = torch.randint(height, (1,), device=device).item()
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    return x1, y1, x2, y2


def apply_mixup_cutmix(images, labels, args, num_classes, device):
    use_mixup = args.mixup_alpha > 0
    use_cutmix = args.cutmix_alpha > 0
    if not use_mixup and not use_cutmix:
        return images, labels

    batch_size = images.size(0)
    indices = torch.randperm(batch_size, device=device)
    labels_a = one_hot(labels, num_classes)
    labels_b = labels_a[indices]

    if use_mixup and use_cutmix:
        method = "cutmix" if torch.rand(1, device=device).item() < 0.5 else "mixup"
    else:
        method = "cutmix" if use_cutmix else "mixup"

    alpha = args.cutmix_alpha if method == "cutmix" else args.mixup_alpha
    lam = np.random.beta(alpha, alpha)

    if method == "mixup":
        mixed_images = images * lam + images[indices] * (1.0 - lam)
        mixed_labels = labels_a * lam + labels_b * (1.0 - lam)
        return mixed_images, mixed_labels

    _, _, height, width = images.shape
    x1, y1, x2, y2 = rand_bbox(width, height, lam, device)
    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
    mixed_labels = labels_a * lam + labels_b * (1.0 - lam)
    return images, mixed_labels


def loss_fn(logits, targets, label_smoothing):
    if targets.ndim == 2:
        log_probs = F.log_softmax(logits, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)


def correct_count(logits, labels):
    return (logits.argmax(dim=1) == labels).sum().item()


def build_optimizer(model, model_name, args):
    optimizer_name = args.optimizer
    if optimizer_name == "auto":
        optimizer_name = "adamw" if model_name == "vit_tony" else "sgd"
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(optimizer, args, steps_per_epoch):
    if args.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
        ), "step"
    if args.scheduler == "cosine":
        warmup_steps = max(0, args.warmup_epochs * steps_per_epoch)
        total_steps = max(1, args.epochs * steps_per_epoch)

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "step"
    return None, "none"


def train_one_epoch(model, loader, optimizer, scheduler, scheduler_mode, scaler, device, args, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    progress = tqdm(loader, desc=f"{args.model_name} Epoch {epoch} Train", leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        hard_labels = labels
        images, targets = apply_mixup_cutmix(images, labels, args, 100, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, targets, args.label_smoothing)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None and scheduler_mode == "step":
            scheduler.step()

        batch_size = hard_labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += correct_count(logits.detach(), hard_labels)
        total_samples += batch_size
        progress.set_postfix(loss=total_loss / total_samples, acc=total_correct / total_samples)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device, args, epoch):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    progress = tqdm(loader, desc=f"{args.model_name} Epoch {epoch} Test", leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += correct_count(logits, labels)
        total_samples += batch_size
        progress.set_postfix(loss=total_loss / total_samples, acc=total_correct / total_samples)

    return total_loss / total_samples, total_correct / total_samples


def save_training_log(log_path, history):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr"],
        )
        writer.writeheader()
        writer.writerows(history)


def annotate_best(ax, epochs, values, mode):
    best_idx = int(np.argmin(values) if mode == "min" else np.argmax(values))
    ax.scatter(epochs[best_idx], values[best_idx], s=55, zorder=5)
    ax.annotate(
        f"best {values[best_idx]:.4f}",
        xy=(epochs[best_idx], values[best_idx]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )


def plot_model_curves(output_dir, model_name, history):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    test_loss = [row["test_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    test_acc = [row["test_acc"] for row in history]

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax.plot(epochs, test_loss, label="Test Loss", linewidth=2)
    annotate_best(ax, epochs, test_loss, "min")
    ax.set_title(f"{model_name} CIFAR-100 Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(epochs, train_acc, label="Train Accuracy", linewidth=2)
    ax.plot(epochs, test_acc, label="Test Accuracy", linewidth=2)
    annotate_best(ax, epochs, test_acc, "max")
    ax.set_title(f"{model_name} CIFAR-100 Accuracy Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=300)
    plt.close()


def read_history(log_path):
    rows = []
    if not log_path.exists():
        return rows
    with log_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_acc": float(row["train_acc"]),
                    "test_loss": float(row["test_loss"]),
                    "test_acc": float(row["test_acc"]),
                    "lr": float(row.get("lr", 0.0)),
                }
            )
    return rows


def plot_comparison(runs_dir, model_names):
    comparison_dir = runs_dir / "model_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    histories = {
        name: read_history(runs_dir / name / "training_log.csv")
        for name in model_names
        if (runs_dir / name / "training_log.csv").exists()
    }
    if not histories:
        print("No model histories found for comparison.")
        return

    plt.figure(figsize=(9, 5.5))
    ax = plt.gca()
    for name, history in histories.items():
        ax.plot(
            [row["epoch"] for row in history],
            [row["test_acc"] for row in history],
            linewidth=2,
            label=f"{name} best={max(row['test_acc'] for row in history):.4f}",
        )
    ax.set_title("CIFAR-100 Test Accuracy Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(comparison_dir / "test_accuracy_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    ax = plt.gca()
    for name, history in histories.items():
        ax.plot(
            [row["epoch"] for row in history],
            [row["test_loss"] for row in history],
            linewidth=2,
            label=name,
        )
    ax.set_title("CIFAR-100 Test Loss Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Loss")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(comparison_dir / "test_loss_comparison.png", dpi=300)
    plt.close()

    summary_rows = []
    for name, history in histories.items():
        best = max(history, key=lambda row: row["test_acc"])
        final = history[-1]
        summary_rows.append(
            {
                "model_name": name,
                "best_epoch": best["epoch"],
                "best_test_acc": best["test_acc"],
                "best_test_loss": best["test_loss"],
                "final_train_acc": final["train_acc"],
                "final_test_acc": final["test_acc"],
            }
        )

    with (comparison_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "best_epoch",
                "best_test_acc",
                "best_test_loss",
                "final_train_acc",
                "final_test_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    names = [row["model_name"] for row in summary_rows]
    best_acc = [row["best_test_acc"] for row in summary_rows]
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    bars = ax.bar(names, best_acc, color=["#4c78a8", "#f58518", "#54a24b"][: len(names)])
    ax.set_title("Best Test Accuracy by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Best Test Accuracy")
    ax.set_ylim(0, max(best_acc) * 1.15 if best_acc else 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, best_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(comparison_dir / "best_accuracy_bar.png", dpi=300)
    plt.close()


def train_model(model_name, args, train_loader, test_loader, device):
    args.model_name = model_name
    output_dir = args.runs_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(model_name).to(device)
    optimizer = build_optimizer(model, model_name, args)
    scheduler, scheduler_mode = build_scheduler(optimizer, args, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    history = []
    best_test_acc = -math.inf
    best_model_path = output_dir / "best_model.pth"
    log_path = output_dir / "training_log.csv"

    print(f"\n=== Training {model_name} ===")
    print(f"Output dir: {output_dir}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Scheduler: {args.scheduler}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scheduler_mode, scaler, device, args, epoch
        )
        test_loss, test_acc = evaluate(model, test_loader, device, args, epoch)
        if scheduler is not None and scheduler_mode == "epoch":
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": current_lr,
        }
        history.append(row)
        save_training_log(log_path, history)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "test_acc": test_acc,
                    "args": vars(args).copy(),
                },
                best_model_path,
            )

        print(
            f"{model_name} Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"best_test_acc={best_test_acc:.4f} lr={current_lr:.6g}"
        )

    plot_model_curves(output_dir, model_name, history)
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-100 models with PyTorch.")
    parser.add_argument("--model_name", choices=MODEL_NAMES, default="vit_tony")
    parser.add_argument("--train_all", action="store_true")
    parser.add_argument("--compare_only", action="store_true")
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--runs_dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--vit_lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--optimizer", choices=["auto", "sgd", "adamw"], default="auto")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", choices=["cosine", "onecycle", "none"], default="cosine")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument("--cutmix_alpha", type=float, default=0.0)
    parser.add_argument("--randaugment", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.pin_memory = not args.no_pin_memory
    args.amp = not args.no_amp
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.compare_only:
        plot_comparison(args.runs_dir, MODEL_NAMES)
        return

    train_loader, test_loader = build_dataloaders(args)
    selected_models = MODEL_NAMES if args.train_all else [args.model_name]

    for model_name in selected_models:
        original_lr = args.lr
        original_weight_decay = args.weight_decay
        if model_name == "vit_tony" and args.optimizer == "auto":
            args.lr = args.vit_lr
            args.weight_decay = 0.05
        train_model(model_name, args, train_loader, test_loader, device)
        args.lr = original_lr
        args.weight_decay = original_weight_decay

    plot_comparison(args.runs_dir, selected_models)
    print("Done.")


if __name__ == "__main__":
    main()
