import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from src.vit import build_vit_tiny, count_parameters


CIFAR_CONFIGS = {
    "cifar10": {
        "dataset_cls": datasets.CIFAR10,
        "num_classes": 10,
        "output_dir": Path("runs/vit_cifar10"),
        # CIFAR-10 常用归一化参数：把 RGB 三个通道调整到更适合神经网络训练的数值范围。
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "dataset_cls": datasets.CIFAR100,
        "num_classes": 100,
        "output_dir": Path("runs/vit_cifar100"),
        # CIFAR-100 的统计值和 CIFAR-10 不完全一样，单独配置会更规范。
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vision Transformer on CIFAR-10 or CIFAR-100.")
    parser.add_argument("--dataset", choices=sorted(CIFAR_CONFIGS), default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training.")
    return parser.parse_args()


def set_seed(seed):
    # 固定随机种子，方便多次实验结果尽量可复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_dataloaders(dataset_name, data_dir, batch_size, num_workers):
    config = CIFAR_CONFIGS[dataset_name]

    # 训练集使用随机裁剪和水平翻转，这是最基础的数据增强，可以减轻过拟合。
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config["mean"], config["std"]),
        ]
    )
    # 测试集不能使用随机增强，否则每次评估的输入都不同，指标不稳定。
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(config["mean"], config["std"]),
        ]
    )

    dataset_cls = config["dataset_cls"]
    train_set = dataset_cls(
        root=str(data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = dataset_cls(
        root=str(data_dir),
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, epoch):
    # model.train() 会开启 Dropout 等训练时行为。
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 每个 batch 更新前先清空上一轮梯度。
        optimizer.zero_grad(set_to_none=True)
        # AMP 只在 CUDA 上启用，用较低精度加速训练并减少显存占用。
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        # GradScaler 可以避免混合精度训练时梯度下溢。
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 累加样本数加权后的 loss 和正确数，用于计算整个 epoch 的平均指标。
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        progress.set_postfix(
            loss=total_loss / total_samples,
            acc=total_correct / total_samples,
        )

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch):
    # model.eval() 会关闭 Dropout，并让评估结果更稳定。
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=f"Epoch {epoch} eval", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        progress.set_postfix(
            loss=total_loss / total_samples,
            acc=total_correct / total_samples,
        )

    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(output_dir, payload, is_best):
    # last.pt 保存最近一次训练状态；best.pt 保存目前验证准确率最高的状态。
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    torch.save(payload, last_path)
    if is_best:
        torch.save(payload, output_dir / "best.pt")


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    # 断点恢复时，不只恢复模型参数，也恢复优化器、学习率调度器和 AMP 状态。
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint["epoch"] + 1, checkpoint.get("best_acc", 0.0)


def main():
    args = parse_args()
    config = CIFAR_CONFIGS[args.dataset]
    if args.output_dir is None:
        args.output_dir = config["output_dir"]

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_dataloaders(
        args.dataset,
        args.data_dir,
        args.batch_size,
        args.num_workers,
    )

    model = build_vit_tiny(num_classes=config["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 1
    best_acc = 0.0
    if args.resume is not None:
        start_epoch, best_acc = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )

    run_config = vars(args).copy()
    run_config["device"] = str(device)
    run_config["amp"] = use_amp
    run_config["parameters"] = count_parameters(model)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=str)

    writer = SummaryWriter(log_dir=str(args.output_dir / "tensorboard"))
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Model parameters: {count_parameters(model):,}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            epoch,
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device, epoch)
        scheduler.step()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        save_checkpoint(
            args.output_dir,
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_acc": best_acc,
                "args": run_config,
            },
            is_best,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"best_acc={best_acc:.4f}"
        )

    writer.close()
    print(f"Training complete. Best accuracy: {best_acc:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
