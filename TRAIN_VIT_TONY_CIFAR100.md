# ViT-Tony CIFAR-100 训练说明

## 数据要求

训练脚本默认读取已经预处理完成的 `.npz` 文件：

```text
outputs/preprocessed_cifar100/train_preprocessed.npz
outputs/preprocessed_cifar100/test_preprocessed.npz
```

每个文件需要包含：

```text
images: (N, 3, 32, 32)
labels: (N,)
```

其中 `images` 已经完成 ToTensor、归一化和 Z-score 标准化，`labels` 是 CIFAR-100 的 `0-99` 整数标签。

## 运行命令

```bash
python train_vit_tony_cifar100.py
```

常用参数：

```bash
python train_vit_tony_cifar100.py --epochs 50 --batch-size 128 --num-workers 2
```

## 输出文件

默认输出目录：

```text
runs/vit_tony_cifar100/
```

训练完成后会生成：

```text
best_vit_tony_cifar100.pth
training_log.csv
loss_curve.png
accuracy_curve.png
```

## 模型结构

脚本中的 ViT-Tony 使用轻量化配置：

```text
image_size = 32
patch_size = 4
embed_dim = 192
depth = 6
num_heads = 6
num_classes = 100
```

包含 Patch Embedding、CLS Token、Position Embedding、Transformer Encoder 和 Classification Head。
