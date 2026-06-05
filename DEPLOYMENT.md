# Docker Deployment Guide

本教程用于让其他人在本地快速部署并运行本项目。项目支持两种方式：

- Docker：最适合复现实验流程、快速测试、避免污染本机 Python 环境。
- Conda：最适合在本机 GPU 上长时间训练。

## 1. 克隆仓库

```powershell
git clone https://github.com/520love-killer/sklearn.git
cd sklearn
```

如果你使用的是本项目分支：

```powershell
git checkout vit-tony-cifar100
```

## 2. Docker CPU 部署

默认 Dockerfile 使用 CPU 版 PyTorch，适合所有电脑直接运行。

构建镜像：

```powershell
docker build -t cifar100-vit-tony .
```

如果 Docker Desktop 配置的镜像源不可用，例如出现 `502 Bad Gateway`，可以临时换一个基础镜像地址：

```powershell
docker build --build-arg PYTHON_BASE_IMAGE=mcr.microsoft.com/devcontainers/python:1-3.12-bookworm -t cifar100-vit-tony .
```

查看训练脚本参数：

```powershell
docker run --rm cifar100-vit-tony
```

运行 1 epoch 流程测试：

```powershell
docker compose up --build
```

这个命令会自动：

1. 构建 Docker 镜像。
2. 下载 CIFAR-100 到本地 `data/`。
3. 训练 ViT-Tony 1 个 epoch。
4. 把训练输出保存到本地 `runs/`。

## 3. Docker 训练单个模型

ViT-Tony：

```powershell
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/runs:/app/runs cifar100-vit-tony python train_cifar100_multi_models.py --model_name vit_tony --epochs 50 --batch_size 128 --num_workers 0
```

ResNet-18：

```powershell
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/runs:/app/runs cifar100-vit-tony python train_cifar100_multi_models.py --model_name resnet18 --epochs 50 --batch_size 128 --num_workers 0
```

EfficientNetV2-S：

```powershell
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/runs:/app/runs cifar100-vit-tony python train_cifar100_multi_models.py --model_name efficientnetv2_s --epochs 50 --batch_size 128 --num_workers 0
```

## 4. Docker 训练全部模型

CPU 训练全部模型会很慢，只建议用于流程验证或服务器环境：

```powershell
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/runs:/app/runs cifar100-vit-tony python train_cifar100_multi_models.py --train_all --epochs 50 --batch_size 128 --num_workers 0
```

重新生成模型对比图：

```powershell
docker run --rm -v ${PWD}/runs:/app/runs cifar100-vit-tony python train_cifar100_multi_models.py --compare_only
```

输出位置：

```text
runs/vit_tony/
runs/resnet18/
runs/efficientnetv2_s/
runs/model_comparison/
```

## 5. Docker GPU 说明

如果你的电脑支持 NVIDIA GPU、Docker Desktop WSL2 和 NVIDIA Container Toolkit，可以构建 CUDA 版 PyTorch 镜像：

```powershell
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 -t cifar100-vit-tony:cu126 .
```

检查容器内 CUDA 是否可用：

```powershell
docker run --rm --gpus all cifar100-vit-tony:cu126 python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

GPU 训练 ViT-Tony：

```powershell
docker run --rm --gpus all -v ${PWD}/data:/app/data -v ${PWD}/runs:/app/runs cifar100-vit-tony:cu126 python train_cifar100_multi_models.py --model_name vit_tony --epochs 50 --batch_size 128 --num_workers 2
```

如果上面 CUDA 检查输出 `False`，说明 Docker 没有正确拿到显卡。可以先用 Conda 在本机训练。

## 6. Conda GPU 训练

创建环境：

```powershell
conda create -n vit-tony python=3.12 -y
conda activate vit-tony
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install tqdm tensorboard matplotlib numpy
```

检查 GPU：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

训练全部模型：

```powershell
python train_cifar100_multi_models.py --train_all --epochs 50 --batch_size 128 --num_workers 2
```

## 7. 常见问题

### CIFAR-100 会下载到哪里？

默认下载到项目根目录的 `data/` 文件夹。这个文件夹被 `.gitignore` 忽略，不会提交到 GitHub。

### 训练结果在哪里？

默认输出到 `runs/`：

- `best_model.pth`：最佳模型权重。
- `training_log.csv`：每个 epoch 的训练指标。
- `loss_curve.png`：loss 曲线。
- `accuracy_curve.png`：accuracy 曲线。

`runs/` 也被 `.gitignore` 忽略，不会把大文件传到 GitHub。

### 为什么 Docker 默认 `num_workers=0`？

Windows + Docker 环境下，多进程 DataLoader 有时会带来额外兼容问题。为了让别人第一次部署更稳，Docker 示例默认使用 `--num_workers 0`。如果你在 Linux/GPU 服务器上运行，可以改成 `2` 或 `4`。

### Docker 和 Conda 选哪个？

- 想快速复现流程：选 Docker。
- 想长时间训练、充分使用本机 RTX GPU：选 Conda。
- 想把项目交给别人部署：优先给 Docker 教程。
