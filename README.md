# ViT on CIFAR-10

这是一个适合新手学习和云服务器训练的 PyTorch Vision Transformer 项目。模型是一个轻量 ViT-Tiny，输入为 CIFAR-10 的 `32x32` 图片，默认把图片切成 `4x4` patch。

## 1. 用 VSCode Remote SSH 连接云服务器

1. 在本机 VSCode 安装插件：`Remote - SSH`。
2. 打开命令面板：`Ctrl + Shift + P`。
3. 选择：`Remote-SSH: Add New SSH Host...`。
4. 输入你的服务器连接命令，例如：

```bash
ssh root@你的服务器公网IP
```

5. 连接后，在远程 VSCode 里打开服务器上的项目目录。

如果你还没有上传项目，可以在本机 PowerShell 使用：

```powershell
scp -r E:\develop\sklearn_model_train\vit_CIFAR-10 root@你的服务器公网IP:/root/
```

之后在 VSCode Remote SSH 中打开：

```bash
/root/vit_CIFAR-10
```

## 2. 在云服务器配置 Python 环境

建议使用 Conda：

```bash
conda create -n vit-cifar10 python=3.10 -y
conda activate vit-cifar10
```

安装 PyTorch 时要匹配你的 CUDA 版本。常见 CUDA 12.1 环境可以用：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install tqdm tensorboard
```

更稳妥的方式是打开 PyTorch 官方安装页，根据你的服务器系统、包管理器和 CUDA 版本生成命令：

```text
https://pytorch.org/get-started/
```

如果你不确定 CUDA 是否可用，运行：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果这里输出 `False`，说明当前 Python 环境没有正确安装 GPU 版 PyTorch，或者服务器 GPU/驱动不可用。

## 3. 开始训练

在项目根目录运行：

```bash
python train.py --epochs 100 --batch-size 128 --num-workers 4
```

也可以在 VSCode 左侧选择 Run and Debug，启动 `Train ViT CIFAR-10` 配置。

程序会自动下载 CIFAR-10 到 `data/`，并把训练结果保存到：

```text
runs/vit_cifar10/
```

主要输出文件：

- `best.pt`：验证集准确率最高的模型权重
- `last.pt`：最后一个 epoch 的断点
- `config.json`：本次训练参数
- `tensorboard/`：TensorBoard 日志

## 4. 断点续训

如果训练中断，可以继续：

```bash
python train.py --resume runs/vit_cifar10/last.pt --epochs 100
```

## 5. 查看训练曲线

在服务器项目目录运行：

```bash
tensorboard --logdir runs/vit_cifar10/tensorboard --host 0.0.0.0 --port 6006
```

如果云服务器安全组开放了 `6006` 端口，可以在浏览器访问：

```text
http://你的服务器公网IP:6006
```

更推荐用 VSCode Remote SSH 的端口转发功能，把服务器 `6006` 转发到本机。

## 6. 常用调参建议

- 显存不够：降低 `--batch-size`，例如 `64` 或 `32`。
- 训练太慢：先用 `--epochs 10` 验证流程。
- 想提高准确率：增加 epoch、使用更强数据增强、调大学习率 warmup 或换更大模型。
- 只想确认代码能跑通：

```bash
python train.py --epochs 1 --batch-size 64 --num-workers 2
```

## 项目结构

```text
.
├── README.md
├── requirements.txt
├── train.py
└── src
    └── vit.py
```
