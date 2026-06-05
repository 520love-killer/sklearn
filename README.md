# CIFAR-100 ViT-Tony 多模型训练项目

这是一个基于 PyTorch 的 CIFAR-100 图像分类训练项目，当前支持：

- ViT-Tony
- ResNet-18
- EfficientNetV2-S

项目包含 CIFAR-100 数据探索、预处理、ViT-Tony 单模型训练、多模型训练对比和结果可视化。训练数据、模型权重和运行日志属于本地生成产物，默认不会提交到 GitHub。

## 环境要求

推荐使用 Conda 创建独立环境：

```powershell
conda create -n vit-tony python=3.12 -y
conda activate vit-tony
pip install -r requirements.txt
```

GPU 训练需要安装与你本机 CUDA/显卡驱动匹配的 PyTorch。检查 GPU 是否可用：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果输出 `True` 和显卡名称，训练会自动使用 GPU。

## 快速训练

训练 ViT-Tony、ResNet-18、EfficientNetV2-S 三个模型并生成对比图：

```powershell
python train_cifar100_multi_models.py --train_all --epochs 50 --batch_size 128 --num_workers 2
```

只训练 ViT-Tony：

```powershell
python train_cifar100_multi_models.py --model_name vit_tony --epochs 50 --batch_size 128 --num_workers 2
```

只根据已有日志重新生成模型对比图：

```powershell
python train_cifar100_multi_models.py --compare_only
```

## Docker 本地部署

如果只想快速在本机部署项目，可以使用 Docker。默认镜像安装 CPU 版 PyTorch，适合验证代码流程；完整 50 epoch 训练建议使用本机 GPU/Conda 环境。

完整部署教程见 `DEPLOYMENT.md`。

构建镜像：

```powershell
docker build -t cifar100-vit-tony .
```

查看训练脚本参数：

```powershell
docker run --rm cifar100-vit-tony
```

运行 1 epoch 流程测试：

```powershell
docker compose up --build
```

在 Docker 中训练 ViT-Tony：

```powershell
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/runs:/app/runs cifar100-vit-tony python train_cifar100_multi_models.py --model_name vit_tony --epochs 50 --batch_size 128 --num_workers 0
```

在 Docker 中重新生成对比图：

```powershell
docker run --rm -v ${PWD}/runs:/app/runs cifar100-vit-tony python train_cifar100_multi_models.py --compare_only
```

训练结果默认保存到：

```text
runs/<model_name>/
```

每个模型目录会生成：

- `best_model.pth`：测试集准确率最高的模型权重
- `training_log.csv`：每个 epoch 的训练/测试指标
- `loss_curve.png`：loss 曲线
- `accuracy_curve.png`：accuracy 曲线

多模型对比结果保存到：

```text
runs/model_comparison/
```

## ViT-Tony 预处理数据训练

如果使用已经预处理好的 CIFAR-100 `.npz` 数据，可以运行：

```powershell
python train_vit_tony_cifar100.py --epochs 50 --batch-size 128 --num-workers 2
```

预处理数据默认路径：

```text
outputs/preprocessed_cifar100/train_preprocessed.npz
outputs/preprocessed_cifar100/test_preprocessed.npz
```

这些 `.npz` 文件是本地数据产物，默认不会提交到 GitHub。

## 当前实验结果

50 epoch 多模型训练结果摘要：

| 模型 | 最佳 Epoch | 最佳测试准确率 | 最终训练准确率 | 最终测试准确率 |
| --- | ---: | ---: | ---: | ---: |
| ViT-Tony | 45 | 58.18% | 77.58% | 58.05% |
| ResNet-18 | 47 | 76.33% | 99.86% | 76.23% |
| EfficientNetV2-S | 49 | 55.57% | 60.27% | 55.48% |

详细说明见 `reports/model_comparison_summary.md`。

模型原理、数据处理、结果图表和改进建议见 `reports/model_and_data_explanation.md`。

## GitHub 提交说明

`.gitignore` 已经排除以下本地内容：

- `data/`：CIFAR-100 原始数据
- `runs/`、`runs_smoke/`：训练日志、权重、曲线图
- `.venv/`、`.idea/`：本机环境和 IDE 状态
- `*.pth`、`*.pt`、`*.npz`：模型权重和预处理数据

因此整理后可以直接检查并提交：

```powershell
git status
git add .
git commit -m "Add CIFAR-100 multi-model training pipeline"
```

## 项目结构

完整结构说明见 `PROJECT_STRUCTURE.md`。

## 重要文档

- `DEPLOYMENT.md`：Docker 和 Conda 本地部署教程。
- `reports/model_and_data_explanation.md`：模型介绍、数据处理、结果图表和改进建议。
- `reports/model_comparison_summary.md`：50 epoch 多模型训练结果摘要。
