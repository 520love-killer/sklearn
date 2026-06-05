# Project Structure

```text
vit_CIFAR-10/
├── README.md
├── DEPLOYMENT.md
├── Dockerfile
├── docker-compose.yml
├── PROJECT_STRUCTURE.md
├── TRAIN_VIT_TONY_CIFAR100.md
├── requirements.txt
├── train.py
├── train_vit_tony_cifar100.py
├── train_cifar100_multi_models.py
├── preprocess_cifar100_numpy.py
├── cifar100_preprocessing.py
├── cifar100_feature_pipeline.py
├── cifar100_feature_pipeline_numpy_fallback.py
├── explore_cifar100.py
├── visualize_cifar.py
├── run_vit_tony_test.ps1
├── run_vit_tony_train.ps1
├── src/
├── reports/
├── outputs/
├── data/
└── runs/
```

## Files to Commit

- `Dockerfile`, `.dockerignore`, `docker-compose.yml`: local Docker deployment files.
- `DEPLOYMENT.md`: Docker and Conda deployment guide.
- `src/vit.py`: reusable ViT implementation.
- `train_cifar100_multi_models.py`: main CIFAR-100 multi-model training and comparison script.
- `train_vit_tony_cifar100.py`: ViT-Tony training script for preprocessed `.npz` data.
- `preprocess_cifar100_numpy.py`: creates preprocessed CIFAR-100 `.npz` files.
- `explore_cifar100.py`, `visualize_cifar.py`: data exploration and visualization utilities.
- `reports/*.md`: written experiment notes and summaries.
- `reports/figures/*.png`: report figures copied from local experiments.
- `outputs/*.png`: small static figures used for reports.
- `run_vit_tony_test.ps1`, `run_vit_tony_train.ps1`: local PowerShell shortcuts for ViT-Tony.
- `.vscode/`: optional editor launch/settings files that are already tracked.

## Files Not to Commit

The following are intentionally ignored by `.gitignore`:

- `.venv/`, `venv/`: local Python virtual environments.
- `.idea/`: local PyCharm state.
- `data/`: downloaded CIFAR datasets.
- `runs/`, `runs_smoke/`: generated training outputs.
- `outputs/preprocessed_cifar100/`: generated `.npz` feature/preprocessing files.
- `*.pth`, `*.pt`, `*.ckpt`: model checkpoints.
- `*.npz`: generated NumPy datasets/features.

## Main Workflows

Create or refresh preprocessed CIFAR-100 data:

```powershell
python preprocess_cifar100_numpy.py
```

Train ViT-Tony from preprocessed data:

```powershell
python train_vit_tony_cifar100.py --epochs 50 --batch-size 128 --num-workers 2
```

Train and compare all supported models:

```powershell
python train_cifar100_multi_models.py --train_all --epochs 50 --batch_size 128 --num_workers 2
```

Regenerate comparison plots from existing `runs/` logs:

```powershell
python train_cifar100_multi_models.py --compare_only
```
