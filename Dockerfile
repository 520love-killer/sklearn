ARG PYTHON_BASE_IMAGE=python:3.12-slim
FROM ${PYTHON_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

WORKDIR /app

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir tqdm tensorboard matplotlib numpy

COPY . .

CMD ["python", "train_cifar100_multi_models.py", "--help"]
