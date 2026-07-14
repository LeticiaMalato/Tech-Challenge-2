# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — builder: instala dependências de produção (torch CPU)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# requirements-prod.txt é gerado via: poetry export --only main --without-hashes
# (sem torch/nvidia — instalamos torch CPU abaixo)
COPY requirements-prod.txt ./

RUN pip install --prefix=/install -r requirements-prod.txt \
    && pip install --prefix=/install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.3,<3.0"

# ---------------------------------------------------------------------------
# Stage 2 — train: imagem com código para rodar o pipeline / treino
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS train

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_TRACKING_URI=http://mlflow:5000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY src ./src
COPY scripts ./scripts
COPY pyproject.toml params.yaml dvc.yaml ./

CMD ["python", "-m", "src.train.run"]

# ---------------------------------------------------------------------------
# Stage 3 — runtime: API de recomendação (Render / serviço api)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    MODEL_PATH=/app/models/checkpoints/mlp/mlp_best.pt

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/models/checkpoints/mlp \
    && chown -R appuser:appuser /app

COPY --from=builder /install /usr/local
COPY src ./src
COPY pyproject.toml ./
COPY artifacts/ ./artifacts/

# Preferência: artifacts/mlp_best.pt (para deploy no Render a partir do Git)
RUN if [ -f /app/artifacts/mlp_best.pt ]; then \
      cp /app/artifacts/mlp_best.pt /app/models/checkpoints/mlp/mlp_best.pt; \
    fi \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health')"

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}"]
