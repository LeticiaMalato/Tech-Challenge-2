## Setup

Pré-requisitos: Python 3.11+, [Poetry](https://python-poetry.org/) ≥ 2.3.

```powershell
# clona o repositório
git clone <url-do-repositorio>
cd Tech-Challenge-2

# instala dependências (prod + dev)
poetry install --with dev --no-root

# copia o template de variáveis de ambiente
Copy-Item .env.example .env
# edita o .env se quiser sobrescrever algum default (paths, MLflow, seed)

# ativa os hooks de lint automático
poetry run pre-commit install
```

Todas as configurações (paths de dados, URI do MLflow, seed) são
centralizadas em `src/config.py` via Pydantic Settings, sobrescrevíveis
pelo `.env` sem alterar código-fonte.

## Rodando o pipeline

O pipeline completo é orquestrado pelo DVC em 5 stages:
`preprocess_events → preprocess_categories → preprocess_item_properties →
feature_eng → train_and_evaluate`.

```powershell
poetry run dvc repro
```

Isso baixa/valida os dados brutos, aplica k-core + split temporal +
encoding, treina os 4 baselines e o modelo neural, avalia todos no mesmo
conjunto de teste e registra o melhor modelo (por NDCG@10) no MLflow Model
Registry.

Para rodar só um stage específico (ex: depois de editar só o feature
engineering):
```powershell
poetry run dvc repro feature_eng
```

O DVC identifica automaticamente quais stages precisam ser reprocessados
com base nas dependências declaradas em `dvc.yaml` — stages a jusante de
uma mudança são recalculados; os demais são reaproveitados do cache.

Após o treino, copie o artefato de inferência para versionar o deploy:

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
Copy-Item models/checkpoints/mlp/mlp_best.pt artifacts/mlp_best.pt
```

## Experimentos e Model Registry (MLflow)

Para visualizar os runs, métricas por época (do modelo neural) e o
histórico do Model Registry:

```powershell
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre em `http://localhost:5000`. O melhor modelo entre todos os candidatos
(baselines + neural) é automaticamente promovido a **Production** no
Registry ao final de cada execução do stage `train_and_evaluate`.

## Docker Compose

Serviços: `mlflow` (tracking UI/server), `train` (pipeline de treino) e
`api` (FastAPI de recomendação).

```powershell
# sobe MLflow em http://localhost:5000
docker compose up -d mlflow

# treina (profile train) apontando para o MLflow do compose
docker compose --profile train run --rm train

# sobe a API em http://localhost:8000 (requer mlp_best.pt local ou em artifacts/)
docker compose up -d --build api
```

## API de recomendação

Com o modelo carregado (`MODEL_PATH` ou default
`models/checkpoints/mlp/mlp_best.pt`):

```powershell
# local (sem Docker)
$env:MODEL_PATH = "models/checkpoints/mlp/mlp_best.pt"
poetry run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

| Endpoint | Descrição |
|---|---|
| `GET /health` | Healthcheck (`{"status":"ok"}`) |
| `GET /recommend/{user_id}?k=10` | Top-k itens para o usuário |
| `GET /docs` | Swagger UI |

Use o **`visitorid` original do RetailRocket** presente no treino
(`data/features/train.parquet`), não índices 0, 1, 2… Usuários fora do
vocabulário retornam **404**.

Exemplos válidos de `user_id` (amostra do treino atual):

`990356`, `1399056`, `90447`, `979664`, `1346730`, `1296675`, `1222911`,
`954142`, `70597`, `1061274`

Para listar outros IDs locais:

```powershell
poetry run python -c "import pandas as pd; print(pd.read_parquet('data/features/train.parquet')['visitorid'].drop_duplicates().head(20).tolist())"
```

Exemplo (local ou Render):

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod "http://localhost:8000/recommend/990356?k=5"

# produção
Invoke-RestMethod https://retailrocket-recommender-api.onrender.com/health
Invoke-RestMethod "https://retailrocket-recommender-api.onrender.com/recommend/990356?k=5"
```

## Deploy no Render

1. Rode o pipeline e publique o artefato em `artifacts/mlp_best.pt` (commitado).
2. No [Render](https://render.com), use **Blueprint** com o `render.yaml` do repo,
   ou **New → Web Service** a partir da imagem Docker publicada.
3. Health check path: `/health`.
4. Env sugeridas: `MODEL_PATH=/app/models/checkpoints/mlp/mlp_best.pt`,
   `PYTHONUNBUFFERED=1` (`PORT` é injetado pelo Render).
5. Após o deploy, a URL pública responde em `/health` e `/recommend/{user_id}`.

O free tier pode hibernar após inatividade e tem pouca RAM — a imagem usa
PyTorch CPU justamente para caber nesse perfil.

## Testes

```powershell
poetry run pytest -v
```

Cobertura: métricas de avaliação (`hit_rate`, `precision`, `recall`,
`ndcg`, `mrr`), filtro k-core, split temporal, encoders de ID, os 4
recomendadores baseline, as factories de modelo/pré-processador, os
componentes centrais do pipeline neural (`EarlyStopping`, negative
sampling do `InteractionDataset`, checkpoint save/load) e a API FastAPI.

## Qualidade de código

```powershell
poetry run pre-commit run --all-files
```

Roda `ruff` (lint + formatação) sobre todo o projeto. Convenções: funções
≤ 20 linhas, type hints obrigatórios, docstrings estilo Google, sem
`iterrows()`, sem dead code.
