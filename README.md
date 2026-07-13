## Setup

Pré-requisitos: Python 3.11+, [Poetry](https://python-poetry.org/) ≥ 2.3.

```powershell
# clona o repositório
git clone <url-do-repositorio>
cd Tech-Challenge-2

# instala dependências (prod + dev)
poetry install --with dev

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

## Experimentos e Model Registry (MLflow)

Para visualizar os runs, métricas por época (do modelo neural) e o
histórico do Model Registry:

```powershell
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre em `http://localhost:5000`. O melhor modelo entre todos os candidatos
(baselines + neural) é automaticamente promovido a **Production** no
Registry ao final de cada execução do stage `train_and_evaluate`.

## Testes

```powershell
poetry run pytest -v
```

Cobertura: métricas de avaliação (`hit_rate`, `precision`, `recall`,
`ndcg`, `mrr`), filtro k-core, split temporal, encoders de ID, os 4
recomendadores baseline, as factories de modelo/pré-processador, e os
componentes centrais do pipeline neural (`EarlyStopping`, negative
sampling do `InteractionDataset`).

## Qualidade de código

```powershell
poetry run pre-commit run --all-files
```

Roda `ruff` (lint + formatação) sobre todo o projeto. Convenções: funções
≤ 20 linhas, type hints obrigatórios, docstrings estilo Google, sem
`iterrows()`, sem dead code.
