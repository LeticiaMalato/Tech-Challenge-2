"""API FastAPI de recomendação baseada no MLP treinado.

Carrega o artefato de inferência uma vez no startup e expõe endpoints
de healthcheck e recomendação top-k por ``user_id``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.models.neural.recommender import MLPRecommender

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path("models/checkpoints/mlp/mlp_best.pt")
_recommender: MLPRecommender | None = None


def _resolve_model_path() -> Path:
    """Resolve o caminho do artefato via env ``MODEL_PATH`` ou default."""
    return Path(os.getenv("MODEL_PATH", str(_DEFAULT_MODEL_PATH)))


def get_recommender() -> MLPRecommender:
    """Retorna o recomendador carregado no startup.

    Raises:
        RuntimeError: Se a API ainda não inicializou o modelo.
    """
    if _recommender is None:
        raise RuntimeError("Modelo não carregado. Verifique o startup da API.")
    return _recommender


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Carrega o artefato de inferência no startup e libera no shutdown."""
    global _recommender
    path = _resolve_model_path()
    logger.info("Carregando modelo de %s", path)
    _recommender = MLPRecommender.load(path, device="cpu")
    yield
    _recommender = None


app = FastAPI(
    title="RetailRocket Recommender API",
    description="API de recomendação do Tech Challenge Fase 02 (MLP PyTorch).",
    version="0.1.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    """Resposta do healthcheck."""

    status: str = Field(examples=["ok"])


class RecommendResponse(BaseModel):
    """Resposta de recomendações top-k."""

    user_id: int
    k: int
    items: list[int]


class RootResponse(BaseModel):
    """Mensagem raiz com links dos endpoints principais."""

    message: str
    health: str = "/health"
    recommend: str = "/recommend/{user_id}?k=10"
    docs: str = "/docs"


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    """Retorna links úteis da API."""
    return RootResponse(message="RetailRocket recommender API")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Healthcheck usado pelo Render e pelo orquestrador Docker."""
    get_recommender()
    return HealthResponse(status="ok")


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(
    user_id: int,
    k: int = Query(default=10, ge=1, le=100),
) -> RecommendResponse:
    """Retorna os k itens mais recomendados para o usuário.

    Args:
        user_id: Identificador do visitante (``visitorid`` do dataset).
        k: Número máximo de recomendações (1–100).

    Raises:
        HTTPException: 404 se o usuário não existir no vocabulário de treino.
    """
    items = get_recommender().recommend(user_id=user_id, k=k)
    if not items:
        raise HTTPException(
            status_code=404,
            detail=f"Usuário {user_id} não encontrado no vocabulário de treino.",
        )
    return RecommendResponse(user_id=user_id, k=k, items=items)
