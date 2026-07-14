"""Testes da API FastAPI de recomendação."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.models.neural.recommender import MLPConfig, MLPRecommender


def _train_tiny_checkpoint(tmp_path: Path) -> Path:
    """Treina um MLP mínimo e retorna o path do artefato."""
    interactions = pd.DataFrame(
        {
            "visitorid": [10, 10, 10, 20, 20, 20, 30, 30, 30],
            "itemid": [100, 101, 102, 100, 103, 104, 101, 102, 105],
            "timestamp": list(range(9)),
        }
    )
    config = MLPConfig(
        embed_dim=4,
        neg_ratio=1,
        batch_size=4,
        max_epochs=2,
        patience=2,
        seed=42,
        device="cpu",
    )
    model = MLPRecommender(config=config, checkpoint_dir=tmp_path)
    model.fit(interactions)
    return tmp_path / "mlp_best.pt"


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Cliente HTTP com modelo de teste carregado via lifespan."""
    checkpoint = _train_tiny_checkpoint(tmp_path)
    monkeypatch.setenv("MODEL_PATH", str(checkpoint))

    from src.api.app import app

    with TestClient(app) as client:
        yield client


def test_health_ok(api_client: TestClient) -> None:
    """GET /health deve retornar status ok quando o modelo está carregado."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_links(api_client: TestClient) -> None:
    """GET / deve expor links dos endpoints principais."""
    response = api_client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["health"] == "/health"
    assert "recommend" in payload


def test_recommend_conhecido(api_client: TestClient) -> None:
    """GET /recommend/{user_id} retorna top-k para usuário do vocabulário."""
    response = api_client.get("/recommend/10", params={"k": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == 10
    assert payload["k"] == 3
    assert len(payload["items"]) == 3


def test_recommend_desconhecido_retorna_404(api_client: TestClient) -> None:
    """Usuário fora do vocabulário deve retornar 404."""
    response = api_client.get("/recommend/999")
    assert response.status_code == 404
