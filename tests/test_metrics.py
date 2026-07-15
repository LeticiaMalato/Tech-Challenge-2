"""Testes unitários para as métricas de avaliação de recomendação."""

import pandas as pd
import pytest

from src.models.baseline.metrics import (
    evaluate_recommender,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class _FakeRecommender:
    """Recomendador stub que devolve uma lista fixa por usuário.

    Usado para testar ``evaluate_recommender`` sem depender de nenhum
    modelo treinado de verdade.
    """

    def __init__(self, recommendations: dict[int, list[int]]) -> None:
        """Armazena as recomendações fixas por usuário.

        Args:
            recommendations: Mapeamento de user_id para lista de item_ids.
        """
        self._recommendations = recommendations

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Retorna a lista fixa de recomendações do usuário, truncada em k.

        Args:
            user_id: Identificador do usuário.
            k: Corte da lista retornada.

        Returns:
            Lista de item_ids recomendados.
        """
        return self._recommendations.get(user_id, [])[:k]


# hit_rate_at_k


def test_hit_rate_at_k_hit_dentro_do_topk() -> None:
    """Retorna 1.0 quando ao menos um item relevante está no top-k."""
    assert hit_rate_at_k([1, 2, 3], {3, 99}, k=3) == 1.0


def test_hit_rate_at_k_sem_hit() -> None:
    """Retorna 0.0 quando nenhum item relevante aparece no top-k."""
    assert hit_rate_at_k([1, 2, 3], {99}, k=3) == 0.0


def test_hit_rate_at_k_hit_fora_do_corte() -> None:
    """Um acerto além da posição k não conta."""
    assert hit_rate_at_k([1, 2, 3, 4], {4}, k=2) == 0.0


def test_hit_rate_at_k_lista_vazia() -> None:
    """Lista de recomendações vazia retorna 0.0."""
    assert hit_rate_at_k([], {1}, k=5) == 0.0


# precision_at_k


def test_precision_at_k_dois_de_cinco() -> None:
    """2 acertos em 5 recomendações resulta em precisão 0.4."""
    recommended = [1, 2, 3, 4, 5]
    relevant = {2, 4, 99}
    assert precision_at_k(recommended, relevant, k=5) == pytest.approx(0.4)


def test_precision_at_k_lista_vazia() -> None:
    """Lista de recomendações vazia retorna 0.0."""
    assert precision_at_k([], {1}, k=5) == 0.0


def test_precision_at_k_k_zero() -> None:
    """k=0 retorna 0.0 em vez de dividir por zero."""
    assert precision_at_k([1, 2], {1}, k=0) == 0.0


# recall_at_k


def test_recall_at_k_metade_recuperada() -> None:
    """Recupera 1 de 2 itens relevantes: recall 0.5."""
    assert recall_at_k([1, 2, 3], {1, 99}, k=3) == pytest.approx(0.5)


def test_recall_at_k_sem_relevantes() -> None:
    """Conjunto de relevantes vazio retorna 0.0."""
    assert recall_at_k([1, 2], set(), k=2) == 0.0


def test_recall_at_k_k_zero() -> None:
    """k=0 retorna 0.0 em vez de dividir por zero."""
    assert recall_at_k([1, 2], {1}, k=0) == 0.0


# ndcg_at_k


def test_ndcg_at_k_ranking_perfeito() -> None:
    """Todos os relevantes nas primeiras posições geram NDCG máximo (1.0)."""
    assert ndcg_at_k([1, 2], {1, 2}, k=2) == pytest.approx(1.0)


def test_ndcg_at_k_ranking_pior_penaliza() -> None:
    """Relevante em posição pior reduz o NDCG abaixo do ranking ideal."""
    perfeito = ndcg_at_k([1, 2, 3], {1}, k=3)
    invertido = ndcg_at_k([3, 2, 1], {1}, k=3)
    assert invertido < perfeito


def test_ndcg_at_k_sem_relevantes() -> None:
    """Conjunto de relevantes vazio retorna 0.0."""
    assert ndcg_at_k([1, 2], set(), k=2) == 0.0


def test_ndcg_at_k_lista_vazia() -> None:
    """Lista de recomendações vazia retorna 0.0."""
    assert ndcg_at_k([], {1}, k=2) == 0.0


# mrr_at_k


def test_mrr_at_k_acerto_na_primeira_posicao() -> None:
    """Acerto na posição 1 retorna MRR 1.0."""
    assert mrr_at_k([1, 2, 3], {1}, k=3) == pytest.approx(1.0)


def test_mrr_at_k_acerto_na_segunda_posicao() -> None:
    """Acerto na posição 2 retorna MRR 0.5 (1/posição)."""
    assert mrr_at_k([2, 1, 3], {1}, k=3) == pytest.approx(0.5)


def test_mrr_at_k_sem_acerto() -> None:
    """Nenhum item relevante no top-k retorna 0.0."""
    assert mrr_at_k([1, 2, 3], {99}, k=3) == 0.0


def test_mrr_at_k_acerto_fora_do_corte() -> None:
    """Acerto após a posição k não conta."""
    assert mrr_at_k([1, 2, 3], {3}, k=2) == 0.0


# evaluate_recommender


@pytest.fixture
def sample_interactions() -> pd.DataFrame:
    """DataFrame de teste com dois usuários e seus itens relevantes."""
    return pd.DataFrame({"visitorid": [1, 1, 2], "itemid": [10, 20, 30]})


def test_evaluate_recommender_estrutura_por_k(
    sample_interactions: pd.DataFrame,
) -> None:
    """Retorna uma linha por valor de k, com as 5 colunas de métrica esperadas."""
    fake = _FakeRecommender({1: [10, 99], 2: [30]})
    results = evaluate_recommender(fake, sample_interactions, k_values=[1, 5])

    assert list(results["k"]) == [1, 5]
    for metric in ("hit_rate", "precision", "recall", "ndcg", "mrr"):
        assert metric in results.columns


def test_evaluate_recommender_hit_rate_perfeito(
    sample_interactions: pd.DataFrame,
) -> None:
    """Quando o recomendador sempre acerta, hit_rate deve ser 1.0."""
    fake = _FakeRecommender({1: [10], 2: [30]})
    results = evaluate_recommender(fake, sample_interactions, k_values=[5])

    hit_rate = results.loc[results["k"] == 5, "hit_rate"].iloc[0]
    assert hit_rate == pytest.approx(1.0)
