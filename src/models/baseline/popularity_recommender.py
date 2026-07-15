"""Recomendador baseado em popularidade global dos itens."""

import pandas as pd

from src.models.baseline.base import Recommender


def _popularity_ranking(interactions: pd.DataFrame) -> list[int]:
    """Ordena itens por peso total acumulado de interações.

    Args:
        interactions: DataFrame com colunas ``itemid`` e ``weight``.

    Returns:
        Lista de item_ids em ordem decrescente de popularidade.
    """
    return (
        interactions.groupby("itemid")["weight"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )


class PopularityRecommender(Recommender):
    """Recomenda os itens globalmente mais populares, excluindo já vistos.

    Baseline não-personalizado: todos os usuários recebem o mesmo ranking
    global, filtrado pelos itens que cada um já consumiu.
    """

    def __init__(self) -> None:
        """Inicializa o recomendador com estado vazio."""
        self._ranking: list[int] = []
        self._interactions: pd.DataFrame = pd.DataFrame()

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        """Aprende o ranking global de popularidade.

        Args:
            interactions: DataFrame com colunas ``visitorid``, ``itemid``
                e ``weight``.
        """
        self._interactions = interactions.copy()
        self._ranking = _popularity_ranking(interactions)
        return self

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Retorna os k itens mais populares não vistos pelo usuário.

        Args:
            user_id: Identificador do usuário.
            k: Número máximo de recomendações.

        Returns:
            Lista de item_ids ordenados por popularidade decrescente.
        """
        seen = set(
            self._interactions[self._interactions["visitorid"] == user_id]["itemid"]
        )
        return [item for item in self._ranking if item not in seen][:k]
