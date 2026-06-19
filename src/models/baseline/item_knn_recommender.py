"""Recomendador baseado em similaridade de cosseno entre itens."""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.models.baseline.base import Recommender
from src.utils.sparse_utils import build_item_user_matrix


class ItemKNNRecommender(Recommender):
    """Recomenda itens por similaridade de cosseno item×usuário.

    Para viabilizar o uso em datasets grandes, a matriz item×usuário é
    construída sobre uma amostra dos usuários mais ativos, controlada por
    ``max_users``. A similaridade é calculada sob demanda por item visto,
    evitando alocar a matriz completa n_items×n_items em memória.
    """

    def __init__(self, top_n_neighbors: int = 20, max_users: int = 5000) -> None:
        """Inicializa o recomendador KNN de itens.

        Args:
            top_n_neighbors: Número de vizinhos considerados por item visto.
            max_users: Número máximo de usuários usados para construir a
                matriz item×usuário. Mantém os mais ativos por contagem
                de interações.
        """
        self._top_n_neighbors = top_n_neighbors
        self._max_users = max_users
        self._item_ids: list[int] = []
        self._item_index: dict[int, int] = {}
        self._matrix: object = None
        self._interactions: pd.DataFrame = pd.DataFrame()

    def fit(self, interactions: pd.DataFrame) -> "ItemKNNRecommender":
        """Armazena a matriz item×usuário amostrada para uso no recommend.

        Args:
            interactions: DataFrame com colunas ``visitorid``, ``itemid``
                e ``weight``.
        """
        self._interactions = interactions.copy()

        # mantém apenas os usuários mais ativos para reduzir memória
        top_users = (
            interactions["visitorid"]
            .value_counts()
            .head(self._max_users)
            .index
        )
        sampled = interactions[interactions["visitorid"].isin(top_users)]

        self._matrix, self._item_ids, self._item_index = build_item_user_matrix(sampled)

        return self
    
    def _score_items(self, user_events: pd.DataFrame) -> np.ndarray:
        seen_mask = user_events["itemid"].isin(self._item_index)
        valid_events = user_events[seen_mask]

        if valid_events.empty:
            return np.zeros(len(self._item_ids))

        seen_indices = [self._item_index[iid] for iid in valid_events["itemid"]]
        weights = valid_events["weight"].to_numpy(dtype=float)

        # uma única chamada: (n_seen, n_items) — muito mais eficiente
        sim_matrix = cosine_similarity(
            self._matrix[seen_indices], self._matrix
        )
        scores = (sim_matrix * weights[:, None]).sum(axis=0)
        scores[seen_indices] = -np.inf  # garante que itens vistos não sejam recomendados
        return scores

    
    def recommend(self, user_id: int, k: int) -> list[int]:
        """Retorna os k itens com maior score de similaridade agregado.

        Args:
            user_id: Identificador do usuário.
            k: Número máximo de recomendações.

        Returns:
            Lista de item_ids ordenados por score descendente.
            Retorna lista vazia se o usuário não possui interações.
        """
        user_events = self._interactions[self._interactions["visitorid"] == user_id]
        if user_events.empty:
            return []
        scores = self._score_items(user_events)
        top_indices = np.argsort(scores)[::-1][:k]
        return [self._item_ids[i] for i in top_indices]
        