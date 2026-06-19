"""Recomendador baseado em decomposição SVD truncada (sklearn)."""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from src.models.baseline.base import Recommender
from src.utils.sparse_utils import build_user_item_matrix


class SVDRecommender(Recommender):
    """Recomenda itens via TruncatedSVD por produto interno no espaço latente.

    Fatora a matriz usuário×item em fatores latentes usando SVD truncado.
    O score de um item para um usuário é o produto interno entre seus
    vetores latentes normalizados em L2.
    """

    def __init__(self, n_components: int = 50, seed: int = 42) -> None:
        """Inicializa o recomendador SVD.

        Args:
            n_components: Dimensão do espaço latente.
            seed: Semente para reprodutibilidade do SVD.
        """
        self._svd = TruncatedSVD(n_components=n_components, random_state=seed)
        self._user_factors: np.ndarray = np.array([])
        self._item_factors: np.ndarray = np.array([])
        self._user_index: dict[int, int] = {}
        self._item_index: dict[int, int] = {}
        self._item_ids: list[int] = []
        self._interactions: pd.DataFrame = pd.DataFrame()

    def fit(self, interactions: pd.DataFrame) -> "SVDRecommender":
        """Treina o SVD a partir da matriz de interações.

        Args:
            interactions: DataFrame com colunas ``visitorid``, ``itemid``
                e ``weight``.
        """
        self._interactions = interactions.copy()
        matrix, _, self._item_ids, self._user_index, self._item_index = (
            build_user_item_matrix(interactions)
        )
        self._user_factors = normalize(self._svd.fit_transform(matrix), norm="l2")
        self._item_factors = normalize(self._svd.components_.T, norm="l2")
        return self

    def _mask_seen(self, scores: np.ndarray, user_id: int) -> np.ndarray:
        """Zera os scores dos itens já vistos pelo usuário.

        Args:
            scores: Array de scores para todos os itens.
            user_id: Identificador do usuário.

        Returns:
            Array de scores com itens vistos marcados como ``-inf``.
        """
        seen = self._interactions[self._interactions["visitorid"] == user_id]["itemid"]
    
        seen_indices = [
            self._item_index[item_id]
            for item_id in seen
            if item_id in self._item_index
        ]
        if seen_indices:
            scores[seen_indices] = -np.inf
        return scores

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Retorna os k itens com maior score latente para o usuário.

        Args:
            user_id: Identificador do usuário.
            k: Número máximo de recomendações.

        Returns:
            Lista de item_ids ordenados por score descendente.
            Retorna lista vazia se o usuário não foi visto no treino.
        """
        if user_id not in self._user_index:
            return []
        scores = self._item_factors @ self._user_factors[self._user_index[user_id]]
        scores = self._mask_seen(scores, user_id)
        return [self._item_ids[i] for i in np.argsort(scores)[::-1][:k]]