"""Recomendador baseado em similaridade de cosseno entre itens."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.models.baseline.base import Recommender
from src.utils.sparse_utils import build_item_user_matrix


class ItemKNNRecommender(Recommender):
    """Recomenda itens por similaridade de cosseno item×usuário.

    Para viabilizar o uso em datasets grandes, a matriz item×usuário é
    construída sobre uma amostra dos usuários mais ativos, controlada por
    ``max_users``. A similaridade é calculada sob demanda por item visto,
    truncada aos ``top_n_neighbors`` vizinhos mais similares antes de
    agregar — evitando que itens pouco relacionados dominem o score
    apenas por soma cumulativa sobre o catálogo inteiro.
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
        self._matrix: csr_matrix | None = None
        self._interactions: pd.DataFrame = pd.DataFrame()

    def fit(self, interactions: pd.DataFrame) -> "ItemKNNRecommender":
        """Armazena a matriz item×usuário amostrada para uso no recommend.

        Args:
            interactions: DataFrame com colunas ``visitorid``, ``itemid``
                e ``weight``.
        """
        self._interactions = interactions.copy()

        # mantém apenas os usuários mais ativos para reduzir memória
        top_users = interactions["visitorid"].value_counts().head(self._max_users).index
        sampled = interactions[interactions["visitorid"].isin(top_users)]

        self._matrix, self._item_ids, self._item_index = build_item_user_matrix(sampled)

        return self

    def _truncate_top_n(self, sim_matrix: csr_matrix) -> csr_matrix:
        """Mantém apenas as top_n_neighbors maiores similaridades por linha.

        Cada linha corresponde a um item visto pelo usuário; o número de
        linhas é tipicamente pequeno (itens que um usuário interagiu),
        então converter linha a linha para denso é seguro em memória —
        ao contrário de converter a matriz inteira, cujas colunas cobrem
        o catálogo inteiro.

        Args:
            sim_matrix: Matriz esparsa de shape (n_seen, n_items).

        Returns:
            Matriz esparsa da mesma shape, com no máximo
            ``top_n_neighbors`` valores não nulos por linha.
        """
        sim_matrix = sim_matrix.tocsr()
        rows, cols, data = [], [], []
        top_n = self._top_n_neighbors

        for row_idx in range(sim_matrix.shape[0]):
            row = sim_matrix.getrow(row_idx).toarray().ravel()
            if top_n < len(row):
                top_indices = np.argpartition(row, -top_n)[-top_n:]
            else:
                top_indices = np.nonzero(row)[0]
            nonzero = top_indices[row[top_indices] != 0]
            rows.extend([row_idx] * len(nonzero))
            cols.extend(nonzero.tolist())
            data.extend(row[nonzero].tolist())

        return csr_matrix((data, (rows, cols)), shape=sim_matrix.shape)

    def _score_items(self, user_events: pd.DataFrame) -> np.ndarray:
        seen_mask = user_events["itemid"].isin(self._item_index)
        valid_events = user_events[seen_mask]

        if valid_events.empty:
            return np.zeros(len(self._item_ids))

        agg = valid_events.groupby("itemid", as_index=False)["weight"].sum()

        seen_indices = [self._item_index[iid] for iid in agg["itemid"]]
        weights = agg["weight"].to_numpy(dtype=np.float32)

        # dense_output=False mantém o resultado em formato esparso,
        # evitando materializar (n_seen x n_items) como array denso.
        sim_matrix = cosine_similarity(
            self._matrix[seen_indices], self._matrix, dense_output=False
        )
        sim_matrix = self._truncate_top_n(sim_matrix)

        scores = np.asarray(sim_matrix.multiply(weights[:, None]).sum(axis=0)).ravel()

        # garante que itens vistos não sejam recomendados
        scores[seen_indices] = -np.inf
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
