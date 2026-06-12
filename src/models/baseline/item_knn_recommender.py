import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.models.baseline.base import Recommender


def _build_item_user_matrix(
    interactions: pd.DataFrame,
) -> tuple[csr_matrix, list[int], dict[int, int]]:
#Constrói a matriz esparsa item × usuário ponderada pelo weight. Retorna a matriz, lista de IDs originais e dicionário de mapeamento ID → índice para itens.
    item_ids = interactions["itemid"].unique().tolist()
    user_ids = interactions["visitorid"].unique().tolist()

    item_index = {item: i for i, item in enumerate(item_ids)}
    user_index = {user: i for i, user in enumerate(user_ids)}

    rows = interactions["itemid"].map(item_index)
    cols = interactions["visitorid"].map(user_index)
    data = interactions["weight"].astype(float)

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(item_ids), len(user_ids)),
    )
    return matrix, item_ids, item_index


class ItemKNNRecommender(Recommender):
# Recomenda itens usando similaridade de cosseno entre vetores de interação item × usuário. O modelo calcula a similaridade entre itens com base nas interações dos usuários, e recomenda itens similares àqueles que o usuário já interagiu. A recomendação é feita sob demanda, calculando a similaridade apenas para os itens do histórico do usuário, o que torna o processo eficiente mesmo sem pré-computar uma matriz completa de similaridade item × item. O método recommend retorna os top-k itens recomendados, filtrando aqueles que o usuário já viu.

    def __init__(self, top_n_neighbors: int = 20) -> None:
        self._top_n_neighbors = top_n_neighbors
        self._item_ids: list[int] = []
        self._item_index: dict[int, int] = {}
        self._matrix: csr_matrix = csr_matrix((0, 0))
        self._interactions: pd.DataFrame = pd.DataFrame()

    def fit(self, interactions: pd.DataFrame) -> None:
        """Guarda a matriz esparsa — similaridade calculada sob demanda."""
        self._interactions = interactions.copy()
        self._matrix, self._item_ids, self._item_index = (
            _build_item_user_matrix(interactions)
        )

    def recommend(self, user_id: int, k: int) -> list[int]:
# Calcula pontuações preditas para um usuário específico e retorna os top-k itens recomendados, filtrando aqueles que o usuário já interagiu. O método recupera as interações do usuário, calcula a similaridade de cosseno entre os
        user_events = self._interactions[
            self._interactions["visitorid"] == user_id
        ]

        if user_events.empty:
            return []

        already_seen = set()
        scores = np.zeros(len(self._item_ids))

        for _, row in user_events.iterrows():
            item_id = row["itemid"]
            if item_id not in self._item_index:
                continue

            hist_index = self._item_index[item_id]
            already_seen.add(hist_index)

            # calcula similaridade só desse item contra todos — O(n_items)
            # em vez de montar a matriz n_items × n_items inteira
            hist_vector = self._matrix[hist_index]
            sims = cosine_similarity(hist_vector, self._matrix).flatten()
            scores += sims * float(row["weight"])

        for seen_index in already_seen:
            scores[seen_index] = 0.0

        top_indices = np.argsort(scores)[::-1][:k]
        return [self._item_ids[i] for i in top_indices]