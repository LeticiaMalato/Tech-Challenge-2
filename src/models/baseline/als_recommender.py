# Modelo de recomendação baseado em fatoração de matriz via Alternating Least Squares (ALS) para feedback implícito.
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from src.models.baseline.base import Recommender


def _build_user_item_matrix(
    interactions: pd.DataFrame,
) -> tuple[csr_matrix, list[int], list[int], dict[int, int]]:
#Constrói a matriz esparsa usuário × item ponderada pelo weight. Retorna a matriz, listas de IDs originais e dicionários de mapeamento ID → índice para usuários e itens.
    user_ids = interactions["visitorid"].unique().tolist()
    item_ids = interactions["itemid"].unique().tolist()

    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {it: i for i, it in enumerate(item_ids)}

    rows = interactions["visitorid"].map(user_index)
    cols = interactions["itemid"].map(item_index)
    data = interactions["weight"].astype(float)

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
    )
    return matrix, user_ids, item_ids, user_index


class ALSRecommender(Recommender):
 # Recomenda itens usando fatoração de matriz via Alternating Least Squares (ALS) otimizada para feedback implícito. O modelo aprende vetores de embedding para usuários e itens, onde a semântica de "preferência" é capturada pela diferença entre interações observadas e não observadas. A confiança nas interações é escalada por um fator alpha, o que ajuda o modelo a lidar melhor com a natureza ruidosa do feedback implícito. O método recommend calcula as pontuações preditas para um usuário específico e retorna os top-k itens recomendados, filtrando aqueles que o usuário já interagiu.

    def __init__(
        self,
        n_factors: int = 64,
        iterations: int = 20,
        regularization: float = 0.01,
        alpha: float = 40.0,
        seed: int = 42,
    ) -> None:
        self._model = AlternatingLeastSquares(
            factors=n_factors,
            iterations=iterations,
            regularization=regularization,
            random_state=seed,
        )
        self._alpha = alpha
        self._user_ids: list[int] = []
        self._item_ids: list[int] = []
        self._user_index: dict[int, int] = {}
        self._user_item_matrix: csr_matrix = csr_matrix((0, 0))

    def fit(self, interactions: pd.DataFrame) -> None:
#Treina os embeddings aplicando escala de confiança sobre os weights."""
        (
            self._user_item_matrix,
            self._user_ids,
            self._item_ids,
            self._user_index,
        ) = _build_user_item_matrix(interactions)

        # aplica escala de confiança: confidence = 1 + alpha * weight
        confidence_matrix = self._user_item_matrix.copy()
        confidence_matrix.data = 1.0 + self._alpha * confidence_matrix.data

        # passa matriz usuário × item diretamente — sem transpor
        self._model.fit(confidence_matrix.tocsr())

    def recommend(self, user_id: int, k: int) -> list[int]:
#Calcula pontuações preditas para um usuário específico e retorna os top-k itens recomendados, filtrando aqueles que o usuário já interagiu.
        if user_id not in self._user_index:
            return []

        user_idx = self._user_index[user_id]
        user_row = self._user_item_matrix[user_idx].tocsr()

        indices, _ = self._model.recommend(
            userid=user_idx,
            user_items=user_row,
            N=k,
            filter_already_liked_items=True,
        )
        return [self._item_ids[i] for i in indices]