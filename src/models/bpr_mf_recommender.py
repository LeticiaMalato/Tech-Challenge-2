# Modelo de recomendação baseado em fatoração de matriz otimizada com Bayesian Personalized Ranking (BPR) para feedback implícito.
import pandas as pd
from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix

from src.models.base import Recommender


def _build_user_item_matrix(
    interactions: pd.DataFrame,
) -> tuple[csr_matrix, list[int], list[int], dict[int, int], dict[int, int]]:
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


class BPRMFRecommender(Recommender):
# Recomenda itens usando fatoração de matriz otimizada com Bayesian Personalized Ranking (BPR) para feedback implícito. O modelo aprende vetores de embedding para usuários e itens, onde a semântica de "preferência" é capturada pela diferença entre interações observadas e não observadas. O método recommend calcula as pontuações preditas para um usuário específico e retorna os top-k itens recomendados, filtrando aqueles que o usuário já interagiu.

    def __init__(
        self,
        n_factors: int = 64,
        iterations: int = 100,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        seed: int = 42,
    ) -> None:
        self._model = BayesianPersonalizedRanking(
            factors=n_factors,
            iterations=iterations,
            learning_rate=learning_rate,
            regularization=regularization,
            random_state=seed,
        )
        self._user_ids: list[int] = []
        self._item_ids: list[int] = []
        self._user_index: dict[int, int] = {}
        self._user_item_matrix: csr_matrix = csr_matrix((0, 0))

    def fit(self, interactions: pd.DataFrame) -> None:
  #Treina os embeddings de usuário e item via BPR."""
        (
            self._user_item_matrix,
            self._user_ids,
            self._item_ids,
            self._user_index,
            _,
        ) = _build_user_item_matrix(interactions)

        self._model.fit(self._user_item_matrix.tocsr())

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