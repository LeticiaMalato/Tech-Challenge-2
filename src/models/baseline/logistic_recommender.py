"""Recomendador binário com LogisticRegression sobre features manuais.

Diferente de abordagens baseadas em embeddings latentes (SVD), este
recomendador usa features interpretáveis derivadas diretamente das
interações brutas — popularidade do item, atividade do usuário e
afinidade comportamental — evitando dependência de fatoração de matriz.
Isso o torna uma família de baseline independente do SVD/ItemKNN,
testando a hipótese de que sinais simples e diretos já permitem
personalização razoável.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.baseline.base import Recommender

_MAX_NEGATIVE_ATTEMPTS_MULTIPLIER = 100
_N_FEATURES = 5


def _sample_negatives(
    positive_pairs: set[tuple[int, int]],
    n_users: int,
    n_items: int,
    n_samples: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Amostra pares negativos não presentes nas interações.

    Args:
        positive_pairs: Conjunto de pares positivos a excluir.
        n_users: Total de usuários no vocabulário.
        n_items: Total de itens no vocabulário.
        n_samples: Quantidade de negativos a gerar.
        seed: Semente para reprodutibilidade.

    Returns:
        Lista de pares (user_idx, item_idx) negativos.

    Raises:
        RuntimeError: Se não for possível amostrar negativos suficientes.
    """
    rng = np.random.default_rng(seed)
    max_attempts = n_samples * _MAX_NEGATIVE_ATTEMPTS_MULTIPLIER
    attempts = 0
    negatives: list[tuple[int, int]] = []

    while len(negatives) < n_samples:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Não foi possível amostrar {n_samples} negativos após "
                f"{max_attempts} tentativas."
            )
        u = int(rng.integers(0, n_users))
        i = int(rng.integers(0, n_items))
        if (u, i) not in positive_pairs:
            negatives.append((u, i))
        attempts += 1

    return negatives


def _build_item_stats(
    interactions: pd.DataFrame,
    item_index: dict[int, int],
    n_items: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula popularidade ponderada e contagem de interações por item.

    Args:
        interactions: DataFrame com colunas ``itemid`` e ``weight``.
        item_index: Mapeamento de itemid original para índice local.
        n_items: Total de itens no vocabulário.

    Returns:
        Tupla (popularidade normalizada, contagem normalizada),
        ambas como arrays de shape (n_items,).
    """
    item_idx = interactions["itemid"].map(item_index)
    weighted = interactions.assign(item_idx=item_idx).groupby("item_idx")["weight"].sum()
    counts = interactions.assign(item_idx=item_idx).groupby("item_idx").size()

    popularity = np.zeros(n_items, dtype=np.float64)
    popularity[weighted.index] = weighted.to_numpy()
    popularity = popularity / (popularity.max() + 1e-9)

    interaction_count = np.zeros(n_items, dtype=np.float64)
    interaction_count[counts.index] = counts.to_numpy()
    interaction_count = interaction_count / (interaction_count.max() + 1e-9)

    return popularity, interaction_count


def _build_user_stats(
    interactions: pd.DataFrame,
    user_index: dict[int, int],
    n_users: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula atividade total e peso médio de interação por usuário.

    Args:
        interactions: DataFrame com colunas ``visitorid`` e ``weight``.
        user_index: Mapeamento de visitorid original para índice local.
        n_users: Total de usuários no vocabulário.

    Returns:
        Tupla (atividade normalizada, peso médio normalizado),
        ambas como arrays de shape (n_users,).
    """
    user_idx = interactions["visitorid"].map(user_index)
    counts = interactions.assign(user_idx=user_idx).groupby("user_idx").size()
    avg_weight = interactions.assign(user_idx=user_idx).groupby("user_idx")["weight"].mean()

    activity = np.zeros(n_users, dtype=np.float64)
    activity[counts.index] = counts.to_numpy()
    activity = activity / (activity.max() + 1e-9)

    mean_weight = np.zeros(n_users, dtype=np.float64)
    mean_weight[avg_weight.index] = avg_weight.to_numpy()
    mean_weight = mean_weight / (mean_weight.max() + 1e-9)

    return activity, mean_weight


def _make_feature_matrix(
    pairs: list[tuple[int, int]],
    item_popularity: np.ndarray,
    item_interaction_count: np.ndarray,
    user_activity: np.ndarray,
    user_mean_weight: np.ndarray,
) -> np.ndarray:
    """Constrói a matriz de features manuais para os pares (usuário, item).

    Features (nesta ordem):
        1. Popularidade ponderada do item.
        2. Contagem normalizada de interações do item.
        3. Atividade normalizada do usuário.
        4. Peso médio de interação do usuário.
        5. Afinidade: produto entre atividade do usuário e
           popularidade do item, capturando se um usuário muito
           ativo tende a interagir com itens muito populares.

    Args:
        pairs: Lista de pares (user_idx, item_idx).
        item_popularity: Vetor de popularidade normalizada (n_items,).
        item_interaction_count: Vetor de contagem normalizada (n_items,).
        user_activity: Vetor de atividade normalizada (n_users,).
        user_mean_weight: Vetor de peso médio normalizado (n_users,).

    Returns:
        Matriz de features de shape (len(pairs), 5).
    """
    user_idxs = np.array([p[0] for p in pairs])
    item_idxs = np.array([p[1] for p in pairs])

    pop = item_popularity[item_idxs]
    count = item_interaction_count[item_idxs]
    activity = user_activity[user_idxs]
    mean_weight = user_mean_weight[user_idxs]
    affinity = activity * pop

    return np.column_stack([pop, count, activity, mean_weight, affinity])


class LogisticRecommender(Recommender):
    """Ranker binário com LogisticRegression sobre features manuais.

    Usa sinais diretos de popularidade e atividade — sem embeddings
    latentes — como uma família de baseline independente do SVD/ItemKNN.
    Para viabilizar uso em datasets grandes, o número de pares positivos
    usados no treino é limitado por ``max_positives``, amostrados
    aleatoriamente das interações reais.
    """

    def __init__(
        self,
        neg_ratio: int = 3,
        seed: int = 42,
        max_positives: int = 50_000,
    ) -> None:
        """Inicializa o recomendador logístico.

        Args:
            neg_ratio: Proporção de negativos por positivo no treino.
            seed: Semente para reprodutibilidade.
            max_positives: Número máximo de pares positivos amostrados
                para treinar o classificador. Reduz tempo de fit em
                datasets grandes sem perder qualidade significativa.
        """
        self._clf = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
        self._scaler = StandardScaler()
        self._neg_ratio = neg_ratio
        self._seed = seed
        self._max_positives = max_positives
        self._item_popularity: np.ndarray = np.array([])
        self._item_interaction_count: np.ndarray = np.array([])
        self._user_activity: np.ndarray = np.array([])
        self._user_mean_weight: np.ndarray = np.array([])
        self._user_index: dict[int, int] = {}
        self._item_index: dict[int, int] = {}
        self._item_ids: list[int] = []
        self._interactions: pd.DataFrame = pd.DataFrame()

    def fit(self, interactions: pd.DataFrame) -> "LogisticRecommender":
        """Treina as estatísticas de features e o classificador logístico.

        Args:
            interactions: DataFrame com colunas ``visitorid``, ``itemid``
                e ``weight``.
        """
        self._interactions = interactions.copy()
        self._build_index(interactions)

        self._item_popularity, self._item_interaction_count = _build_item_stats(
            interactions, self._item_index, len(self._item_ids)
        )
        self._user_activity, self._user_mean_weight = _build_user_stats(
            interactions, self._user_index, len(self._user_index)
        )

        x_train, y_train = self._build_training_data(interactions)
        x_train = self._scaler.fit_transform(x_train)
        self._clf.fit(x_train, y_train)
        return self
    
    def _build_index(self, interactions: pd.DataFrame) -> None:
        """Constrói os índices locais de usuário e item.

        Args:
            interactions: DataFrame com colunas ``visitorid`` e ``itemid``.
        """
        unique_users = interactions["visitorid"].unique()
        unique_items = interactions["itemid"].unique()
        self._user_index = {uid: idx for idx, uid in enumerate(unique_users)}
        self._item_index = {iid: idx for idx, iid in enumerate(unique_items)}
        self._item_ids = list(unique_items)

    def _build_training_data(
        self, interactions: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Monta arrays de treino com positivos amostrados e negativos.

        Args:
            interactions: DataFrame de interações para extrair positivos.

        Returns:
            Tupla (X, y) com matriz de features e rótulos binários.
        """
        all_positives = list(zip(
            interactions["visitorid"].map(self._user_index),
            interactions["itemid"].map(self._item_index),
        ))

        rng = np.random.default_rng(self._seed)
        if len(all_positives) > self._max_positives:
            indices = rng.choice(len(all_positives), size=self._max_positives, replace=False)
            positives = [all_positives[i] for i in indices]
        else:
            positives = all_positives

        negatives = _sample_negatives(
            set(all_positives),
            n_users=len(self._user_index),
            n_items=len(self._item_index),
            n_samples=len(positives) * self._neg_ratio,
            seed=self._seed,
        )
        x_pos = self._features_for(positives)
        x_neg = self._features_for(negatives)
        return np.vstack([x_pos, x_neg]), np.array([1] * len(positives) + [0] * len(negatives))

    def _features_for(self, pairs: list[tuple[int, int]]) -> np.ndarray:
        """Atalho para construir features usando o estado atual do modelo.

        Args:
            pairs: Lista de pares (user_idx, item_idx).

        Returns:
            Matriz de features de shape (len(pairs), 5).
        """
        return _make_feature_matrix(
            pairs,
            self._item_popularity,
            self._item_interaction_count,
            self._user_activity,
            self._user_mean_weight,
        )

    def _mask_seen(self, scores: np.ndarray, user_id: int) -> np.ndarray:
        """Descarta itens já vistos atribuindo score ``-inf``.

        Args:
            scores: Array de scores de shape (n_items,).
            user_id: Identificador do usuário.

        Returns:
            Array com itens vistos marcados como ``-inf``.
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

    def score_all_items(self, user_id: int) -> np.ndarray:
        """Retorna scores para todos os itens do catálogo de uma vez.

        Args:
            user_id: Identificador do usuário.

        Returns:
            Array de scores shape (n_items,) com itens vistos em -inf.
        """
        n_items = len(self._item_ids)
        if user_id not in self._user_index:
            return np.full(n_items, -np.inf)

        u_idx = self._user_index[user_id]
        pairs = [(u_idx, item_idx) for item_idx in range(n_items)]
        x_score = self._features_for(pairs)
        x_score = self._scaler.transform(x_score)
        scores = self._clf.predict_proba(x_score)[:, 1]
        return self._mask_seen(scores, user_id)

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Retorna os k itens com maior probabilidade prevista para o usuário.

        Args:
            user_id: Identificador do usuário.
            k: Número máximo de recomendações.

        Returns:
            Lista de item_ids ordenados por score descendente.
        """
        scores = self.score_all_items(user_id)
        return [self._item_ids[i] for i in np.argsort(scores)[::-1][:k]]