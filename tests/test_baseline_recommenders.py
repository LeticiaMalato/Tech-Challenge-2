"""Testes unitários para os recomendadores baseline."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.models.baseline.item_knn_recommender import ItemKNNRecommender
from src.models.baseline.logistic_recommender import LogisticRecommender
from src.models.baseline.popularity_recommender import PopularityRecommender
from src.models.baseline.svd_recommender import SVDRecommender


def _make_synthetic_interactions(
    n_users: int = 10, n_items: int = 10, n_draws: int = 40, seed: int = 0
) -> pd.DataFrame:
    """Gera interações sintéticas com repetição de usuários e itens.

    Args:
        n_users: Número de usuários distintos possíveis.
        n_items: Número de itens distintos possíveis.
        n_draws: Número de sorteios (antes de remover duplicatas).
        seed: Semente para reprodutibilidade.

    Returns:
        DataFrame com colunas visitorid, itemid e weight, sem pares
        (usuário, item) duplicados.
    """
    rng = np.random.default_rng(seed)
    users = rng.integers(0, n_users, size=n_draws)
    items = rng.integers(100, 100 + n_items, size=n_draws)
    weights = rng.integers(1, 4, size=n_draws)
    df = pd.DataFrame({"visitorid": users, "itemid": items, "weight": weights})
    return df.drop_duplicates(subset=["visitorid", "itemid"]).reset_index(drop=True)


# PopularityRecommender


def test_popularity_recommend_respeita_k() -> None:
    """Não deve retornar mais que k itens."""
    df = _make_synthetic_interactions()
    rec = PopularityRecommender().fit(df)
    assert len(rec.recommend(user_id=0, k=3)) <= 3


def test_popularity_recommend_exclui_itens_vistos() -> None:
    """Itens já interagidos pelo usuário não devem aparecer na recomendação."""
    df = _make_synthetic_interactions()
    rec = PopularityRecommender().fit(df)
    user_id = df["visitorid"].iloc[0]
    seen = set(df[df["visitorid"] == user_id]["itemid"])

    recs = rec.recommend(user_id=user_id, k=len(df["itemid"].unique()))
    assert seen.isdisjoint(recs)


def test_popularity_ranking_reflete_soma_de_pesos() -> None:
    """O item com maior peso acumulado deve vir primeiro no ranking."""
    df = pd.DataFrame(
        {
            "visitorid": [1, 2, 3],
            "itemid": [10, 10, 20],
            "weight": [1, 1, 3],
        }
    )
    rec = PopularityRecommender().fit(df)
    ranking = rec.recommend(user_id=99, k=2)

    assert ranking[0] == 20  # peso acumulado 3, contra 2 do item 10


# ItemKNNRecommender


def test_item_knn_truncate_top_n_mantem_apenas_maiores_valores() -> None:
    """_truncate_top_n deve manter só as top_n_neighbors maiores similaridades.

    Regressão direta do bug em que top_n_neighbors era guardado mas nunca
    aplicado, fazendo a agregação considerar o catálogo inteiro.
    """
    rec = ItemKNNRecommender(top_n_neighbors=2)
    sim = csr_matrix(np.array([[0.1, 0.9, 0.5, 0.2]]))

    truncated = rec._truncate_top_n(sim)
    dense = truncated.toarray().ravel()

    assert np.count_nonzero(dense) == 2
    assert set(dense[dense != 0]) == {0.9, 0.5}


def test_item_knn_recommend_respeita_k() -> None:
    """Não deve retornar mais que k itens."""
    df = _make_synthetic_interactions()
    rec = ItemKNNRecommender(top_n_neighbors=3, max_users=100).fit(df)
    user_id = df["visitorid"].iloc[0]
    assert len(rec.recommend(user_id=user_id, k=2)) <= 2


def test_item_knn_recommend_usuario_sem_interacoes_retorna_vazio() -> None:
    """Usuário sem nenhuma interação deve retornar lista vazia."""
    df = _make_synthetic_interactions()
    rec = ItemKNNRecommender().fit(df)
    assert rec.recommend(user_id=-1, k=5) == []


# SVDRecommender


def test_svd_recommend_respeita_k() -> None:
    """Não deve retornar mais que k itens."""
    df = _make_synthetic_interactions()
    rec = SVDRecommender(n_components=2, seed=42).fit(df)
    user_id = df["visitorid"].iloc[0]
    assert len(rec.recommend(user_id=user_id, k=3)) <= 3


def test_svd_recommend_usuario_desconhecido_retorna_vazio() -> None:
    """Usuário que não existia no treino deve retornar lista vazia."""
    df = _make_synthetic_interactions()
    rec = SVDRecommender(n_components=2, seed=42).fit(df)
    assert rec.recommend(user_id=-1, k=5) == []


def test_logistic_recommend_exclui_itens_vistos() -> None:
    """Itens já interagidos pelo usuário não devem aparecer na recomendação."""
    df = _make_synthetic_interactions()
    rec = LogisticRecommender(neg_ratio=1, seed=42, max_positives=1000).fit(df)
    user_id = df["visitorid"].iloc[0]
    seen = set(df[df["visitorid"] == user_id]["itemid"])

    recs = rec.recommend(user_id=user_id, k=3)
    assert seen.isdisjoint(recs)


# LogisticRecommender


def test_logistic_recommend_respeita_k() -> None:
    """Não deve retornar mais que k itens."""
    df = _make_synthetic_interactions()
    rec = LogisticRecommender(neg_ratio=1, seed=42, max_positives=1000).fit(df)
    user_id = df["visitorid"].iloc[0]
    assert len(rec.recommend(user_id=user_id, k=3)) <= 3


def test_logistic_recommend_usuario_desconhecido_nao_levanta_erro() -> None:
    """Usuário fora do vocabulário não deve quebrar o recommend.

    Nota: ao contrário de SVDRecommender (que retorna lista vazia),
    LogisticRecommender retorna k itens com score -inf para usuário
    desconhecido — comportamento inconsistente entre os baselines,
    documentado aqui para não regredir silenciosamente.
    """
    df = _make_synthetic_interactions()
    rec = LogisticRecommender(neg_ratio=1, seed=42, max_positives=1000).fit(df)
    recs = rec.recommend(user_id=-1, k=3)
    assert len(recs) == 3
