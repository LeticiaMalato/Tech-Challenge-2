"""Métricas de avaliação para sistemas de recomendação."""

import numpy as np
import pandas as pd

from src.models.baseline.base import Recommender


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Verifica se ao menos um item relevante aparece no top-k recomendado.

    Args:
        recommended: Lista de item_ids recomendados em ordem de score.
        relevant: Conjunto de item_ids relevantes para o usuário.
        k: Posição de corte da lista.

    Returns:
        1.0 se houver ao menos um acerto no top-k, 0.0 caso contrário.
    """
    return float(any(item in relevant for item in recommended[:k]))


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calcula a precisão no top-k: fração de recomendações relevantes.

    Args:
        recommended: Lista de item_ids recomendados em ordem de score.
        relevant: Conjunto de item_ids relevantes para o usuário.
        k: Posição de corte da lista.

    Returns:
        Proporção de itens relevantes entre os k recomendados.
        Retorna 0.0 se a lista for vazia ou k == 0.
    """
    if not recommended or k == 0:
        return 0.0
    return sum(item in relevant for item in recommended[:k]) / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calcula o recall no top-k: fração do relevante que foi recuperado.

    Args:
        recommended: Lista de item_ids recomendados em ordem de score.
        relevant: Conjunto de item_ids relevantes para o usuário.
        k: Posição de corte da lista.

    Returns:
        Proporção dos itens relevantes recuperados no top-k.
        Retorna 0.0 se não houver itens relevantes ou k == 0.
    """
    if not relevant or k == 0:
        return 0.0
    return sum(item in relevant for item in recommended[:k]) / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calcula o NDCG (Normalized Discounted Cumulative Gain) no top-k.

    Penaliza acertos em posições inferiores da lista via desconto
    logarítmico, capturando a qualidade da ordenação das recomendações.

    Args:
        recommended: Lista de item_ids recomendados em ordem de score.
        relevant: Conjunto de item_ids relevantes para o usuário.
        k: Posição de corte da lista.

    Returns:
        NDCG no top-k, valor em [0.0, 1.0].
        Retorna 0.0 se listas forem vazias ou k == 0.
    """
    if not recommended or not relevant or k == 0:
        return 0.0
    dcg = sum(
        (item in relevant) / np.log2(pos + 2)
        for pos, item in enumerate(recommended[:k])
    )
    idcg = sum(1.0 / np.log2(pos + 2) for pos in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calcula o MRR (Mean Reciprocal Rank) no top-k.

    Mede a posição do primeiro item relevante na lista recomendada.
    Especialmente adequado para e-commerce, onde o usuário escaneia
    a lista de cima para baixo e para no primeiro produto de interesse.

    Args:
        recommended: Lista de item_ids recomendados em ordem de score.
        relevant: Conjunto de item_ids relevantes para o usuário.
        k: Posição de corte da lista.

    Returns:
        Recíproco da posição do primeiro acerto no top-k (1/pos).
        Retorna 0.0 se não houver acerto ou k == 0.
    """
    if not recommended or not relevant or k == 0:
        return 0.0
    for pos, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / pos
    return 0.0


def evaluate_recommender(
    recommender: Recommender,
    test: pd.DataFrame,
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """Avalia um recomendador em múltiplos valores de k.

    Para cada valor de k, calcula a média de Hit Rate, Precision, Recall,
    NDCG e MRR sobre todos os usuários do conjunto de teste. Usuários que
    recebem lista vazia contribuem com 0 em todas as métricas.

    Args:
        recommender: Instância treinada que implementa ``Recommender``.
        test: DataFrame com colunas ``visitorid`` e ``itemid``.
        n_items: Mantido por compatibilidade com a interface existente.
        k_values: Lista de cortes k a avaliar. Padrão: [5, 10, 20].

    Returns:
        DataFrame com uma linha por valor de k e colunas:
        ``k``, ``hit_rate``, ``precision``, ``recall``, ``ndcg``, ``mrr``.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    relevant_per_user: dict[int, set[int]] = (
        test.groupby("visitorid")["itemid"].apply(set).to_dict()
    )

    # gera recomendações uma única vez para o maior k
    scores_cache: dict[int, list[int]] = {
        user_id: recommender.recommend(user_id, k=max(k_values))
        for user_id in relevant_per_user
    }

    results = []
    for k in k_values:
        hit_rates, precisions, recalls, ndcgs, mrrs = [], [], [], [], []

        for user_id, relevant in relevant_per_user.items():
            recs = scores_cache[user_id]
            hit_rates.append(hit_rate_at_k(recs, relevant, k))
            precisions.append(precision_at_k(recs, relevant, k))
            recalls.append(recall_at_k(recs, relevant, k))
            ndcgs.append(ndcg_at_k(recs, relevant, k))
            mrrs.append(mrr_at_k(recs, relevant, k))

        results.append(
            {
                "k": k,
                "hit_rate": np.mean(hit_rates),
                "precision": np.mean(precisions),
                "recall": np.mean(recalls),
                "ndcg": np.mean(ndcgs),
                "mrr": np.mean(mrrs),
            }
        )

    return pd.DataFrame(results)


def compare_models_metrics(resultados: list[dict]) -> pd.DataFrame:
    """Consolida e imprime uma tabela comparativa entre múltiplos modelos.

    Args:
        resultados: Lista de dicionários, cada um com as chaves ``model``,
            ``k``, ``hit_rate``, ``precision``, ``recall``, ``ndcg`` e ``mrr``.

    Returns:
        DataFrame consolidado, ordenado por ``ndcg`` descendente.
    """
    cols = ["model", "k", "hit_rate", "precision", "recall", "ndcg", "mrr"]
    df = pd.DataFrame(resultados)[cols]

    print("\n" + "=" * 70)
    print("COMPARAÇÃO DE MODELOS — MÉTRICAS DE RECOMENDAÇÃO")
    print("=" * 70)
    print(
        df.sort_values("ndcg", ascending=False).to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )
    print("=" * 70)

    melhores = [
        ("NDCG", df.loc[df["ndcg"].idxmax()]),
        ("MRR", df.loc[df["mrr"].idxmax()]),
        ("Recall", df.loc[df["recall"].idxmax()]),
        ("Precision", df.loc[df["precision"].idxmax()]),
    ]
    for titulo, melhor in melhores:
        print(
            f"\n  Melhor modelo por {titulo}: {melhor['model']} (k={int(melhor['k'])})"
        )
        print(f"    hit_rate:  {melhor['hit_rate']:.4f}")
        print(f"    precision: {melhor['precision']:.4f}")
        print(f"    recall:    {melhor['recall']:.4f}")
        print(f"    ndcg:      {melhor['ndcg']:.4f}")
        print(f"    mrr:       {melhor['mrr']:.4f}")

    return df
