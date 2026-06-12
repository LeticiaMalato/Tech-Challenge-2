#Metricas de avaliação para sistemas de recomendação. Este módulo define funções para calcular métricas comuns como Precision@K, Recall@K, NDCG@K e MAP@K, que avaliam a qualidade das recomendações geradas por um modelo. A função principal `evaluate_recommender` recebe um modelo de recomendação treinado e um conjunto de teste, e retorna um DataFrame com as médias das métricas para diferentes valores de K, permitindo uma análise abrangente do desempenho do sistema de recomendação.
import numpy as np
import pandas as pd


def _hits_at_k(recommended: list[int], relevant: set[int], k: int) -> list[bool]:
#retorna uma lista de booleanos indicando se cada um dos k itens recomendados está presente no conjunto de itens relevantes. A função compara os itens recomendados com os itens relevantes e marca como True aqueles que são acertos (hits) e False para os erros (misses). Essa lista é usada posteriormente para calcular as métricas de avaliação, como precisão e recall.
    return [item in relevant for item in recommended[:k]]


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
#Fração dos itens recomendados que são relevantes. Mede exatidão — penaliza recomendações irrelevantes.
    if not recommended or k == 0:
        return 0.0
    hits = sum(_hits_at_k(recommended, relevant, k))
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
#Fração dos itens relevantes que foram recomendados. Mede completude — penaliza recomendações que deixam de fora itens relevantes.
    if not relevant or k == 0:
        return 0.0
    hits = sum(_hits_at_k(recommended, relevant, k))
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
#Mede a qualidade das recomendações considerando a posição dos acertos. A métrica NDCG (Normalized Discounted Cumulative Gain) penaliza acertos que aparecem mais abaixo na lista de recomendações, atribuindo uma pontuação maior para itens relevantes que são recomendados nas primeiras posições. O resultado é normalizado para que o valor máximo seja 1.0, indicando uma lista de recomendações ideal.
    if not recommended or not relevant or k == 0:
        return 0.0

    hits = _hits_at_k(recommended, relevant, k)
    dcg = sum(
        hit / np.log2(pos + 2)
        for pos, hit in enumerate(hits)
    )

    n_ideal = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(pos + 2) for pos in range(n_ideal))

    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Mean Average Precision nas k primeiras posições.

    Calcula a precisão em cada posição onde há um acerto e tira a média.
    Sensível à ordem — penaliza acertos que aparecem no final da lista.
    """
    if not recommended or not relevant or k == 0:
        return 0.0

    hits = 0
    cumulative_precision = 0.0

    for pos, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            cumulative_precision += hits / (pos + 1)

    return cumulative_precision / min(len(relevant), k)


def evaluate_recommender(
    recommender,
    test: pd.DataFrame,
    k_values: list[int] = [5, 10, 20],
) -> pd.DataFrame:
#Avalia um modelo de recomendação usando as métricas Precision@K, Recall@K, NDCG@K e MAP@K para diferentes valores de K. A função recebe um modelo de recomendação treinado, um conjunto de teste contendo as interações reais dos usuários e uma lista de valores de K para os quais as métricas serão calculadas. O resultado é um DataFrame que resume o desempenho do modelo em cada métrica para os diferentes valores de K, permitindo uma análise comparativa do modelo em termos de precisão, recall, qualidade das recomendações e média da precisão.
    # Agrupa itens relevantes por usuário uma única vez
    relevant_per_user: dict[int, set[int]] = (
        test.groupby("visitorid")["itemid"]
        .apply(set)
        .to_dict()
    )

    results = []

    for k in k_values:
        precisions, recalls, ndcgs, maps = [], [], [], []

        for user_id, relevant in relevant_per_user.items():
            recommended = recommender.recommend(user_id, k=k)

            precisions.append(precision_at_k(recommended, relevant, k))
            recalls.append(recall_at_k(recommended, relevant, k))
            ndcgs.append(ndcg_at_k(recommended, relevant, k))
            maps.append(map_at_k(recommended, relevant, k))

        results.append({
            "k":         k,
            "precision": np.mean(precisions),
            "recall":    np.mean(recalls),
            "ndcg":      np.mean(ndcgs),
            "map":       np.mean(maps),
        })

    return pd.DataFrame(results)