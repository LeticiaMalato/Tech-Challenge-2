#divisão temporal de lake de interações em treino e teste. A função `split_by_time` recebe um DataFrame de interações com uma coluna de timestamp e um parâmetro de proporção para teste, e retorna dois DataFrames: um para treino contendo as interações anteriores a um ponto de corte temporal, e outro para teste contendo as interações posteriores. O ponto de corte é calculado com base no intervalo total de tempo das interações, garantindo que o teste contenha os eventos mais recentes. Além disso, o conjunto de teste é filtrado para incluir apenas usuários e itens que também estão presentes no conjunto de treino, evitando problemas de cold start durante a avaliação do modelo.
import pandas as pd


def split_by_time(
    interactions: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
#divide o DataFrame de interações em conjuntos de treino e teste com base em um corte temporal. O parâmetro `test_ratio` determina a proporção do conjunto de teste em relação ao total de interações, e o corte é calculado para garantir que o teste contenha os eventos mais recentes. O conjunto de teste é filtrado para incluir apenas usuários e itens que também estão presentes no conjunto de treino, garantindo que o modelo possa ser avaliado sem enfrentar problemas de cold start.
    t_min = interactions["timestamp"].min()
    t_max = interactions["timestamp"].max()
    cutoff = t_min + (t_max - t_min) * (1 - test_ratio)

    train = interactions[interactions["timestamp"] < cutoff].copy()
    test = interactions[interactions["timestamp"] >= cutoff].copy()

    # Garante que o teste só contenha usuários e itens vistos no treino.
    # Modelos colaborativos não conseguem recomendar entidades desconhecidas.
    known_users = set(train["visitorid"])
    known_items = set(train["itemid"])
    test = test[
        test["visitorid"].isin(known_users) &
        test["itemid"].isin(known_items)
    ]

    return train, test