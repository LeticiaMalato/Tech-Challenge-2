# Este módulo implementa a filtragem k-core para sistemas de recomendação. A função `apply_kcore` remove iterativamente usuários e itens que têm menos de k interações, garantindo que o dataset resultante seja mais denso e adequado para treinamento de modelos de recomendação. A filtragem é feita em um loop até que nenhuma remoção adicional seja necessária, ou seja, até que o dataset estabilize. O resultado é um DataFrame filtrado que mantém os IDs originais dos usuários e itens, sem resetar o índice.
import pandas as pd


def apply_kcore(interactions: pd.DataFrame, k: int = 5) -> pd.DataFrame:

    df = interactions.copy()

    while True:
        n_before = len(df)

        user_counts = df["visitorid"].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df["visitorid"].isin(valid_users)]

        item_counts = df["itemid"].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df["itemid"].isin(valid_items)]

        if len(df) == n_before:
            break

    return df