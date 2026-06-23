"""Filtragem k-core para densificação de datasets de interações."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# def apply_kcore(interactions: pd.DataFrame, k: int = 5) -> pd.DataFrame:
#     """Remove iterativamente usuários e itens com menos de k interações.

#     A filtragem é aplicada em loop até que o dataset estabilize, ou seja,
#     nenhuma remoção adicional ocorra. Preserva os IDs originais sem reset.

#     Args:
#         interactions: DataFrame com colunas ``visitorid`` e ``itemid``.
#         k: Número mínimo de interações exigido por usuário e por item.

#     Returns:
#         DataFrame filtrado satisfazendo o critério de k-core.
#     """
#     df = interactions.copy()
#     iteration = 0

#     while True:
#         n_before = len(df)
#         iteration += 1

#         user_counts = df["visitorid"].value_counts()
#         valid_users = user_counts[user_counts >= k].index
#         df = df[df["visitorid"].isin(valid_users)]

#         item_counts = df["itemid"].value_counts()
#         valid_items = item_counts[item_counts >= k].index
#         df = df[df["itemid"].isin(valid_items)]

#         n_removed = n_before - len(df)
#         logger.debug("k-core iteração %d: %d interações removidas.", iteration, n_removed)

#         if len(df) == n_before:
#             break


#     logger.info(
#         "k-core finalizado em %d iterações. %d → %d interações (removidas: %d).",
#         iteration,
#         len(interactions),
#         len(df),
#         len(interactions) - len(df),
#     )
#     return df
def apply_kcore(interactions: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Remove iterativamente usuários e itens com menos de k interações.

    A filtragem é aplicada em loop até que o dataset estabilize, ou seja,
    nenhuma remoção adicional ocorra. Preserva os IDs originais sem reset.

    Args:
        interactions: DataFrame com colunas ``visitorid`` e ``itemid``.
        k: Número mínimo de interações exigido por usuário e por item.

    Returns:
        DataFrame filtrado satisfazendo o critério de k-core.

    Raises:
        ValueError: Se ``k`` for menor ou igual a zero.
    """
    if k <= 0:
        raise ValueError(f"k deve ser positivo, recebido: {k}")

    df = interactions.copy()
    iteration = 0

    while True:
        n_before = len(df)
        iteration += 1

        user_counts = df["visitorid"].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df["visitorid"].isin(valid_users)]
        logger.debug(
            "k-core iteração %d (usuários): %d → %d.",
            iteration,
            n_before,
            len(df),
        )

        n_after_users = len(df)
        item_counts = df["itemid"].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df["itemid"].isin(valid_items)]
        logger.debug(
            "k-core iteração %d (itens): %d → %d.",
            iteration,
            n_after_users,
            len(df),
        )

        if len(df) == n_before:
            break

    logger.info(
        "k-core finalizado em %d iterações. %d → %d interações (removidas: %d).",
        iteration,
        len(interactions),
        len(df),
        len(interactions) - len(df),
    )
    return df
