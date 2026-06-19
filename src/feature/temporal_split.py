"""Divisão temporal de interações em conjuntos de treino e teste."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# def split_by_time(
#     interactions: pd.DataFrame,
#     test_ratio: float = 0.2,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """Divide interações em treino e teste com base em um corte temporal.

#     O corte é calculado como o percentil ``(1 - test_ratio)`` do intervalo
#     total de timestamps. O teste é filtrado para conter apenas usuários e
#     itens presentes no treino, evitando cold start na avaliação.

#     Args:
#         interactions: DataFrame com colunas ``visitorid``, ``itemid``
#             e ``timestamp``.
#         test_ratio: Proporção do intervalo de tempo destinada ao teste.
#             Deve estar em (0.0, 1.0). Padrão: 0.2.

#     Returns:
#         Tupla (train, test) de DataFrames.

#     Raises:
#         ValueError: Se ``test_ratio`` não estiver no intervalo (0.0, 1.0).
#     """
#     if not 0.0 < test_ratio < 1.0:
#         raise ValueError(
#             f"test_ratio deve estar em (0.0, 1.0), recebido: {test_ratio}"
#         )

#     t_min = interactions["timestamp"].min()
#     t_max = interactions["timestamp"].max()
#     # cutoff = t_min + (t_max - t_min) * (1 - test_ratio)
#     cutoff = interactions["timestamp"].quantile(1 - test_ratio)

#     train = interactions[interactions["timestamp"] < cutoff].copy()
#     test = interactions[interactions["timestamp"] >= cutoff].copy()

#     known_users = set(train["visitorid"])
#     known_items = set(train["itemid"])

#     n_before = len(test)
#     test = test[
#         test["visitorid"].isin(known_users) & test["itemid"].isin(known_items)
#     ]

#     logger.info(
#         "Split temporal: cutoff=%s | treino=%d | teste=%d | cold-start removidos=%d.",
#         cutoff, len(train), len(test), n_before - len(test),
#     )
#     return train, test

def split_by_time(
    interactions: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide interações em treino e teste com base em um corte temporal.

    O corte é calculado pelo quantil ``(1 - test_ratio)`` da distribuição
    real de timestamps — mais robusto que corte linear para dados com
    distribuição temporal não uniforme. O teste é filtrado para conter
    apenas usuários e itens presentes no treino, evitando cold start.

    Args:
        interactions: DataFrame com colunas ``visitorid``, ``itemid``
            e ``timestamp``.
        test_ratio: Proporção de eventos destinada ao teste.
            Deve estar em (0.0, 1.0). Padrão: 0.2.

    Returns:
        Tupla (train, test) de DataFrames.

    Raises:
        ValueError: Se ``test_ratio`` não estiver no intervalo (0.0, 1.0).
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(
            f"test_ratio deve estar em (0.0, 1.0), recebido: {test_ratio}"
        )

    cutoff = interactions["timestamp"].quantile(1 - test_ratio)

    train = interactions[interactions["timestamp"] < cutoff].copy()
    test = interactions[interactions["timestamp"] >= cutoff].copy()

    known_users = set(train["visitorid"])
    known_items = set(train["itemid"])

    n_before = len(test)
    n_test_users_before = test["visitorid"].nunique()

    test = test[
        test["visitorid"].isin(known_users) & test["itemid"].isin(known_items)
    ]

    logger.info(
        "Split temporal: cutoff=%s | treino=%d | teste=%d | cold-start removidos=%d.",
        cutoff, len(train), len(test), n_before - len(test),
    )
    logger.info(
        "Cobertura de usuários no teste: %d/%d (%.1f%%).",
        test["visitorid"].nunique(),
        n_test_users_before,
        100 * test["visitorid"].nunique() / max(n_test_users_before, 1),
    )
    return train, test