"""Utilitários de matrizes esparsas compartilhados entre recomendadores."""

import pandas as pd
from scipy.sparse import csr_matrix


def build_user_item_matrix(
    interactions: pd.DataFrame,
) -> tuple[csr_matrix, list[int], list[int], dict[int, int], dict[int, int]]:
    """Constrói a matriz esparsa usuário×item a partir das interações.

    Args:
        interactions: DataFrame com colunas ``visitorid``, ``itemid`` e ``weight``.

    Returns:
        Tupla com:
        - matriz CSR de shape (n_users, n_items);
        - lista de user_ids na ordem das linhas;
        - lista de item_ids na ordem das colunas;
        - mapeamento user_id → índice de linha;
        - mapeamento item_id → índice de coluna.
    """
    user_ids = interactions["visitorid"].unique().tolist()
    item_ids = interactions["itemid"].unique().tolist()
    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {it: i for i, it in enumerate(item_ids)}
    matrix = csr_matrix(
        (
            interactions["weight"].astype(float),
            (
                interactions["visitorid"].map(user_index),
                interactions["itemid"].map(item_index),
            ),
        ),
        shape=(len(user_ids), len(item_ids)),
    )
    return matrix, user_ids, item_ids, user_index, item_index


def build_item_user_matrix(
    interactions: pd.DataFrame,
) -> tuple[csr_matrix, list[int], dict[int, int]]:
    """Constrói a matriz esparsa item×usuário a partir das interações.

    Versão transposta de ``build_user_item_matrix``, usada pelo ItemKNN
    para calcular similaridade entre itens via seus vetores de usuários.

    Args:
        interactions: DataFrame com colunas ``visitorid``, ``itemid`` e ``weight``.

    Returns:
        Tupla com:
        - matriz CSR de shape (n_items, n_users);
        - lista de item_ids na ordem das linhas;
        - mapeamento item_id → índice de linha.
    """
    item_ids = interactions["itemid"].unique().tolist()
    user_ids = interactions["visitorid"].unique().tolist()
    item_index = {item: i for i, item in enumerate(item_ids)}
    user_index = {user: i for i, user in enumerate(user_ids)}
    matrix = csr_matrix(
        (
            interactions["weight"].astype(float),
            (
                interactions["itemid"].map(item_index),
                interactions["visitorid"].map(user_index),
            ),
        ),
        shape=(len(item_ids), len(user_ids)),
    )
    return matrix, item_ids, item_index
