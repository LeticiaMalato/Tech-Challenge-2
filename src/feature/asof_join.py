"""Junção temporal as-of entre interações e propriedades de itens."""

import pandas as pd


def _pivot_properties_to_wide(properties: pd.DataFrame) -> pd.DataFrame:
    """Converte propriedades de formato long para wide, ordenadas por tempo.

    Args:
        properties: DataFrame em formato long com colunas ``itemid``,
            ``timestamp``, ``property`` e ``value``.

    Returns:
        DataFrame wide com uma coluna por propriedade, ordenado por timestamp.
    """
    props_sorted = properties.sort_values(["itemid", "timestamp"])
    props_wide = props_sorted.pivot_table(
        index=["itemid", "timestamp"],
        columns="property",
        values="value",
        aggfunc="last",
    ).reset_index()
    props_wide.columns.name = None
    return props_wide.sort_values("timestamp").reset_index(drop=True)


def asof_join_properties(
    interactions: pd.DataFrame,
    properties: pd.DataFrame,
) -> pd.DataFrame:
    """Anexa propriedades de itens às interações via merge temporal as-of.

    Para cada interação, associa as propriedades do item mais recentes
    disponíveis até o timestamp do evento, evitando leakage de informações
    futuras ao usar features de item na modelagem.

    Args:
        interactions: DataFrame com colunas ``visitorid``, ``itemid``
            e ``timestamp``.
        properties: DataFrame em formato long com colunas ``itemid``,
            ``timestamp``, ``property`` e ``value``.

    Returns:
        DataFrame com as colunas de ``interactions`` acrescidas de uma
        coluna por propriedade. Linhas sem propriedade disponível terão
        ``NaN`` nas colunas adicionadas.
    """
    props_wide = _pivot_properties_to_wide(properties)
    interactions_sorted = interactions.sort_values("timestamp").reset_index(drop=True)
    return pd.merge_asof(
        interactions_sorted,
        props_wide,
        on="timestamp",
        by="itemid",
        direction="backward",
    )
