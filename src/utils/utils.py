"""Transformações reutilizáveis para os datasets de eventos e produtos."""

import pandas as pd


def to_datetime_utc(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Converte coluna de timestamp Unix (ms) para datetime UTC.

    Args:
        df: DataFrame contendo a coluna de timestamp.
        column: Nome da coluna a ser convertida.

    Returns:
        DataFrame com a coluna convertida para datetime UTC.
    """
    df = df.copy()
    df[column] = pd.to_datetime(df[column], unit="ms", utc=True)
    return df
