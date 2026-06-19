"""Pré-processamento do dataset de propriedades de itens."""
import pandas as pd

from src.preprocessing.base import Preprocessor
from src.utils.utils import to_datetime_utc 

LEGIBLE_PROPERTIES: tuple[str, ...] = ("categoryid", "available")


def filter_legible(
    df: pd.DataFrame,
    properties: tuple[str, ...] = LEGIBLE_PROPERTIES,
) -> pd.DataFrame:
    """Filtra linhas com propriedades semanticamente úteis para o modelo.

    Args:
        df: DataFrame com coluna 'property'.
        properties: Tupla de valores de propriedade a manter.

    Returns:
        DataFrame filtrado pelas propriedades especificadas.
    """
    return df[df["property"].isin(properties)]


class ItemPropertiesPreprocessor(Preprocessor):
    """Pré-processador para o dataset de propriedades de itens.

    Converte timestamp e filtra propriedades legíveis,
    mantendo o formato long do dataset.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa o dataset de propriedades de itens.

        Args:
            df: DataFrame bruto com colunas [timestamp, itemid, property, value].

        Returns:
            DataFrame filtrado e com timestamp convertido para UTC.
        """
        df = df.copy()
        df = to_datetime_utc(df, "timestamp")
        return filter_legible(df)