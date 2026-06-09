#Pre processamento do dataset item_properties.
import pandas as pd

from src.preprocessing.base import Preprocessor
from src.preprocessing.utils import to_datetime_utc

# Propriedades legíveis, que fazem sentido de serem usadas como features.
LEGIBLE_PROPERTIES: tuple[str, ...] = ("categoryid", "available")


def filter_legible(df: pd.DataFrame) -> pd.DataFrame:
    #filtra as linhas onde a coluna "property" tem os valores "categoryid" ou "available"
    return df[df["property"].isin(LEGIBLE_PROPERTIES)]


class ItemPropertiesPreprocessor(Preprocessor):
    #Converte timestamp e filtra propriedades legíveis, mantendo o formato long.

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = to_datetime_utc(df, "timestamp")
        return filter_legible(df)