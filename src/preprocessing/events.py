"""Pré-processamento do dataset de eventos de comportamento do usuário."""
import pandas as pd

from src.preprocessing.base import Preprocessor
from src.utils.utils import to_datetime_utc

EVENT_WEIGHTS: dict[str, int] = {"view": 1, "addtocart": 2, "transaction": 3}

_DUPLICATE_SUBSET: list[str] = ["visitorid", "itemid", "event", "timestamp"]


def add_event_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Mapeia tipos de evento para pesos numéricos de preferência implícita.

    Args:
        df: DataFrame com coluna 'event'.

    Returns:
        DataFrame com coluna 'weight' adicionada.
    """
    df = df.copy()
    df["weight"] = df["event"].map(EVENT_WEIGHTS)
    return df


class EventsPreprocessor(Preprocessor):
    """Pré-processador para o dataset de eventos de navegação.

    Converte timestamp, remove duplicatas exatas, atribui pesos
    e remove transactionid (irrelevante para o modelo).
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa e enriquece o dataset de eventos.

        Args:
            df: DataFrame bruto com colunas de eventos.

        Returns:
            DataFrame processado sem transactionid e com coluna weight.
        """
        df = df.copy()
        df = to_datetime_utc(df, "timestamp")
        df = df.drop_duplicates(subset=_DUPLICATE_SUBSET)
        df = add_event_weight(df)
        return df.drop(columns=["transactionid"])