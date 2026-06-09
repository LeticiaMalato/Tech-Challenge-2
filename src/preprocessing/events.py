#Pre Processmanento do dataset de eventos.
import pandas as pd

from src.preprocessing.base import Preprocessor
from src.preprocessing.utils import to_datetime_utc

EVENT_WEIGHTS: dict[str, int] = {"view": 1, "addtocart": 2, "transaction": 3}


def add_event_weight(df: pd.DataFrame) -> pd.DataFrame:
    #Mapeia os tipos de evento para pesos numéricos, refletindo a força do sinal de preferência implícita.
    df = df.copy()
    df["weight"] = df["event"].map(EVENT_WEIGHTS)
    return df


class EventsPreprocessor(Preprocessor):
    #Converte timestamp, remove duplicatas, atribui pesos e dropa transactionid, que é irrelevante para o modelo e tem muitos valores nulos.
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = to_datetime_utc(df, "timestamp")
        df = df.drop_duplicates()
        df = add_event_weight(df)
        return df.drop(columns=["transactionid"])