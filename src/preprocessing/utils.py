#Transformações reutilizáveis para os datasets de eventos e produtos. 
import pandas as pd


def to_datetime_utc(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(df[column], unit="ms", utc=True)
    return df