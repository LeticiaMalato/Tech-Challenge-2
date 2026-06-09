# Pre processamento do dataset categorias.
import pandas as pd

from src.preprocessing.base import Preprocessor

ROOT_SENTINEL: int = -1


class CategoriesPreprocessor(Preprocessor):
    #Classifica a categoria raiz com um valor sentinela para evitar perda de dados e manter a hierarquia.
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["parentid"] = df["parentid"].fillna(ROOT_SENTINEL).astype(int)
        return df