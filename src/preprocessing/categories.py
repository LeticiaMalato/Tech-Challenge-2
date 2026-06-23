"""Pré-processamento do dataset de categorias."""

import pandas as pd

from src.preprocessing.base import Preprocessor

ROOT_SENTINEL: int = -1


class CategoriesPreprocessor(Preprocessor):
    """Pré-processador para o dataset de árvore de categorias.

    Classifica a categoria raiz com um valor sentinela para evitar
    perda de dados e manter a hierarquia intacta.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preenche parentid nulo com sentinela e converte para int.

        Args:
            df: DataFrame com colunas [categoryid, parentid].

        Returns:
            DataFrame com parentid sem nulos e tipado como int.
        """
        df = df.copy()
        df["parentid"] = df["parentid"].fillna(ROOT_SENTINEL).astype(int)
        return df
