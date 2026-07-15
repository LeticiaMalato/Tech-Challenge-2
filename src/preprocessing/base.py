"""Contrato base para todos os pré-processadores do pipeline."""

from abc import ABC, abstractmethod

import pandas as pd


class Preprocessor(ABC):
    """Interface base para pré-processadores de dados.

    Todos os pré-processadores concretos devem herdar desta classe
    e implementar o método transform.
    """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica transformações ao DataFrame bruto.

        Args:
            df: DataFrame com os dados brutos da fonte.

        Returns:
            DataFrame transformado e limpo.
        """
