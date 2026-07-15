"""Interface base para todos os recomendadores do sistema."""

from abc import ABC, abstractmethod

import pandas as pd


class Recommender(ABC):
    """Interface comum para todos os recomendadores.

    Define o contrato que todo recomendador deve seguir:
    ajuste sobre interações históricas e geração de recomendações
    personalizadas por usuário.
    """

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> "Recommender":
        """Treina o recomendador a partir das interações históricas.

        Args:
            interactions: DataFrame com colunas ``visitorid``, ``itemid``
                e ``weight``, representando eventos de interação usuário-item.
        """
        ...

    @abstractmethod
    def recommend(self, user_id: int, k: int) -> list[int]:
        """Retorna os k itens mais recomendados para o usuário.

        Itens já vistos pelo usuário devem ser excluídos da lista.

        Args:
            user_id: Identificador do usuário.
            k: Número máximo de recomendações a retornar.

        Returns:
            Lista de item_ids ordenados por score descendente,
            com no máximo k elementos.
        """
        ...
