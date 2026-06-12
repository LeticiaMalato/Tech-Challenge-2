#baseline de recomendação que sugere os itens mais populares com base nas interações dos usuários. O modelo é simples e não personalizado, ou seja, ele recomenda os mesmos itens para todos os usuários, independentemente de suas preferências individuais. O ranking de popularidade é calculado somando os pesos das interações para cada item e ordenando-os do mais popular para o menos popular.
import pandas as pd

from src.models.base import Recommender


def _compute_popularity_ranking(interactions: pd.DataFrame) -> list[int]:
  #Calcula o ranking global de popularidade dos itens com base nas interações fornecidas. O ranking é determinado pela soma dos pesos das interações para cada item, ordenados do mais popular para o menos popular. O resultado é uma lista de IDs dos itens ordenada por popularidade.
    ranking = (
        interactions
        .groupby("itemid")["weight"]
        .sum()
        .sort_values(ascending=False)
    )
    return ranking.index.tolist()


class PopularityRecommender(Recommender):
#O método __init__ é o construtor da classe PopularityRecommender. Ele inicializa um atributo privado _ranking como uma lista vazia. Esse atributo será usado para armazenar o ranking global de popularidade dos itens, que será calculado posteriormente no método fit.
    def __init__(self) -> None:
        self._ranking: list[int] = []

    def fit(self, interactions: pd.DataFrame) -> None:
        self._interactions = interactions.copy() 
#calcula o ranking global de popularidade dos itens com base nas interações fornecidas e armazena esse ranking no atributo _ranking. O método utiliza a função _compute_popularity_ranking para realizar o cálculo do ranking, que é então armazenado para ser usado posteriormente na geração de recomendações.
        self._ranking = _compute_popularity_ranking(interactions)

    def recommend(self, user_id: int, k: int) -> list[int]:
#deve retornar os k itens mais populares para o usuário especificado. No entanto, como este é um modelo de recomendação baseado na popularidade global, ele não leva em consideração as preferências individuais do usuário. Portanto, ele simplesmente retorna os k itens mais populares do ranking global, independentemente do ID do usuário fornecido.
        seen = set(
        self._interactions[
            self._interactions["visitorid"] == user_id
        ]["itemid"]
          )
        return [item for item in self._ranking if item not in seen][:k]