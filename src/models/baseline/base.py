#innterface comum para todos os recomendadores, para garantir que eles tenham os mesmos métodos e possam ser usados de forma intercambiável.
import pandas as pd

class Recommender:
#O que todo recomendador deve implementar para ser considerado válido. Ele define os métodos fit e recommend, mas não implementa a lógica específica de cada um, deixando isso para as subclasses que herdam dessa classe base.
    def fit(self, interactions: pd.DataFrame) -> None:
#treina o modelo de recomendação usando os dados de interações fornecidos. O DataFrame de interações deve conter informações sobre as interações dos usuários com os itens, como visualizações, cliques, compras, etc. A implementação específica do treinamento do modelo será feita nas subclasses que herdam dessa classe base.
        raise NotImplementedError

    def recommend(self, user_id: int, k: int) -> list[int]:
#gera uma lista de recomendações para um usuário específico. O método recebe o ID do usuário e o número de recomendações desejadas (k) e retorna uma lista de IDs dos itens recomendados. A lógica para gerar as recomendações será implementada nas subclasses que herdam dessa classe base.
        raise NotImplementedError