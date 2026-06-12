import pandas as pd

class IdEncoder:
#mapeia IDs originais (como visitorid e itemid) para índices inteiros sequenciais, que são usados para treinar modelos de recomendação baseados em embeddings. O encoder constrói um vocabulário a partir dos IDs únicos encontrados no conjunto de treino, e fornece métodos para transformar IDs em índices e decodificar índices de volta para os IDs originais. Isso é essencial para garantir que os modelos de recomendação possam trabalhar com dados categóricos de forma eficiente, usando os índices inteiros como entrada para camadas de embedding.

    def __init__(self) -> None:
        self._id_to_index: dict[int, int] = {}
        self._index_to_id: list[int] = []

    def fit(self, ids: pd.Series) -> None:
#Constrói o mapeamento a partir dos IDs únicos da série.
        unique_ids = sorted(ids.unique().tolist())
        self._index_to_id = unique_ids
        self._id_to_index = {id_: i for i, id_ in enumerate(unique_ids)}

    def transform(self, ids: pd.Series) -> pd.Series:
#Converte IDs originais para índices inteiros
        return ids.map(self._id_to_index)

    def decode(self, index: int) -> int:
#Converte um índice de volta para o ID original.
        return self._index_to_id[index]

    @property
    def vocab_size(self) -> int:
#Número de IDs únicos — usado para definir o tamanho do Embedding.
        return len(self._index_to_id)


def encode_interactions(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, IdEncoder, IdEncoder]:
#Ajusta os IDs de usuário e item para índices inteiros usando IdEncoder. A função recebe os DataFrames de treino e teste, cria encoders para os IDs de usuário e item com base no conjunto de treino, e transforma os IDs originais em índices inteiros. O resultado é uma tupla contendo os DataFrames de treino e teste com as colunas "user_idx" e "item_idx" substituindo "visitorid" e "itemid", respectivamente, além dos encoders usados para a transformação. Isso prepara os dados para serem usados em modelos de recomendação que exigem entradas numéricas.
    user_enc = IdEncoder()
    item_enc = IdEncoder()

    user_enc.fit(train["visitorid"])
    item_enc.fit(train["itemid"])

    train = train.copy()
    test = test.copy()

    train["user_idx"] = user_enc.transform(train["visitorid"])
    train["item_idx"] = item_enc.transform(train["itemid"])

    test["user_idx"] = user_enc.transform(test["visitorid"])
    test["item_idx"] = item_enc.transform(test["itemid"])

    return train, test, user_enc, item_enc