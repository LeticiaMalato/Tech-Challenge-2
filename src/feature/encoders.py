"""Encoders de IDs categóricos para índices inteiros sequenciais."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_UNKNOWN_SENTINEL = -1


class IdEncoder:
    """Mapeia IDs originais para índices inteiros sequenciais.

    Usado para converter ``visitorid`` e ``itemid`` em índices compatíveis
    com camadas de Embedding do PyTorch e operações de indexação numpy.
    IDs desconhecidos no ``transform`` recebem o sentinel ``-1``.

    Example:
        >>> enc = IdEncoder()
        >>> enc.fit(train_df["visitorid"])
        >>> train_df["user_idx"] = enc.transform(train_df["visitorid"])
    """

    def __init__(self) -> None:
        """Inicializa o encoder com vocabulário vazio."""
        self._id_to_index: dict[int, int] = {}
        self._index_to_id: list[int] = []

    def fit(self, ids: pd.Series) -> "IdEncoder":
        """Constrói o vocabulário a partir dos IDs únicos da série.

        Args:
            ids: Série de IDs inteiros (ex: ``train["visitorid"]``).

        Returns:
            A própria instância, permitindo encadeamento (ex:
            ``IdEncoder().fit(ids).transform(ids)``).
        """
        unique_ids = sorted(ids.unique().tolist())
        self._index_to_id = unique_ids
        self._id_to_index = {id_: i for i, id_ in enumerate(unique_ids)}
        return self

    def transform(self, ids: pd.Series) -> pd.Series:
        """Converte IDs originais em índices inteiros.

        IDs fora do vocabulário são mapeados para ``-1`` com aviso de log.

        Args:
            ids: Série de IDs a converter.

        Returns:
            Série de índices inteiros. IDs desconhecidos retornam ``-1``.
        """
        transformed = ids.map(self._id_to_index)
        n_unknown = transformed.isna().sum()
        if n_unknown > 0:
            logger.warning(
                "%d IDs fora do vocabulário encontrados; mapeados para %d.",
                n_unknown,
                _UNKNOWN_SENTINEL,
            )
        return transformed.fillna(_UNKNOWN_SENTINEL).astype(int)

    def decode(self, index: int) -> int:
        """Converte um índice de volta ao ID original.

        Args:
            index: Índice inteiro dentro do vocabulário.

        Returns:
            ID original correspondente ao índice.

        Raises:
            IndexError: Se ``index`` estiver fora dos limites do vocabulário.
        """
        return self._index_to_id[index]

    @property
    def vocab_size(self) -> int:
        """Número de IDs únicos no vocabulário.

        Usado para definir o tamanho da camada ``nn.Embedding`` no PyTorch.
        """
        return len(self._index_to_id)

    def __repr__(self) -> str:
        """Representação legível do encoder."""
        return f"IdEncoder(vocab_size={self.vocab_size})"


def encode_interactions(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, IdEncoder, IdEncoder]:
    """Codifica IDs de usuário e item em índices inteiros nos dois splits.

    Ajusta os encoders exclusivamente sobre o treino e aplica a transformação
    em ambos. Adiciona colunas ``user_idx`` e ``item_idx`` preservando as
    colunas originais. IDs do teste ausentes no treino recebem índice ``-1``.

    Args:
        train: DataFrame de treino com colunas ``visitorid`` e ``itemid``.
        test: DataFrame de teste com colunas ``visitorid`` e ``itemid``.

    Returns:
        Tupla com:
        - DataFrame de treino com ``user_idx`` e ``item_idx``;
        - DataFrame de teste com ``user_idx`` e ``item_idx``;
        - ``IdEncoder`` ajustado para ``visitorid``;
        - ``IdEncoder`` ajustado para ``itemid``.
    """
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
