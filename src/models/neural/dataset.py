"""Dataset PyTorch para pares positivos/negativos de interação usuário-item."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_MAX_OVERSAMPLE_FACTOR = 3


class InteractionDataset(Dataset):
    """Dataset de interações com negative sampling vetorizado e reshuffling por epoch.

    Args:
        interactions: DataFrame com colunas ``user_idx`` e ``item_idx``.
        n_items: Tamanho total do vocabulário de itens.
        neg_ratio: Número de negativos por positivo.
        seed: Semente para reprodutibilidade da amostragem.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        n_items: int,
        neg_ratio: int = 4,
        seed: int = 42,
    ) -> None:
        self._n_items = n_items
        self._neg_ratio = neg_ratio
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._pos_users = interactions["user_idx"].to_numpy(dtype=np.int32)
        self._pos_items = interactions["item_idx"].to_numpy(dtype=np.int32)

        # Codifica cada par (user, item) como um único inteiro para lookup O(1)
        # sem loop Python: user * n_items + item → chave única por par.
        self._seen_keys = self._build_seen_keys(self._pos_users, self._pos_items)

        self._user_idxs, self._item_idxs, self._labels = self._build_samples()

    def reshuffle(self, epoch: int = 0) -> None:
        """Re-amostra os negativos com nova semente derivada da epoch.

        Chamar no início de cada epoch garante que o modelo veja
        negativos diferentes a cada passagem, forçando generalização
        em vez de memorização de pares fixos.

        Args:
            epoch: Número da epoch atual, usado para variar a semente.
        """
        self._rng = np.random.default_rng(self._seed + epoch)
        self._user_idxs, self._item_idxs, self._labels = self._build_samples()

    def _build_seen_keys(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Codifica pares (user, item) como inteiros únicos para lookup vetorizado.

        Usa a fórmula ``user * n_items + item`` para mapear cada par a um
        inteiro único, permitindo checagem de colisão via ``np.isin`` em vez
        de loop Python sobre um set de tuplas.

        Args:
            users: Array de user_idx dos positivos.
            items: Array de item_idx dos positivos.

        Returns:
            Array de int64 com as chaves únicas dos pares positivos.
        """
        return (users.astype(np.int64) * self._n_items + items.astype(np.int64))

    def _build_samples(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Constrói arrays de (user, item, label) com negativos vetorizados.

        Também cacheia as versões em tensor para evitar criar tensores
        novos a cada chamada de __getitem__.
        """
        neg_users, neg_items = self._sample_negatives_vectorized(self._pos_users)
        user_idxs, item_idxs, labels = self._concat_samples(
            self._pos_users, self._pos_items, neg_users, neg_items
        )
        self._user_t = torch.from_numpy(user_idxs.astype(np.int64))
        self._item_t = torch.from_numpy(item_idxs.astype(np.int64))
        self._label_t = torch.from_numpy(labels)
        return user_idxs, item_idxs, labels

    def _collision_mask(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Retorna máscara booleana True para pares que colidem com positivos.

        Usa ``np.isin`` sobre chaves inteiras — completamente vetorizado,
        sem loop Python, ~100x mais rápido que iterar sobre set de tuplas.

        Args:
            users: Array de user_idx candidatos.
            items: Array de item_idx candidatos.

        Returns:
            Array booleano de shape (len(users),).
        """
        candidate_keys = users.astype(np.int64) * self._n_items + items.astype(np.int64)
        return np.isin(candidate_keys, self._seen_keys)

    def _sample_negatives_vectorized(
        self,
        pos_users: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Amostra negativos com rejeição vetorizada por lotes.

        Args:
            pos_users: Array de user_idx dos positivos.

        Returns:
            Tupla (neg_users, neg_items) como arrays int32.
        """
        n_needed = len(pos_users) * self._neg_ratio
        neg_users = np.repeat(pos_users, self._neg_ratio)
        neg_items = self._rng.integers(0, self._n_items, size=n_needed, dtype=np.int32)

        mask = self._collision_mask(neg_users, neg_items)
        neg_users, neg_items = neg_users[~mask], neg_items[~mask]

        while len(neg_users) < n_needed:
            neg_users, neg_items = self._fill_missing(neg_users, neg_items, n_needed)

        return neg_users[:n_needed], neg_items[:n_needed]

    def _fill_missing(
        self,
        neg_users: np.ndarray,
        neg_items: np.ndarray,
        n_needed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gera candidatos extras para repor colisões rejeitadas.

        Args:
            neg_users: Array de negativos válidos até agora.
            neg_items: Array de negativos válidos até agora.
            n_needed: Quantidade total necessária.

        Returns:
            Arrays atualizados com os candidatos extras válidos.
        """
        n_missing = n_needed - len(neg_users)
        extra_size = n_missing * _MAX_OVERSAMPLE_FACTOR
        extra_users = np.resize(neg_users[:n_missing], extra_size)
        extra_items = self._rng.integers(0, self._n_items, size=extra_size, dtype=np.int32)
        mask = self._collision_mask(extra_users, extra_items)
        valid_u = extra_users[~mask][:n_missing]
        valid_i = extra_items[~mask][:n_missing]
        return np.concatenate([neg_users, valid_u]), np.concatenate([neg_items, valid_i])

    def _concat_samples(
        self,
        pos_users: np.ndarray,
        pos_items: np.ndarray,
        neg_users: np.ndarray,
        neg_items: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Concatena positivos e negativos com seus rótulos."""
        user_idxs = np.concatenate([pos_users, neg_users])
        item_idxs = np.concatenate([pos_items, neg_items])
        labels = np.array(
            [1.0] * len(pos_users) + [0.0] * len(neg_users), dtype=np.float32
        )
        return user_idxs, item_idxs, labels

    def __len__(self) -> int:
        """Número total de amostras (positivos + negativos)."""
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna (user_idx, item_idx, label) indexando tensores pré-alocados."""
        return self._user_t[idx], self._item_t[idx], self._label_t[idx]