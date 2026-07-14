"""Recomendador neural via Matrix Factorization com bias, treinado em PyTorch."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.baseline.base import Recommender
from src.models.neural.dataset import InteractionDataset
from src.models.neural.mlp import MatrixFactorizationNet
from src.models.neural.trainer import train

logger = logging.getLogger(__name__)

EpochCallback = Callable[[int, float, float], None]


@dataclass
class MLPConfig:
    """Hiperparâmetros do MLPRecommender.

    Attributes:
        embed_dim: Dimensão dos vetores de embedding latente.
        neg_ratio: Número de negativos amostrados por positivo.
        lr: Taxa de aprendizado do otimizador.
        weight_decay: Penalização L2 do AdamW.
        batch_size: Tamanho do batch de treino/validação.
        max_epochs: Número máximo de épocas.
        patience: Épocas sem melhora na validação antes do early stopping.
        seed: Semente para reprodutibilidade.
        num_workers: Número de workers do DataLoader.
        device: ``"auto"``, ``"cpu"`` ou ``"cuda"``.
    """

    embed_dim: int = 16
    neg_ratio: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 2048
    max_epochs: int = 50
    patience: int = 5
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"


def _resolve_device(device: str) -> torch.device:
    """Resolve a string de configuração para um ``torch.device``.

    Args:
        device: ``"auto"`` para detecção automática, ou nome explícito
            (``"cpu"``, ``"cuda"``).

    Returns:
        Instância de ``torch.device`` correspondente.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class MLPRecommender(Recommender):
    """Recomendador neural via Matrix Factorization com bias.

    Treina embeddings de usuário e item com negative sampling e
    BCEWithLogitsLoss, delegando o loop de treino (early stopping,
    scheduler, grad clipping) ao módulo ``src.models.neural.trainer``.
    """

    def __init__(
        self,
        config: MLPConfig | None = None,
        checkpoint_dir: Path = Path("models/checkpoints"),
    ) -> None:
        """Inicializa o recomendador com configuração e diretório de checkpoint.

        Args:
            config: Hiperparâmetros do modelo. Usa defaults se ``None``.
            checkpoint_dir: Diretório onde o checkpoint final é salvo.
        """
        self._cfg = config or MLPConfig()
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_path = checkpoint_dir / "mlp_best.pt"
        self._device = _resolve_device(self._cfg.device)
        self._net: MatrixFactorizationNet | None = None
        self._item_ids: list[int] = []
        self._user_index: dict[int, int] = {}
        self._item_index: dict[int, int] = {}
        self._seen_by_user: dict[int, set[int]] = {}
        self._interactions: pd.DataFrame = pd.DataFrame()
        self._item_popularity: np.ndarray = np.array([])
        self._history: list[dict] = []
        torch.manual_seed(self._cfg.seed)
        np.random.seed(self._cfg.seed)

    def fit(
        self,
        interactions: pd.DataFrame,
        epoch_callback: EpochCallback | None = None,
    ) -> "MLPRecommender":
        """Treina o modelo a partir das interações históricas.

        Args:
            interactions: DataFrame com colunas ``visitorid``, ``itemid``
                e, opcionalmente, ``timestamp``.
            epoch_callback: Função opcional chamada após cada epoch com
                ``(epoch, train_loss, val_loss)``. Permite que o chamador
                registre métricas (ex: MLflow) sem acoplar este modelo a
                nenhuma ferramenta de tracking específica.
        """
        interactions = interactions.copy()
        self._interactions = interactions
        self._build_index(interactions)
        self._compute_popularity(interactions)
        self._net = self._build_network()

        train_loader, val_loader, train_ds = self._build_loaders(interactions)
        self._net, self._history = train(
            self._net,
            train_loader,
            val_loader,
            epochs=self._cfg.max_epochs,
            lr=self._cfg.lr,
            weight_decay=self._cfg.weight_decay,
            patience=self._cfg.patience,
            device=self._device,
            pos_weight=float(self._cfg.neg_ratio),
            reshuffle_dataset=train_ds,
            epoch_callback=epoch_callback,
        )
        self._persist_checkpoint()
        return self

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Retorna os k itens mais recomendados para o usuário.

        Args:
            user_id: Identificador do usuário.
            k: Número máximo de recomendações.

        Returns:
            Lista de item_ids ordenados por score descendente.
            Retorna lista vazia se o usuário não foi visto no treino.
        """
        if user_id not in self._user_index or self._net is None:
            return []
        scores = self._score_candidates(user_id)
        self._mask_seen_items(scores, user_id)
        top_indices = np.argsort(scores)[::-1][:k]
        return [self._item_ids[i] for i in top_indices]

    # Indexação e popularidade--

    def _build_index(self, interactions: pd.DataFrame) -> None:
        """Re-indexa usuários e itens para índices locais contíguos 0..N-1.

        Também pré-computa o conjunto de itens vistos por usuário, usado em
        ``recommend()`` para mascarar sem escanear o DataFrame a cada chamada.

        Args:
            interactions: DataFrame de treino. É mutado com as colunas
                ``user_idx`` e ``item_idx``.
        """
        unique_users = interactions["visitorid"].unique()
        unique_items = interactions["itemid"].unique()
        self._user_index = {uid: idx for idx, uid in enumerate(unique_users)}
        self._item_index = {iid: idx for idx, iid in enumerate(unique_items)}
        self._item_ids = list(unique_items)
        interactions["user_idx"] = interactions["visitorid"].map(self._user_index)
        interactions["item_idx"] = interactions["itemid"].map(self._item_index)
        self._seen_by_user = (
            interactions.groupby("user_idx")["item_idx"].apply(set).to_dict()
        )
        logger.info(
            "Re-indexação local | usuários: %d | itens: %d",
            len(self._user_index),
            len(self._item_index),
        )

    def _compute_popularity(self, interactions: pd.DataFrame) -> None:
        """Calcula o vetor de popularidade dos itens por contagem de interações.

        Args:
            interactions: DataFrame já re-indexado com coluna ``item_idx``.
        """
        counts = interactions.groupby("item_idx").size()
        n_items = len(self._item_index)
        pop = np.zeros(n_items, dtype=np.float32)
        valid = counts[counts.index < n_items]
        pop[valid.index] = valid.to_numpy(dtype=np.float32)
        self._item_popularity = pop

    def _build_network(self) -> MatrixFactorizationNet:
        """Instancia a rede de Matrix Factorization no device configurado.

        Returns:
            Instância de ``MatrixFactorizationNet`` movida para o device.
        """
        n_users = len(self._user_index)
        n_items = len(self._item_index)
        logger.info("Rede | n_users=%d | n_items=%d", n_users, n_items)
        return MatrixFactorizationNet(
            n_users=n_users,
            n_items=n_items,
            embed_dim=self._cfg.embed_dim,
        ).to(self._device)

    # DataLoaders

    def _build_loaders(
        self, interactions: pd.DataFrame
    ) -> tuple[DataLoader, DataLoader, InteractionDataset]:
        """Constrói os DataLoaders de treino/validação via split leave-one-out.

        Args:
            interactions: DataFrame re-indexado com ``user_idx``/``item_idx``.

        Returns:
            Tupla (train_loader, val_loader, dataset de treino — necessário
            para o ``reshuffle()`` por epoch dentro do loop de treino).
        """
        train_df, val_df = self._split_leave_one_out(interactions)
        n_items = len(self._item_index)
        train_ds = InteractionDataset(
            train_df, n_items, self._cfg.neg_ratio, self._cfg.seed
        )
        val_ds = InteractionDataset(val_df, n_items, neg_ratio=1, seed=self._cfg.seed)
        return (
            self._make_loader(train_ds, shuffle=True),
            self._make_loader(val_ds, shuffle=False),
            train_ds,
        )

    def _split_leave_one_out(
        self, interactions: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Divide em treino/validação via leave-one-out temporal por usuário.

        O último evento de cada usuário (por timestamp, se disponível)
        vai para validação; o restante permanece no treino.

        Args:
            interactions: DataFrame re-indexado com ``user_idx``.

        Returns:
            Tupla (train_df, val_df).
        """
        if "timestamp" in interactions.columns:
            df = interactions.sort_values(["user_idx", "timestamp"]).reset_index(
                drop=True
            )
        else:
            df = interactions.reset_index(drop=True)

        user_counts = df.groupby("user_idx").size()
        eligible = user_counts[user_counts >= 2].index

        last_row_indices = (
            df[df["user_idx"].isin(eligible)]
            .groupby("user_idx", sort=False)["user_idx"]
            .apply(lambda g: g.index[-1])
            .to_numpy()
        )

        val_mask = np.zeros(len(df), dtype=bool)
        val_mask[last_row_indices] = True
        train_df = df[~val_mask].reset_index(drop=True)
        val_df = df[val_mask].reset_index(drop=True)

        cold = set(val_df["user_idx"]) - set(train_df["user_idx"])
        logger.info(
            "Leave-one-out | treino: %d (%d users) | "
            "val: %d (%d users) | cold-start: %d",
            len(train_df),
            train_df["user_idx"].nunique(),
            len(val_df),
            val_df["user_idx"].nunique(),
            len(cold),
        )
        return train_df, val_df

    def _make_loader(self, dataset: InteractionDataset, shuffle: bool) -> DataLoader:
        """Constrói um DataLoader a partir das configurações do modelo.

        Args:
            dataset: Dataset de interações.
            shuffle: Se True, embaralha as amostras a cada epoch.

        Returns:
            Instância configurada de ``DataLoader``.
        """
        return DataLoader(
            dataset,
            batch_size=self._cfg.batch_size,
            shuffle=shuffle,
            num_workers=self._cfg.num_workers,
            pin_memory=self._device.type == "cuda",
        )

    # Inferência

    @torch.no_grad()
    def _score_candidates(self, user_id: int) -> np.ndarray:
        """Calcula o score de todos os itens do catálogo para o usuário.

        score(u, i) = <user_emb[u], item_emb[i]> + b_u + b_i — consistente
        com a função objetivo usada no treino (``MatrixFactorizationNet``).

        Args:
            user_id: Identificador do usuário.

        Returns:
            Array de scores de shape (n_items,).
        """
        self._net.eval()
        n_items = len(self._item_ids)

        u_tensor = torch.tensor(
            [self._user_index[user_id]], dtype=torch.long, device=self._device
        )
        user_emb = self._net.user_emb(u_tensor).squeeze(0)
        user_bias = self._net.user_bias(u_tensor).squeeze()

        all_item_idx = torch.arange(n_items, dtype=torch.long, device=self._device)
        item_embs = self._net.item_emb(all_item_idx)
        item_bias = self._net.item_bias(all_item_idx).squeeze(-1)

        scores = (item_embs @ user_emb + item_bias + user_bias).cpu().numpy()
        return scores.astype(np.float32)

    def _mask_seen_items(self, scores: np.ndarray, user_id: int) -> None:
        """Marca itens já vistos pelo usuário com score ``-inf``.

        Args:
            scores: Array de scores de shape (n_items,), mutado in-place.
            user_id: Identificador do usuário.
        """
        seen = self._seen_by_user.get(self._user_index[user_id], set())
        if seen:
            scores[list(seen)] = -np.inf

    # Checkpoint
    @property
    def checkpoint_path(self) -> Path:
        """Caminho do checkpoint salvo pelo treino mais recente."""
        return self._checkpoint_path

    def _inference_payload(self) -> dict:
        """Monta o dicionário serializável necessário para inferência."""
        if self._net is None:
            raise RuntimeError("Não há rede treinada para persistir.")
        return {
            "state_dict": self._net.state_dict(),
            "config": self._cfg.__dict__,
            "user_index": self._user_index,
            "item_index": self._item_index,
            "item_ids": self._item_ids,
            "seen_by_user": self._seen_by_user,
        }

    def _persist_checkpoint(self) -> None:
        """Salva artefato completo de inferência em disco.

        Inclui pesos da rede e metadados necessários para ``recommend()``
        sem reler o parquet de treino. O ``EarlyStopping`` já restaurou
        os melhores pesos dentro de ``trainer.train`` antes desta chamada.
        """
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._inference_payload(), self._checkpoint_path)
        logger.info("Checkpoint de inferência salvo em %s.", self._checkpoint_path)

    def _restore_from_payload(self, payload: dict) -> None:
        """Restaura índices, itens vistos e pesos a partir do payload."""
        self._user_index = payload["user_index"]
        self._item_index = payload["item_index"]
        self._item_ids = list(payload["item_ids"])
        self._seen_by_user = payload["seen_by_user"]
        self._net = MatrixFactorizationNet(
            n_users=len(self._user_index),
            n_items=len(self._item_index),
            embed_dim=self._cfg.embed_dim,
        ).to(self._device)
        self._net.load_state_dict(payload["state_dict"])
        self._net.eval()

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "MLPRecommender":
        """Carrega um artefato de inferência e reconstrói o recomendador.

        Args:
            path: Caminho do arquivo ``.pt`` gerado por ``_persist_checkpoint``.
            device: Device PyTorch para a rede (``"cpu"`` ou ``"cuda"``).

        Returns:
            Instância pronta para ``recommend()``.

        Raises:
            FileNotFoundError: Se ``path`` não existir.
            KeyError: Se o artefato não tiver os campos esperados.
        """
        if not path.exists():
            raise FileNotFoundError(f"Artefato não encontrado: {path}")
        payload = torch.load(path, map_location=device, weights_only=False)
        config = MLPConfig(**{**payload["config"], "device": device})
        instance = cls(config=config, checkpoint_dir=path.parent)
        instance._checkpoint_path = path
        instance._restore_from_payload(payload)
        logger.info("Artefato de inferência carregado de %s.", path)
        return instance
