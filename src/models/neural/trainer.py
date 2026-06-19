"""Loop de treino genérico com early stopping, scheduler e grad clipping."""

import copy
import logging
from typing import Protocol

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class _Reshuffleable(Protocol):
    """Qualquer dataset que saiba se re-amostrar entre epochs."""
    def reshuffle(self, epoch: int) -> None: ...


def _move_to_device(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    """Move um batch de tensores para o device de treino."""
    return tuple(t.to(device) for t in batch)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None,
) -> float:
    """Executa uma epoch de treino e retorna a loss média."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        u, i, y = _move_to_device(batch, device)
        optimizer.zero_grad()
        loss = criterion(model(u, i), y)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Avalia a loss de validação sem atualizar pesos."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            u, i, y = _move_to_device(batch, device)
            total_loss += criterion(model(u, i), y).item()
    return total_loss / len(loader)


def _is_improvement(val_loss: float, best_loss: float, min_delta: float) -> bool:
    """Verifica se houve melhora significativa na loss de validação."""
    return val_loss < best_loss - min_delta


class EarlyStopping:
    """Monitora a loss de validação e mantém os melhores pesos em memória.

    Args:
        patience: Épocas sem melhora antes de sinalizar parada.
        min_delta: Melhora mínima para ser considerada significativa.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._counter = 0
        self.best_loss = float("inf")
        self.best_weights: dict | None = None
        self.best_epoch = 0

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """Atualiza o estado e retorna True se o treino deve parar."""
        if _is_improvement(val_loss, self.best_loss, self.min_delta):
            self.best_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Restaura os melhores pesos no modelo."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def _build_optimizer(
    model: nn.Module, lr: float, weight_decay: float
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """Cria o otimizador AdamW e o scheduler ReduceLROnPlateau."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5, min_lr=1e-6
    )
    return optimizer, scheduler


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    pos_weight: float | None = None,
    grad_clip: float | None = 1.0,
    reshuffle_dataset: _Reshuffleable | None = None,
    epoch_callback=None,
) -> tuple[nn.Module, list[dict]]:
    """Treina o modelo com early stopping, scheduler e grad clipping.

    Args:
        model: Rede neural a treinar (já no device correto).
        train_loader: DataLoader de treino.
        val_loader: DataLoader de validação.
        epochs: Número máximo de épocas.
        lr: Taxa de aprendizado do AdamW.
        weight_decay: Penalização L2.
        patience: Épocas sem melhora antes do early stopping.
        device: Device onde os tensores são movidos.
        pos_weight: Peso da classe positiva no BCEWithLogitsLoss
            (ex: ``neg_ratio``, para compensar o desbalanceamento).
        grad_clip: Norma máxima do gradiente. ``None`` desativa.
        reshuffle_dataset: Objeto com método ``reshuffle(epoch)``,
            chamado no início de cada epoch. ``None`` desativa.
        epoch_callback: Função opcional ``(epoch, train_loss, val_loss)``
            chamada após cada epoch (ex: logging no MLflow).

    Returns:
        Tupla (modelo restaurado ao melhor checkpoint, histórico por época).
    """
    weight_t = torch.tensor([pos_weight], device=device) if pos_weight else None
    train_criterion = nn.BCEWithLogitsLoss(pos_weight=weight_t)
    val_criterion = nn.BCEWithLogitsLoss()
    optimizer, scheduler = _build_optimizer(model, lr, weight_decay)
    stopper = EarlyStopping(patience=patience)
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        if reshuffle_dataset is not None:
            reshuffle_dataset.reshuffle(epoch)

        train_loss = _train_epoch(model, train_loader, train_criterion, optimizer, device, grad_clip)
        val_loss = _val_epoch(model, val_loader, val_criterion, device)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("Epoch %02d | train=%.4f | val=%.4f", epoch, train_loss, val_loss)

        if epoch_callback is not None:
            epoch_callback(epoch, train_loss, val_loss)

        if stopper.step(val_loss, model, epoch):
            logger.info("Early stopping na epoch %d.", epoch)
            break

    stopper.restore_best(model)
    logger.info("Melhor época: %d | val_loss=%.4f", stopper.best_epoch, stopper.best_loss)
    return model, history