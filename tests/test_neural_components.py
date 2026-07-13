"""Testes unitários para EarlyStopping e InteractionDataset."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.neural.dataset import InteractionDataset
from src.models.neural.trainer import EarlyStopping


def _make_model() -> nn.Module:
    """Cria um modelo linear simples para testar checkpointing."""
    return nn.Linear(2, 1)


# EarlyStopping


def test_early_stopping_melhora_nao_sinaliza_parada() -> None:
    """Uma melhora na loss não deve sinalizar parada."""
    stopper = EarlyStopping(patience=2)
    should_stop = stopper.step(val_loss=0.5, model=_make_model(), epoch=1)

    assert should_stop is False
    assert stopper.best_loss == 0.5
    assert stopper.best_epoch == 1


def test_early_stopping_conta_epocas_sem_melhora() -> None:
    """Losses piores em sequência devem incrementar o contador interno."""
    stopper = EarlyStopping(patience=3)
    model = _make_model()

    stopper.step(val_loss=0.5, model=model, epoch=1)
    stopper.step(val_loss=0.6, model=model, epoch=2)  # piora

    assert stopper._counter == 1
    assert stopper.best_loss == 0.5


def test_early_stopping_sinaliza_parada_apos_patience_esgotada() -> None:
    """Após patience epochs sem melhora, step deve retornar True."""
    stopper = EarlyStopping(patience=2)
    model = _make_model()

    stopper.step(val_loss=0.5, model=model, epoch=1)
    stopper.step(val_loss=0.6, model=model, epoch=2)
    should_stop = stopper.step(val_loss=0.7, model=model, epoch=3)

    assert should_stop is True


def test_early_stopping_min_delta_ignora_melhora_insignificante() -> None:
    """Melhora menor que min_delta não deve resetar o contador."""
    stopper = EarlyStopping(patience=2, min_delta=0.1)
    model = _make_model()

    stopper.step(val_loss=0.500, model=model, epoch=1)
    stopper.step(val_loss=0.495, model=model, epoch=2)  # melhora < min_delta

    assert stopper._counter == 1


def test_early_stopping_restore_best_recupera_melhores_pesos() -> None:
    """restore_best deve restaurar os pesos salvos na melhor época."""
    stopper = EarlyStopping(patience=5)
    model = _make_model()

    with torch.no_grad():
        model.weight.fill_(1.0)
    stopper.step(val_loss=0.5, model=model, epoch=1)  # salva pesos com weight=1.0

    with torch.no_grad():
        model.weight.fill_(99.0)  # simula degradação em epochs seguintes

    stopper.restore_best(model)

    assert torch.allclose(model.weight, torch.ones_like(model.weight))


# InteractionDataset


def _make_positive_interactions() -> pd.DataFrame:
    """Gera interações positivas cobrindo parte de um espaço 5x5 (user x item)."""
    return pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 2, 3, 4],
            "item_idx": [0, 1, 2, 3, 4, 0],
        }
    )


def test_dataset_len_inclui_positivos_e_negativos() -> None:
    """O tamanho do dataset deve ser positivos * (1 + neg_ratio)."""
    df = _make_positive_interactions()
    ds = InteractionDataset(df, n_items=5, neg_ratio=2, seed=42)

    assert len(ds) == len(df) * (1 + 2)


def test_dataset_negativos_nao_colidem_com_positivos() -> None:
    """Nenhum par negativo amostrado deve coincidir com um par positivo real.

    Regressão direta da vetorização do negative sampling via chaves
    int64 + np.isin — garante que a checagem de colisão continua correta.
    """
    df = _make_positive_interactions()
    ds = InteractionDataset(df, n_items=5, neg_ratio=4, seed=42)

    positive_pairs = set(zip(df["user_idx"], df["item_idx"], strict=True))
    for idx in range(len(ds)):
        user, item, label = ds[idx]
        if label.item() == 0.0:
            assert (user.item(), item.item()) not in positive_pairs


def test_dataset_reshuffle_gera_negativos_diferentes_entre_epochs() -> None:
    """Reshuffle com epochs diferentes deve alterar os negativos amostrados."""
    df = _make_positive_interactions()
    ds = InteractionDataset(df, n_items=5, neg_ratio=3, seed=42)

    before = ds._item_idxs.copy()
    ds.reshuffle(epoch=1)
    after = ds._item_idxs.copy()

    assert not np.array_equal(before, after)


def test_dataset_getitem_retorna_dtypes_esperados() -> None:
    """__getitem__ deve retornar tensores nos dtypes esperados pelo modelo."""
    df = _make_positive_interactions()
    ds = InteractionDataset(df, n_items=5, neg_ratio=1, seed=0)

    user, item, label = ds[0]
    assert user.dtype == torch.int64
    assert item.dtype == torch.int64
    assert label.dtype == torch.float32
