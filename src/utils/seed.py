"""Utilitários para reprodutibilidade de experimentos."""

import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Fixa todas as seeds para garantir reprodutibilidade total.

    Deve ser chamada como primeira instrução do stage de treino,
    antes de qualquer inicialização de modelo ou DataLoader.

    Args:
        seed: Valor da seed. Padrão: 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False