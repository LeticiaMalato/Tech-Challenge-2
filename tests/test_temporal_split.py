"""Testes unitários para o split temporal treino/teste."""

import pandas as pd
import pytest

from src.feature.temporal_split import split_by_time


def _make_interactions(n: int, n_users: int = 5, n_items: int = 5) -> pd.DataFrame:
    """Gera n interações sobre um conjunto pequeno e fixo de usuários/itens.

    Os usuários e itens se repetem ciclicamente ao longo de todo o
    período de tempo — necessário para que existam usuários em comum
    entre treino e teste, já que o cold-start filter do split_by_time
    removeria o teste inteiro se cada usuário aparecesse só uma vez.

    Args:
        n: Número de interações a gerar.
        n_users: Número de usuários distintos, repetidos ciclicamente.
        n_items: Número de itens distintos, repetidos ciclicamente.

    Returns:
        DataFrame com colunas visitorid, itemid e timestamp.
    """
    return pd.DataFrame(
        {
            "visitorid": [i % n_users for i in range(n)],
            "itemid": [100 + (i % n_items) for i in range(n)],
            "timestamp": pd.date_range("2015-01-01", periods=n, freq="h", tz="UTC"),
        }
    )


def test_split_by_time_respeita_proporcao_aproximada() -> None:
    """O corte temporal deve gerar aproximadamente test_ratio de eventos.

    Medido no conjunto de teste, antes do filtro de cold-start.
    """
    df = _make_interactions(100)
    train, test = split_by_time(df, test_ratio=0.2)

    assert len(train) + len(test) <= len(df)
    assert len(train) > 0
    assert len(test) > 0


def test_split_by_time_train_sempre_antes_do_test() -> None:
    """Nenhum timestamp de treino deve ser posterior ao de teste."""
    df = _make_interactions(50)
    train, test = split_by_time(df, test_ratio=0.3)

    assert train["timestamp"].max() <= test["timestamp"].min()


def test_split_by_time_remove_cold_start_usuarios() -> None:
    """Usuário que só aparece no período de teste deve ser removido do teste."""
    df = pd.DataFrame(
        {
            "visitorid": [1, 1, 2],
            "itemid": [10, 20, 30],
            "timestamp": pd.to_datetime(
                ["2015-01-01", "2015-01-02", "2015-01-10"], utc=True
            ),
        }
    )
    train, test = split_by_time(df, test_ratio=0.3)

    assert 2 not in set(test["visitorid"])


def test_split_by_time_remove_cold_start_itens() -> None:
    """Item que só aparece no período de teste deve ser removido do teste."""
    df = pd.DataFrame(
        {
            "visitorid": [1, 1, 1],
            "itemid": [10, 20, 999],
            "timestamp": pd.to_datetime(
                ["2015-01-01", "2015-01-02", "2015-01-10"], utc=True
            ),
        }
    )
    train, test = split_by_time(df, test_ratio=0.3)

    assert 999 not in set(test["itemid"])


def test_split_by_time_ratio_invalido_zero_levanta_erro() -> None:
    """test_ratio=0.0 deve levantar ValueError."""
    df = _make_interactions(10)
    with pytest.raises(ValueError):
        split_by_time(df, test_ratio=0.0)


def test_split_by_time_ratio_invalido_um_levanta_erro() -> None:
    """test_ratio=1.0 deve levantar ValueError."""
    df = _make_interactions(10)
    with pytest.raises(ValueError):
        split_by_time(df, test_ratio=1.0)


def test_split_by_time_ratio_maior_que_um_levanta_erro() -> None:
    """test_ratio > 1.0 deve levantar ValueError."""
    df = _make_interactions(10)
    with pytest.raises(ValueError):
        split_by_time(df, test_ratio=1.5)
