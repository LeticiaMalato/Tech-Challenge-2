"""Testes unitários para o filtro k-core de densificação."""

import pandas as pd
import pytest

from src.feature.kcore import apply_kcore


def test_apply_kcore_remove_usuarios_abaixo_do_limiar() -> None:
    """Item com menos de k interações deve ser removido.

    Mesmo que o usuário sozinho satisfaça o limiar.
    """
    df = pd.DataFrame(
        {
            "visitorid": [1, 1, 1, 2],
            "itemid": [10, 10, 10, 20],
        }
    )
    result = apply_kcore(df, k=3)

    assert set(result["visitorid"]) == {1}
    assert len(result) == 3


def test_apply_kcore_remove_itens_abaixo_do_limiar() -> None:
    """Item com menos de k interações deve ser removido.

    Mesmo que o usuário sozinho satisfaça o limiar.
    """
    df = pd.DataFrame(
        {
            "visitorid": [1, 1, 1, 2, 2, 2],
            "itemid": [10, 10, 10, 20, 30, 40],
        }
    )
    result = apply_kcore(df, k=3)

    assert set(result["itemid"]) == {10}
    assert set(result["visitorid"]) == {1}


def test_apply_kcore_convergencia_iterativa() -> None:
    """Remover um usuário pode derrubar um item abaixo do limiar.

    Exigindo mais de uma iteração até estabilizar.
    """
    df = pd.DataFrame(
        {
            # item 99 só tem 2 interações: usuário 3 (que será removido
            # por atividade insuficiente) e usuário 1.
            "visitorid": [1, 1, 1, 2, 2, 2, 3],
            "itemid": [10, 20, 99, 10, 20, 30, 99],
        }
    )
    result = apply_kcore(df, k=3)

    assert 3 not in set(result["visitorid"])
    assert 99 not in set(result["itemid"])


def test_apply_kcore_dataset_ja_denso_nao_remove_nada() -> None:
    """Quando todos já satisfazem k, o resultado é idêntico ao input."""
    df = pd.DataFrame(
        {
            "visitorid": [1, 1, 2, 2],
            "itemid": [10, 20, 10, 20],
        }
    )
    result = apply_kcore(df, k=2)

    assert len(result) == len(df)


def test_apply_kcore_k_zero_levanta_erro() -> None:
    """k=0 deve levantar ValueError."""
    df = pd.DataFrame({"visitorid": [1], "itemid": [10]})
    with pytest.raises(ValueError):
        apply_kcore(df, k=0)


def test_apply_kcore_k_negativo_levanta_erro() -> None:
    """K negativo deve levantar ValueError."""
    df = pd.DataFrame({"visitorid": [1], "itemid": [10]})
    with pytest.raises(ValueError):
        apply_kcore(df, k=-1)
