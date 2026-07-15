"""Testes unitários para IdEncoder e encode_interactions."""

import pandas as pd
import pytest

from src.feature.encoders import IdEncoder, encode_interactions

# IdEncoder


def test_id_encoder_fit_transform_mapeia_para_indices_contiguos() -> None:
    """IDs únicos devem virar índices 0..N-1."""
    enc = IdEncoder()
    ids = pd.Series([30, 10, 20, 10])
    enc.fit(ids)
    transformed = enc.transform(ids)

    assert set(transformed) == {0, 1, 2}
    assert enc.vocab_size == 3


def test_id_encoder_transform_e_decode_sao_inversas() -> None:
    """decode(transform(id)) deve retornar o id original."""
    enc = IdEncoder()
    enc.fit(pd.Series([100, 200, 300]))
    encoded = enc.transform(pd.Series([200])).iloc[0]

    assert enc.decode(encoded) == 200


def test_id_encoder_id_desconhecido_vira_sentinela() -> None:
    """ID fora do vocabulário de fit deve ser mapeado para -1."""
    enc = IdEncoder()
    enc.fit(pd.Series([1, 2, 3]))
    transformed = enc.transform(pd.Series([999]))

    assert transformed.iloc[0] == -1


def test_id_encoder_vocab_size_vazio() -> None:
    """Encoder sem fit tem vocabulário vazio."""
    enc = IdEncoder()
    assert enc.vocab_size == 0


def test_id_encoder_decode_indice_invalido_levanta_erro() -> None:
    """Decodificar um índice fora do vocabulário levanta IndexError."""
    enc = IdEncoder()
    enc.fit(pd.Series([1, 2]))
    with pytest.raises(IndexError):
        enc.decode(99)


# encode_interactions


def test_encode_interactions_adiciona_colunas_idx() -> None:
    """Train e test devem ganhar as colunas user_idx e item_idx."""
    train = pd.DataFrame({"visitorid": [1, 2], "itemid": [10, 20]})
    test = pd.DataFrame({"visitorid": [1], "itemid": [10]})

    train_out, test_out, user_enc, item_enc = encode_interactions(train, test)

    assert "user_idx" in train_out.columns
    assert "item_idx" in train_out.columns
    assert "user_idx" in test_out.columns
    assert "item_idx" in test_out.columns


def test_encode_interactions_ajusta_encoder_so_no_treino() -> None:
    """Encoders devem refletir o vocabulário do treino, não do teste."""
    train = pd.DataFrame({"visitorid": [1, 2], "itemid": [10, 20]})
    test = pd.DataFrame({"visitorid": [1, 2, 3], "itemid": [10, 20, 30]})

    _, _, user_enc, item_enc = encode_interactions(train, test)

    assert user_enc.vocab_size == 2
    assert item_enc.vocab_size == 2


def test_encode_interactions_usuario_novo_no_teste_vira_sentinela() -> None:
    """Usuário presente só no teste deve ser codificado como -1."""
    train = pd.DataFrame({"visitorid": [1], "itemid": [10]})
    test = pd.DataFrame({"visitorid": [1, 2], "itemid": [10, 20]})

    _, test_out, _, _ = encode_interactions(train, test)

    novo_usuario_row = test_out[test_out["visitorid"] == 2].iloc[0]
    assert novo_usuario_row["user_idx"] == -1
    assert novo_usuario_row["item_idx"] == -1


def test_encode_interactions_nao_mescla_colunas_originais() -> None:
    """As colunas originais (visitorid, itemid) devem ser preservadas."""
    train = pd.DataFrame({"visitorid": [1], "itemid": [10]})
    test = pd.DataFrame({"visitorid": [1], "itemid": [10]})

    train_out, _, _, _ = encode_interactions(train, test)

    assert "visitorid" in train_out.columns
    assert "itemid" in train_out.columns
