"""Pipeline de feature engineering: k-core → split → asof-join → encode."""

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from src.feature.asof_join import asof_join_properties
from src.feature.encoders import encode_interactions
from src.feature.kcore import apply_kcore
from src.feature.temporal_split import split_by_time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _load_raw_data(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega eventos e propriedades do diretório processado.

    Args:
        processed_dir: Diretório contendo ``events.parquet`` e
            ``item_properties.parquet``.

    Returns:
        Tupla (events, properties) como DataFrames.

    Raises:
        FileNotFoundError: Se algum dos arquivos não existir.
    """
    events_path = processed_dir / "events.parquet"
    props_path = processed_dir / "item_properties.parquet"
    for path in (events_path, props_path):
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_parquet(events_path), pd.read_parquet(props_path)


def _save_encoders(user_enc: object, item_enc: object, out_dir: Path) -> None:
    """Serializa os encoders em disco com pickle.

    Args:
        user_enc: ``IdEncoder`` de usuários.
        item_enc: ``IdEncoder`` de itens.
        out_dir: Diretório de saída.
    """
    with open(out_dir / "user_encoder.pkl", "wb") as f:
        pickle.dump(user_enc, f)
    with open(out_dir / "item_encoder.pkl", "wb") as f:
        pickle.dump(item_enc, f)


def run(processed_dir: Path, out_dir: Path, k: int, test_ratio: float) -> None:
    """Executa o pipeline completo de feature engineering.

    Etapas: carrega dados → k-core → split temporal →
    asof-join com propriedades → encode de IDs → salva artefatos.

    Args:
        processed_dir: Diretório com os dados processados de entrada.
        out_dir: Diretório de saída para features e encoders.
        k: Limiar mínimo de interações para o filtro k-core.
        test_ratio: Proporção do intervalo temporal destinada ao teste.
    """
    logger.info("Carregando dados de %s.", processed_dir)
    events, properties = _load_raw_data(processed_dir)

    logger.info("Aplicando k-core com k=%d.", k)
    events = apply_kcore(events, k=k)

    logger.info("Dividindo por tempo (test_ratio=%.2f).", test_ratio)
    train, test = split_by_time(events, test_ratio=test_ratio)

    logger.info("Executando merge as-of com propriedades dos itens.")
    train = asof_join_properties(train, properties)
    test = asof_join_properties(test, properties)

    logger.info("Codificando IDs de usuário e item.")
    train, test, user_enc, item_enc = encode_interactions(train, test)

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    _save_encoders(user_enc, item_enc, out_dir)

    logger.info(
        "Concluído. Treino: %d | Teste: %d | Usuários: %d | Itens: %d.",
        len(train), len(test), user_enc.vocab_size, item_enc.vocab_size,
    )


def main() -> None:
    """Ponto de entrada CLI do pipeline de feature engineering.

    Uso::

        python -m src.feature.run --k 5 --test-ratio 0.2
    """
    parser = argparse.ArgumentParser(description="Stage de feature engineering")
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed"),
        help="Diretório com events.parquet e item_properties.parquet.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/features"),
        help="Diretório de saída para os artefatos de features.",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Limiar mínimo de interações para o filtro k-core.",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.2,
        help="Proporção temporal destinada ao conjunto de teste (0 < valor < 1).",
    )
    args = parser.parse_args()
    run(args.processed_dir, args.out_dir, args.k, args.test_ratio)


if __name__ == "__main__":
    main()