"""Entrada do stage de pré-processamento do pipeline DVC."""
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import settings
from src.preprocessing.factory import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_RAW_FILES: dict[str, list[str]] = {
    "events": ["events.csv"],
    "categories": ["category_tree.csv"],
    "item_properties": ["item_properties_part1.csv", "item_properties_part2.csv"],
}


def load_raw(source: str, raw_dir: Path) -> pd.DataFrame:
    """Lê e concatena os arquivos CSV brutos de uma fonte.

    Args:
        source: Identificador da fonte de dados.
        raw_dir: Diretório onde os CSVs brutos estão armazenados.

    Returns:
        DataFrame com todos os dados brutos da fonte concatenados.

    Raises:
        KeyError: Se a fonte não estiver mapeada em _RAW_FILES.
    """
    if source not in _RAW_FILES:
        raise KeyError(f"Fonte desconhecida: {source!r}")
    parts = [pd.read_csv(raw_dir / fname) for fname in _RAW_FILES[source]]
    return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]


def run(source: str, raw_dir: Path, out_dir: Path) -> Path:
    """Executa o pré-processamento de uma fonte e salva em parquet.

    Args:
        source: Identificador da fonte de dados.
        raw_dir: Diretório dos dados brutos.
        out_dir: Diretório de saída para os dados processados.

    Returns:
        Path do arquivo parquet gerado.
    """
    logger.info("Iniciando pré-processamento: %s", source)
    df = load_raw(source, raw_dir)
    clean = build_preprocessor(source).transform(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source}.parquet"
    clean.to_parquet(out_path, index=False)
    logger.info("Salvo em: %s | Shape: %s", out_path, clean.shape)
    return out_path


def main() -> None:
    """CLI: python -m src.preprocessing.run --source events."""
    parser = argparse.ArgumentParser(description="Stage de pré-processamento")
    parser.add_argument("--source", required=True, choices=list(_RAW_FILES))
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_data_dir)
    parser.add_argument("--out-dir", type=Path, default=settings.processed_data_dir)
    args = parser.parse_args()
    run(args.source, args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()