"""Entrada do stage de pré-processamento do pipeline DVC."""
import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing.factory import build_preprocessor

_RAW_FILES: dict[str, str] = {
    "events": "events.csv",
    "categories": "category_tree.csv"
}


def load_raw(source: str, raw_dir: Path) -> pd.DataFrame:
    #le os csvs brutos. No caso de item_properties, são dois arquivos que precisam ser concatenados.
    if source == "item_properties":
        parts = [pd.read_csv(raw_dir / f"item_properties_part{i}.csv")
                 for i in (1, 2)]
        return pd.concat(parts, ignore_index=True)
    return pd.read_csv(raw_dir / _RAW_FILES[source])


def run(source: str, raw_dir: Path, out_dir: Path) -> Path:
 #Executa o pré-processamento da fonte e salva o resultado em parquet.
    df = load_raw(source, raw_dir)
    clean = build_preprocessor(source).transform(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source}.parquet"
    clean.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    #CLI: python -m src.preprocessing.run --source events.
    parser = argparse.ArgumentParser(description="Stage de pré-processamento")
    parser.add_argument("--source", required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    print(f"salvo: {run(args.source, args.raw_dir, args.out_dir)}")


if __name__ == "__main__":
    main()