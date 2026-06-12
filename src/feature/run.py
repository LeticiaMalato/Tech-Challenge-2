
import argparse
import pickle
from pathlib import Path

import pandas as pd

from src.feature.asof_join import asof_join_properties
from src.feature.encoders import encode_interactions
from src.feature.kcore import apply_kcore
from src.feature.temporal_split import split_by_time


def run(processed_dir: Path, out_dir: Path, k: int, test_ratio: float) -> None:
 #executa o pipeline de feature engineering completo, desde a leitura dos dados processados até a escrita dos conjuntos de treino e teste prontos para modelagem. O processo inclui a aplicação da filtragem k-core para garantir um dataset mais denso, a divisão temporal dos dados em treino e teste para evitar leakage de informação futura, a junção as-of para incorporar as propriedades dos itens às interações, e a codificação dos IDs de usuário e item em índices inteiros usando IdEncoder. O resultado é salvo em arquivos Parquet para os conjuntos de treino e teste, e os encoders são serializados com pickle para uso posterior na fase de modelagem.
    events = pd.read_parquet(processed_dir / "events.parquet")
    properties = pd.read_parquet(processed_dir / "item_properties.parquet")

    events = apply_kcore(events, k=k)
    train, test = split_by_time(events, test_ratio=test_ratio)
    train = asof_join_properties(train, properties)
    test = asof_join_properties(test, properties)
    train, test, user_enc, item_enc = encode_interactions(train, test)

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    with open(out_dir / "user_encoder.pkl", "wb") as f:
        pickle.dump(user_enc, f)
    with open(out_dir / "item_encoder.pkl", "wb") as f:
        pickle.dump(item_enc, f)

    print(f"treino: {len(train):,} interações | teste: {len(test):,} interações")
    print(f"usuários: {user_enc.vocab_size:,} | itens: {item_enc.vocab_size:,}")


def main() -> None:
#CLI: python -m src.feature_eng.run --k 5 --test-ratio 0.2"""
    parser = argparse.ArgumentParser(description="Stage de feature engineering")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    run(args.processed_dir, args.out_dir, args.k, args.test_ratio)


if __name__ == "__main__":
    main()