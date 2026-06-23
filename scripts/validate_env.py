"""Valida se as variáveis de ambiente necessárias estão configuradas corretamente."""

import sys

from pydantic import ValidationError

from src.config import Settings


def main() -> None:
    """Tenta instanciar Settings e reporta sucesso ou erro de validação."""
    try:
        settings = Settings()
    except ValidationError as exc:
        print("Erro de validação no ambiente:")
        print(exc)
        sys.exit(1)

    print("Ambiente validado com sucesso:")
    print(f"  RAW_DATA_DIR       = {settings.raw_data_dir}")
    print(f"  PROCESSED_DATA_DIR = {settings.processed_data_dir}")
    print(f"  TRAIN_PATH          = {settings.train_path}")
    print(f"  TEST_PATH           = {settings.test_path}")
    print(f"  CHECKPOINT_DIR      = {settings.checkpoint_dir}")
    print(f"  MLFLOW_TRACKING_URI = {settings.mlflow_tracking_uri}")
    print(f"  MLFLOW_EXPERIMENT   = {settings.mlflow_experiment}")
    print(f"  MODEL_REGISTRY_NAME = {settings.model_registry_name}")
    print(f"  RANDOM_SEED         = {settings.random_seed}")


if __name__ == "__main__":
    main()
