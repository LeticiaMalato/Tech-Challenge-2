"""Configurações centrais da aplicação, carregadas de variáveis de ambiente."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações do pipeline, com defaults sensatos para dev local.

    Os valores podem ser sobrescritos via arquivo .env ou variáveis
    de ambiente do sistema, sem necessidade de alterar código-fonte.
    """

    # Pré-processamento
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))

    # Treino
    train_path: Path = Field(default=Path("data/features/train.parquet"))
    test_path: Path = Field(default=Path("data/features/test.parquet"))
    checkpoint_dir: Path = Field(default=Path("models/checkpoints/mlp"))

    # MLflow
    mlflow_tracking_uri: str = Field(default="sqlite:///mlflow.db")
    mlflow_experiment: str = Field(default="retailrocket-recommender")
    model_registry_name: str = Field(default="retailrocket-recommender")

    # Reprodutibilidade
    random_seed: int = Field(default=42)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
