"""Script de avaliação: baselines vs. MLP neural (Matrix Factorization)."""

import logging
from pathlib import Path

import pandas as pd

from src.models.baseline.factory import build_recommender, list_available_recommenders
from src.models.baseline.metrics import evaluate_recommender
from src.models.neural.recommender import MLPConfig, MLPRecommender
from src.utils.seed import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_TRAIN_PATH = Path("data/features/train.parquet")
_TEST_PATH  = Path("data/features/test.parquet")
_K_VALUES   = [5, 10, 20]
_CHECKPOINT_DIR = Path("models/checkpoints/mlp")

# Configuração do MLP, validada empiricamente via varredura de
# experimentos: treinar com 100% dos dados disponíveis supera, por margem
# ampla, as variações de embed_dim/weight_decay testadas sobre amostras
# reduzidas. hit_rate@20 = 0.0665 vs. 0.0343 do baseline com 150k interações
# (+94%), superando também todos os baselines clássicos do projeto.
MLP_CONFIG = MLPConfig(
    embed_dim=32,
    neg_ratio=4,
    lr=5e-4,
    weight_decay=1e-2,
    batch_size=2048,
    max_epochs=50,
    patience=10,
    seed=42,
)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_data(
    train_path: Path = _TRAIN_PATH,
    test_path: Path  = _TEST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os conjuntos de treino e teste em formato Parquet.

    Args:
        train_path: Caminho para o arquivo ``train.parquet``.
        test_path: Caminho para o arquivo ``test.parquet``.

    Returns:
        Tupla (train, test) como DataFrames.

    Raises:
        FileNotFoundError: Se algum dos arquivos não existir.
    """
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_parquet(train_path), pd.read_parquet(test_path)


# ---------------------------------------------------------------------------
# Amostragem
# ---------------------------------------------------------------------------

def _sample_test_users(test: pd.DataFrame, max_test_users: int) -> pd.DataFrame:
    """Amostra um subconjunto de usuários do conjunto de teste.

    Args:
        test: DataFrame de teste completo.
        max_test_users: Número máximo de usuários a amostrar.

    Returns:
        DataFrame filtrado com os usuários amostrados.
    """
    n = min(max_test_users, test["visitorid"].nunique())
    sample_users = (
        test["visitorid"]
        .drop_duplicates()
        .sample(n=n, random_state=42)
    )
    return test[test["visitorid"].isin(sample_users)]


def _build_train_with_full_coverage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    min_interactions: int,
) -> pd.DataFrame:
    """Filtra o treino por atividade mínima, preservando usuários do teste.

    Aplica o filtro ``min_interactions`` para remover usuários pouco ativos
    do treino, mas garante que todo usuário presente no teste permaneça,
    mesmo abaixo do limiar — evita cold-start na avaliação.

    Args:
        train: DataFrame de treino completo.
        test: DataFrame de teste, usado apenas para extrair visitorids.
        min_interactions: Mínimo de interações por usuário para permanecer
            no treino (não aplicado aos usuários do teste).

    Returns:
        DataFrame de treino filtrado, ordenado por timestamp se disponível.
    """
    test_users = set(test["visitorid"].unique())
    counts = train.groupby("visitorid").size()
    users_to_keep = set(counts[counts >= min_interactions].index) | test_users
    result = train[train["visitorid"].isin(users_to_keep)]

    logger.info(
        "Filtro min_interactions=%d (preservando usuários do teste): "
        "%d → %d usuários | %d → %d interações.",
        min_interactions,
        train["visitorid"].nunique(),
        result["visitorid"].nunique(),
        len(train),
        len(result),
    )

    if "timestamp" in result.columns:
        result = result.sort_values("timestamp").reset_index(drop=True)
    else:
        result = result.reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Apresentação de resultados
# ---------------------------------------------------------------------------

def _log_results(name: str, results: pd.DataFrame) -> None:
    """Loga os resultados de avaliação formatados.

    Args:
        name: Nome do modelo avaliado.
        results: DataFrame com métricas por valor de k.
    """
    separator = "=" * 60
    logger.info(
        "\n%s\n  %s\n%s\n%s",
        separator,
        name.upper(),
        separator,
        results.to_string(index=False),
    )


# ---------------------------------------------------------------------------
# Execução de baselines
# ---------------------------------------------------------------------------

def run_baseline(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k_values: list[int],
    max_test_users: int | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """Treina e avalia um modelo baseline, logando e retornando os resultados.

    Args:
        name: Chave do recomendador na factory (ex: ``"svd"``).
        train: DataFrame de treino com colunas ``visitorid``, ``itemid``, ``weight``.
        test: DataFrame de teste com colunas ``visitorid`` e ``itemid``.
        k_values: Lista de cortes k para avaliação.
        max_test_users: Se informado, avalia apenas uma amostra de usuários.
        **kwargs: Hiperparâmetros repassados ao construtor do recomendador.

    Returns:
        DataFrame de métricas por valor de k.
    """
    logger.info("Treinando %s | params=%s.", name, kwargs)
    recommender = build_recommender(name, **kwargs)
    recommender.fit(train)

    test_eval = test
    if max_test_users is not None:
        test_eval = _sample_test_users(test, max_test_users)
        logger.info(
            "Avaliando %s sobre %d usuários (amostra).",
            name,
            test_eval["visitorid"].nunique(),
        )

    results = evaluate_recommender(recommender, test_eval, k_values=k_values)
    _log_results(name, results)
    return results


# ---------------------------------------------------------------------------
# Execução do modelo neural
# ---------------------------------------------------------------------------

def run_mlp(
    train: pd.DataFrame,
    test: pd.DataFrame,
    k_values: list[int],
    min_interactions: int = 3,
) -> pd.DataFrame | None:
    """Treina e avalia o MLP com a configuração, usando 100% dos dados.

    Configuração definida empiricamente: treinar sobre o volume completo de
    interações disponíveis (após filtro de atividade mínima) supera por
    margem ampla as variações testadas com amostras reduzidas de dados.

    Args:
        train: DataFrame de treino completo.
        test: DataFrame de teste com colunas ``visitorid`` e ``itemid``.
        k_values: Lista de cortes k para avaliação.
        min_interactions: Mínimo de interações por usuário no treino filtrado.
            Usuários do teste são sempre preservados independentemente deste valor.

    Returns:
        DataFrame de métricas por valor de k, ou ``None`` se nenhum
        usuário do teste sobreviveu ao filtro.
    """
    train_full = _build_train_with_full_coverage(train, test, min_interactions)

    logger.info(
        "Treinando MLP | interações=%d | n_users=%d | n_items=%d",
        len(train_full),
        train_full["visitorid"].nunique(),
        train_full["itemid"].nunique(),
    )

    recommender = MLPRecommender(config=MLP_CONFIG, checkpoint_dir=_CHECKPOINT_DIR)
    recommender.fit(train_full)

    train_users = set(train_full["visitorid"].unique())
    test_eval = test[test["visitorid"].isin(train_users)].copy()

    logger.info(
        "Usuários do teste com embedding treinado: %d / %d.",
        test_eval["visitorid"].nunique(),
        test["visitorid"].nunique(),
    )

    if test_eval.empty:
        logger.warning("Nenhum usuário do teste sobreviveu ao filtro. Abortando avaliação MLP.")
        return None

    results = evaluate_recommender(recommender, test_eval, k_values=k_values)
    _log_results("mlp", results)
    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Avalia todos os baselines e o MLP na configuração escolhida."""
    set_global_seed(42)

    train, test = load_data()
    k_values = _K_VALUES

    baselines: list[tuple[str, dict, int | None]] = [
        ("popularity", {},                                                          None),
        ("item_knn",   {"top_n_neighbors": 20, "max_users": 5_000},                 None),
        ("svd",        {"n_components": 50, "seed": 42},                            None),
        ("logistic",   {"neg_ratio": 3, "seed": 42, "max_positives": 20_000},        None),
    ]

    logger.info(
        "Modelos disponíveis: %s. Avaliando %d baselines.",
        list_available_recommenders(),
        len(baselines),
    )

    for name, params, max_test_users in baselines:
        run_baseline(
            name, train, test, k_values,
            max_test_users=max_test_users,
            **params,
        )

    run_mlp(train, test, k_values, min_interactions=3)

    logger.info("Avaliação concluída.")


if __name__ == "__main__":
    main()