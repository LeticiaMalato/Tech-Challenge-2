"""Script de avaliação dos modelos baseline e MLP neural."""

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


def _sample_train_covering_test_users(
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_interactions: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Amostra o treino garantindo cobertura total dos usuários do teste.

    Estratégia:
    1. Inclui TODAS as interações dos usuários presentes no teste.
    2. Completa até ``max_interactions`` com amostra aleatória dos demais.

    Args:
        train: DataFrame de treino completo.
        test: DataFrame de teste (usado apenas para extrair visitorids).
        max_interactions: Limite máximo de interações no treino amostrado.
        seed: Semente para reprodutibilidade da amostragem.

    Returns:
        DataFrame de treino amostrado com cobertura dos usuários do teste.
    """
    test_users = set(test["visitorid"].unique())
    test_user_interactions = train[train["visitorid"].isin(test_users)]

    logger.info(
        "Usuários do teste no treino: %d | Interações: %d",
        test_user_interactions["visitorid"].nunique(),
        len(test_user_interactions),
    )

    remaining_budget = max_interactions - len(test_user_interactions)

    if remaining_budget > 0:
        other_interactions = train[~train["visitorid"].isin(test_users)]
        n_sample = min(remaining_budget, len(other_interactions))
        sampled_others = other_interactions.sample(n=n_sample, random_state=seed)
        result = pd.concat([test_user_interactions, sampled_others], ignore_index=True)
    else:
        result = test_user_interactions.copy()
        logger.warning(
            "Interações dos usuários do teste (%d) excedem max_interactions (%d). "
            "Usando apenas essas.",
            len(test_user_interactions),
            max_interactions,
        )

    if "timestamp" in result.columns:
        result = result.sort_values("timestamp").reset_index(drop=True)
    else:
        result = result.reset_index(drop=True)

    logger.info(
        "Treino final: %d interações | %d usuários únicos.",
        len(result),
        result["visitorid"].nunique(),
    )
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
    config: MLPConfig | None = None,
    max_train_interactions: int = 150_000,
    max_test_users: int | None = None,
    min_interactions: int = 3,
    run_name: str = "mlp",
) -> pd.DataFrame | None:
    """Treina e avalia o modelo neural (Matrix Factorization), logando os resultados.

    Garante que todos os usuários presentes no teste estejam no treino,
    independentemente do filtro ``min_interactions``.

    Args:
        train: DataFrame de treino completo.
        test: DataFrame de teste com colunas ``visitorid`` e ``itemid``.
        k_values: Lista de cortes k para avaliação.
        config: Configuração do ``MLPRecommender``. Usa defaults se ``None``.
        max_train_interactions: Limite de interações na amostra de treino.
        max_test_users: Se informado, avalia apenas uma amostra de usuários.
        min_interactions: Mínimo de interações por usuário no treino filtrado.
            Usuários do teste são sempre preservados independentemente deste valor.
        run_name: Identificador do experimento, usado no log e no checkpoint.

    Returns:
        DataFrame de métricas por valor de k, ou ``None`` se nenhum
        usuário do teste sobreviveu ao filtro.
    """
    test_users = set(test["visitorid"].unique())

    counts = train.groupby("visitorid").size()
    users_to_keep = set(counts[counts >= min_interactions].index) | test_users
    train_filtered = train[train["visitorid"].isin(users_to_keep)]

    logger.info(
        "[%s] Filtro min_interactions=%d (preservando usuários do teste): "
        "%d → %d usuários | %d → %d interações.",
        run_name, min_interactions,
        train["visitorid"].nunique(),
        train_filtered["visitorid"].nunique(),
        len(train),
        len(train_filtered),
    )

    train_sample = _sample_train_covering_test_users(
        train_filtered, test, max_train_interactions
    )

    logger.info(
        "[%s] Iniciando treino MLP | interações=%d | n_users=%d | n_items=%d",
        run_name,
        len(train_sample),
        train_sample["visitorid"].nunique(),
        train_sample["itemid"].nunique(),
    )

    mlp_config = config or MLPConfig(
        embed_dim=32,
        neg_ratio=4,
        lr=5e-4,
        weight_decay=1e-2,
        batch_size=2048,
        max_epochs=50,
        patience=10,
        seed=42,
    )

    checkpoint_dir = Path("models/checkpoints") / run_name
    recommender = MLPRecommender(config=mlp_config, checkpoint_dir=checkpoint_dir)
    recommender.fit(train_sample)

    train_users = set(train_sample["visitorid"].unique())
    test_eval = test[test["visitorid"].isin(train_users)].copy()

    logger.info(
        "[%s] Usuários do teste com embedding treinado: %d / %d.",
        run_name,
        test_eval["visitorid"].nunique(),
        test["visitorid"].nunique(),
    )

    if max_test_users is not None:
        test_eval = _sample_test_users(test_eval, max_test_users)
        logger.info(
            "[%s] Avaliando MLP sobre %d usuários (amostra).",
            run_name,
            test_eval["visitorid"].nunique(),
        )

    if test_eval.empty:
        logger.warning(
            "[%s] Nenhum usuário do teste sobreviveu ao filtro. Abortando avaliação.",
            run_name,
        )
        return None

    results = evaluate_recommender(recommender, test_eval, k_values=k_values)
    _log_results(run_name, results)
    return results


# ---------------------------------------------------------------------------
# Varredura de experimentos do MLP
# ---------------------------------------------------------------------------

# Cada experimento sobrescreve só os campos relevantes em relação ao
# baseline (run_001), permitindo isolar o efeito de cada hiperparâmetro.
_MLP_EXPERIMENTS: list[dict] = [
    {
        "run_name": "mlp_baseline_150k",
        "max_train_interactions": 150_000,
        "config_overrides": {},
    },
    {
        "run_name": "mlp_more_data_400k",
        "max_train_interactions": 400_000,
        "config_overrides": {},
    },
    {
        "run_name": "mlp_more_data_full",
        "max_train_interactions": 700_000,  # acima do total disponível, será truncado
        "config_overrides": {},
    },
    {
        "run_name": "mlp_lower_embed_dim",
        "max_train_interactions": 150_000,
        "config_overrides": {"embed_dim": 16},
    },
    {
        "run_name": "mlp_higher_weight_decay",
        "max_train_interactions": 150_000,
        "config_overrides": {"weight_decay": 5e-2},
    },
    {
        "run_name": "mlp_lower_embed_higher_decay",
        "max_train_interactions": 400_000,
        "config_overrides": {"embed_dim": 16, "weight_decay": 5e-2},
    },
]

_BASE_MLP_KWARGS: dict = dict(
    embed_dim=32,
    neg_ratio=4,
    lr=5e-4,
    weight_decay=1e-2,
    batch_size=2048,
    max_epochs=50,
    patience=10,
    seed=42,
)


def run_mlp_sweep(
    train: pd.DataFrame,
    test: pd.DataFrame,
    k_values: list[int],
    experiments: list[dict] | None = None,
) -> pd.DataFrame:
    """Executa uma varredura de experimentos do MLP e consolida os resultados.

    Cada experimento define ``max_train_interactions`` e um dicionário de
    ``config_overrides`` aplicado sobre os hiperparâmetros base, permitindo
    isolar o efeito de cada mudança (mais dados, embedding menor, etc.).

    Args:
        train: DataFrame de treino completo.
        test: DataFrame de teste.
        k_values: Lista de cortes k para avaliação.
        experiments: Lista de specs de experimento. Usa ``_MLP_EXPERIMENTS``
            se ``None``.

    Returns:
        DataFrame consolidado com uma linha por (experimento, k), incluindo
        a coluna ``run_name`` para identificar cada configuração.
    """
    experiments = experiments or _MLP_EXPERIMENTS
    all_results: list[pd.DataFrame] = []

    for spec in experiments:
        run_name = spec["run_name"]
        cfg_kwargs = {**_BASE_MLP_KWARGS, **spec["config_overrides"]}
        config = MLPConfig(**cfg_kwargs)

        logger.info(
            "=== Experimento [%s] | max_train_interactions=%d | overrides=%s ===",
            run_name, spec["max_train_interactions"], spec["config_overrides"],
        )

        results = run_mlp(
            train=train,
            test=test,
            k_values=k_values,
            config=config,
            max_train_interactions=spec["max_train_interactions"],
            max_test_users=None,
            min_interactions=3,
            run_name=run_name,
        )

        if results is not None:
            results = results.copy()
            results.insert(0, "run_name", run_name)
            all_results.append(results)

    if not all_results:
        logger.warning("Nenhum experimento produziu resultados.")
        return pd.DataFrame()

    consolidated = pd.concat(all_results, ignore_index=True)
    return consolidated


def _log_sweep_summary(consolidated: pd.DataFrame, k_focus: int = 20) -> None:
    """Loga um resumo comparativo da varredura, ordenado por hit_rate.

    Args:
        consolidated: DataFrame retornado por ``run_mlp_sweep``.
        k_focus: Valor de k usado para ordenar o ranking de comparação.
    """
    if consolidated.empty:
        return
    subset = consolidated[consolidated["k"] == k_focus].sort_values(
        "hit_rate", ascending=False
    )
    logger.info(
        "\n%s\n  RESUMO DA VARREDURA (k=%d, ordenado por hit_rate)\n%s\n%s",
        "=" * 70, k_focus, "=" * 70,
        subset.to_string(index=False),
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Avalia todos os baselines e executa a varredura de experimentos do MLP."""
    set_global_seed(42)

    train, test = load_data()
    k_values = _K_VALUES

    baselines: list[tuple[str, dict, int | None]] = [
        ("popularity", {},                                                                      None),
        ("item_knn",   {"top_n_neighbors": 20, "max_users": 5_000},                            None),
        ("svd",        {"n_components": 50, "seed": 42},                                       None),
        ("logistic",   {"n_components": 32, "neg_ratio": 3, "seed": 42, "max_positives": 20_000}, None),
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

    logger.info("Iniciando varredura de experimentos do MLP (%d configs).", len(_MLP_EXPERIMENTS))
    consolidated = run_mlp_sweep(train, test, k_values)
    _log_sweep_summary(consolidated, k_focus=20)

    logger.info("Avaliação concluída.")


if __name__ == "__main__":
    main()