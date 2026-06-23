"""Script de avaliação: baselines vs. MLP neural (Matrix Factorization).

Cada modelo é treinado e avaliado dentro de um run do MLflow, registrando
hiperparâmetros, métricas finais (e métricas por epoch, no caso do MLP)
para rastreabilidade de experimentos. Todos os modelos — baselines e MLP —
são treinados sobre exatamente o mesmo conjunto de treino filtrado
(`train_full`) e avaliados sobre o mesmo conjunto de teste (`test_eval`),
garantindo uma comparação controlada entre arquiteturas.
"""

import logging
from pathlib import Path

import mlflow
import pandas as pd

from src.models.baseline.factory import build_recommender, list_available_recommenders
from src.models.baseline.metrics import compare_models_metrics, evaluate_recommender
from src.models.neural.recommender import MLPConfig
from src.utils.seed import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_TRAIN_PATH = Path("data/features/train.parquet")
_TEST_PATH = Path("data/features/test.parquet")
_K_VALUES = [5, 10, 20]
_MIN_INTERACTIONS = 3
_CHECKPOINT_DIR = Path("models/checkpoints/mlp")
_MLFLOW_EXPERIMENT = "retailrocket-recommender"
_METRIC_NAMES = ("hit_rate", "precision", "recall", "ndcg", "mrr")

# Configuração do MLP, validada empiricamente via varredura de
# experimentos sobre o mesmo conjunto de treino filtrado (min_interactions=3)
# usado por todos os baselines — garantindo que o ganho reportado reflita
# diferença de modelo, não diferença de dado de treino.
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
    test_path: Path = _TEST_PATH,
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
# Preparação de dados — compartilhada entre TODOS os modelos
# ---------------------------------------------------------------------------


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


def _filter_test_to_known_users(
    test: pd.DataFrame, train: pd.DataFrame
) -> pd.DataFrame:
    """Filtra o teste para usuários presentes no treino fornecido.

    Aplicada uma única vez sobre ``train_full`` e reutilizada por todos os
    modelos, garante que baselines e MLP sejam avaliados sobre exatamente
    o mesmo conjunto de usuários — condição necessária para que a
    comparação de métricas entre modelos seja válida.

    Args:
        test: DataFrame de teste completo.
        train: DataFrame de treino (já filtrado) usado como referência.

    Returns:
        DataFrame de teste filtrado.
    """
    known_users = set(train["visitorid"].unique())
    filtered = test[test["visitorid"].isin(known_users)].copy()
    logger.info(
        "Teste filtrado por cobertura do treino: %d → %d usuários | %d → %d interações.",
        test["visitorid"].nunique(),
        filtered["visitorid"].nunique(),
        len(test),
        len(filtered),
    )
    return filtered


def _sample_test_users(test: pd.DataFrame, max_test_users: int) -> pd.DataFrame:
    """Amostra um subconjunto de usuários do conjunto de teste.

    Args:
        test: DataFrame de teste completo.
        max_test_users: Número máximo de usuários a amostrar.

    Returns:
        DataFrame filtrado com os usuários amostrados.
    """
    n = min(max_test_users, test["visitorid"].nunique())
    sample_users = test["visitorid"].drop_duplicates().sample(n=n, random_state=42)
    return test[test["visitorid"].isin(sample_users)]


# ---------------------------------------------------------------------------
# Apresentação e tracking de resultados
# ---------------------------------------------------------------------------


def _log_results(name: str, results: pd.DataFrame) -> None:
    """Loga os resultados de avaliação formatados no console.

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


def _log_metrics_to_mlflow(results: pd.DataFrame) -> None:
    """Registra as métricas de avaliação no run ativo do MLflow.

    Convenção de nome: ``{metrica}_at_{k}`` (ex: ``hit_rate_at_10``).
    O MLflow não aceita o caractere ``@`` em nomes de métricas.

    Args:
        results: DataFrame com colunas ``k`` e as métricas em ``_METRIC_NAMES``.
    """
    for _, row in results.iterrows():
        k = int(row["k"])
        for metric in _METRIC_NAMES:
            mlflow.log_metric(f"{metric}_at_{k}", float(row[metric]))


def _results_to_records(name: str, results: pd.DataFrame) -> list[dict]:
    """Converte o DataFrame de métricas de um modelo em registros nomeados.

    Args:
        name: Nome do modelo avaliado.
        results: DataFrame de métricas retornado por ``evaluate_recommender``.

    Returns:
        Lista de dicionários com a chave ``model`` adicionada a cada linha.
    """
    records = results.to_dict(orient="records")
    for record in records:
        record["model"] = name
    return records


# ---------------------------------------------------------------------------
# Execução de baselines
# ---------------------------------------------------------------------------


def run_baseline(
    name: str,
    train_full: pd.DataFrame,
    test_eval: pd.DataFrame,
    k_values: list[int],
    max_test_users: int | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """Treina e avalia um modelo baseline dentro de um run do MLflow.

    Recebe ``train_full``/``test_eval`` já preparados em ``main()`` — os
    mesmos conjuntos usados pelo MLP — garantindo comparação justa entre
    modelos.

    Args:
        name: Chave do recomendador na factory (ex: ``"svd"``).
        train_full: DataFrame de treino (já filtrado e compartilhado entre
            todos os modelos) com colunas ``visitorid``, ``itemid``, ``weight``.
        test_eval: DataFrame de teste (já filtrado por cobertura do treino)
            com colunas ``visitorid`` e ``itemid``.
        k_values: Lista de cortes k para avaliação.
        max_test_users: Se informado, avalia apenas uma amostra de usuários
            (sobre ``test_eval``, não sobre o teste bruto).
        **kwargs: Hiperparâmetros repassados ao construtor do recomendador.

    Returns:
        DataFrame de métricas por valor de k.
    """
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_type", name)
        mlflow.log_params(kwargs)
        mlflow.log_param("n_train_interactions", len(train_full))
        mlflow.log_param("n_train_users", train_full["visitorid"].nunique())
        if max_test_users is not None:
            mlflow.log_param("max_test_users", max_test_users)

        logger.info("Treinando %s | params=%s.", name, kwargs)
        recommender = build_recommender(name, **kwargs)
        recommender.fit(train_full)

        eval_set = test_eval
        if max_test_users is not None:
            eval_set = _sample_test_users(test_eval, max_test_users)
            logger.info(
                "Avaliando %s sobre %d usuários (amostra).",
                name,
                eval_set["visitorid"].nunique(),
            )

        results = evaluate_recommender(recommender, eval_set, k_values=k_values)
        _log_results(name, results)
        _log_metrics_to_mlflow(results)
        return results


# ---------------------------------------------------------------------------
# Execução do modelo neural
# ---------------------------------------------------------------------------


def run_mlp(
    train_full: pd.DataFrame,
    test_eval: pd.DataFrame,
    k_values: list[int],
) -> pd.DataFrame | None:
    """Treina e avalia o MLP dentro de um run do MLflow.

    Recebe ``train_full``/``test_eval`` já preparados em ``main()`` — os
    mesmos conjuntos usados pelos baselines — garantindo comparação justa
    entre arquiteturas. Registra métricas de loss por epoch além das
    métricas finais de ranking.

    Args:
        train_full: DataFrame de treino (já filtrado e compartilhado entre
            todos os modelos).
        test_eval: DataFrame de teste (já filtrado por cobertura do treino).
        k_values: Lista de cortes k para avaliação.

    Returns:
        DataFrame de métricas por valor de k, ou ``None`` se ``test_eval``
        estiver vazio.
    """
    if test_eval.empty:
        logger.warning("test_eval vazio. Abortando avaliação do MLP.")
        return None

    with mlflow.start_run(run_name="mlp"):
        mlflow.log_param("model_type", "mlp")
        mlflow.log_params(vars(MLP_CONFIG))
        mlflow.log_param("n_train_interactions", len(train_full))
        mlflow.log_param("n_train_users", train_full["visitorid"].nunique())
        mlflow.log_param("n_train_items", train_full["itemid"].nunique())

        logger.info(
            "Treinando MLP | interações=%d | n_users=%d | n_items=%d",
            len(train_full),
            train_full["visitorid"].nunique(),
            train_full["itemid"].nunique(),
        )

        def log_epoch(epoch: int, train_loss: float, val_loss: float) -> None:
            """Registra as losses da epoch no run ativo do MLflow."""
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        recommender = build_recommender(
            "mlp", config=MLP_CONFIG, checkpoint_dir=_CHECKPOINT_DIR
        )
        recommender.fit(train_full, epoch_callback=log_epoch)

        results = evaluate_recommender(recommender, test_eval, k_values=k_values)
        _log_results("mlp", results)
        _log_metrics_to_mlflow(results)
        mlflow.log_artifact(str(recommender.checkpoint_path))
        return results


def main() -> None:
    """Avalia todos os baselines e o MLP, rastreando tudo no MLflow.

    Prepara ``train_full``/``test_eval`` uma única vez e os reutiliza em
    todos os modelos (baselines + MLP), garantindo que diferenças nas
    métricas finais reflitam diferença de arquitetura, não de dado.
    """
    set_global_seed(42)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)

    train, test = load_data()

    train_full = _build_train_with_full_coverage(train, test, _MIN_INTERACTIONS)
    test_eval = _filter_test_to_known_users(test, train_full)

    k_values = _K_VALUES

    baselines: list[tuple[str, dict, int | None]] = [
        ("popularity", {}, None),
        ("item_knn", {"top_n_neighbors": 20, "max_users": 5_000}, None),
        ("svd", {"n_components": 50, "seed": 42}, None),
        ("logistic", {"neg_ratio": 3, "seed": 42, "max_positives": 20_000}, None),
    ]

    logger.info(
        "Modelos disponíveis: %s. Avaliando %d baselines.",
        list_available_recommenders(),
        len(baselines),
    )

    all_results: list[dict] = []

    for name, params, max_test_users in baselines:
        baseline_results = run_baseline(
            name,
            train_full,
            test_eval,
            k_values,
            max_test_users=max_test_users,
            **params,
        )
        all_results.extend(_results_to_records(name, baseline_results))

    mlp_results = run_mlp(train_full, test_eval, k_values)
    if mlp_results is not None:
        all_results.extend(_results_to_records("mlp", mlp_results))

    compare_models_metrics(all_results)

    logger.info(
        "Avaliação concluída. Execute 'mlflow ui' para visualizar os experimentos."
    )


if __name__ == "__main__":
    main()
