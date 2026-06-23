"""Pipeline de treino e avaliação: baselines vs. MLP, com tracking e registry no MLflow.

Cada modelo é treinado e avaliado dentro de um run do MLflow, registrando
hiperparâmetros, métricas finais (e métricas por epoch, no caso do MLP)
para rastreabilidade de experimentos. Todos os modelos — baselines e MLP —
são treinados sobre exatamente o mesmo conjunto de treino filtrado
(`train_full`) e avaliados sobre o mesmo conjunto de teste (`test_eval`),
garantindo uma comparação controlada entre arquiteturas.

Ao final, o modelo com melhor NDCG@10 entre todos os candidatos é
registrado no MLflow Model Registry e promovido a Production.
"""

import logging
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient

from src.models.baseline.base import Recommender
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
_SELECTION_K = 10  # critério de seleção do melhor modelo para o Registry
_MIN_INTERACTIONS = 3
_CHECKPOINT_DIR = Path("models/checkpoints/mlp")
_MLFLOW_EXPERIMENT = "retailrocket-recommender"
_MODEL_REGISTRY_NAME = "retailrocket-recommender"
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


# I/O


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


# Preparação de dados — compartilhada entre TODOS os modelos


def _build_train_with_full_coverage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    min_interactions: int,
) -> pd.DataFrame:
    """Filtra o treino por atividade mínima, preservando usuários do teste.

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


# Wrapper genérico para registro de qualquer Recommender no MLflow


class RecommenderPyfuncWrapper(mlflow.pyfunc.PythonModel):
    """Adapta qualquer ``Recommender`` para o formato pyfunc do MLflow.

    Funciona uniformemente para baselines (sklearn/numpy puro) e para o
    MLP (PyTorch), já que todas as implementações compartilham a mesma
    interface ``recommend(user_id, k) -> list[int]`` definida em
    ``Recommender``. Isso evita duplicar lógica de logging por tipo de
    modelo (ex: ``mlflow.sklearn`` vs ``mlflow.pytorch``).
    """

    def __init__(self, recommender: Recommender) -> None:
        """Armazena o recomendador já treinado a ser empacotado.

        Args:
            recommender: Instância treinada (``fit`` já chamado).
        """
        self._recommender = recommender

    def predict(
        self,
        context: object,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> list[list[int]]:
        """Gera recomendações para cada par (visitorid, k) na entrada.

        Args:
            context: Contexto do pyfunc (não utilizado).
            model_input: DataFrame com colunas ``visitorid`` e ``k``.
            params: Não utilizado; presente para compatibilidade com a
                assinatura esperada pelo MLflow.

        Returns:
            Lista de listas de item_ids recomendados, uma por linha de
            ``model_input``.
        """
        return [
            self._recommender.recommend(int(row["visitorid"]), int(row["k"]))
            for _, row in model_input.iterrows()
        ]


def _log_model(recommender: Recommender, artifact_path: str = "model") -> None:
    """Loga o recomendador treinado no run ativo via wrapper pyfunc.

    Args:
        recommender: Instância já treinada (``fit`` chamado).
        artifact_path: Subcaminho do artefato dentro do run.
    """
    sample_input = pd.DataFrame({"visitorid": [0], "k": [10]})
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=RecommenderPyfuncWrapper(recommender),
        input_example=sample_input,
    )


# Apresentação e tracking de resultados


def _log_results(name: str, results: pd.DataFrame) -> None:
    """Imprime os resultados de avaliação formatados no console.

    Usa print() em vez de logger.info() porque o MLflow, ao inicializar
    o backend SQLite via Alembic, desabilita loggers pré-existentes
    (fileConfig com disable_existing_loggers=True) — o que silenciaria
    esta saída se dependesse do logger da aplicação.

    Args:
        name: Nome do modelo avaliado.
        results: DataFrame com métricas por valor de k.
    """
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"  {name.upper()}")
    print(separator)
    print(results.to_string(index=False))


def _log_metrics_to_mlflow(results: pd.DataFrame) -> None:
    """Registra as métricas de avaliação no run ativo do MLflow.

    Convenção de nome: ``{metrica}_at_{k}`` (ex: ``hit_rate_at_10``).

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


def _ndcg_at_selection_k(results: pd.DataFrame, k: int = _SELECTION_K) -> float:
    """Extrai o NDCG no valor de k usado como critério de seleção.

    Args:
        results: DataFrame de métricas retornado por ``evaluate_recommender``.
        k: Valor de k usado para comparar modelos entre si.

    Returns:
        NDCG no valor de k informado.

    Raises:
        ValueError: Se ``k`` não estiver presente em ``results``.
    """
    row = results[results["k"] == k]
    if row.empty:
        raise ValueError(
            f"k={k} não encontrado nos resultados: {results['k'].tolist()}"
        )
    return float(row["ndcg"].iloc[0])


# Execução de baselines


def run_baseline(
    name: str,
    train_full: pd.DataFrame,
    test_eval: pd.DataFrame,
    k_values: list[int],
    max_test_users: int | None = None,
    **kwargs: object,
) -> tuple[pd.DataFrame, str]:
    """Treina e avalia um modelo baseline dentro de um run do MLflow.

    Args:
        name: Chave do recomendador na factory (ex: ``"svd"``).
        train_full: DataFrame de treino (filtrado e compartilhado entre
            todos os modelos) com colunas ``visitorid``, ``itemid``, ``weight``.
        test_eval: DataFrame de teste (filtrado por cobertura do treino)
            com colunas ``visitorid`` e ``itemid``.
        k_values: Lista de cortes k para avaliação.
        max_test_users: Se informado, avalia apenas uma amostra de usuários.
        **kwargs: Hiperparâmetros repassados ao construtor do recomendador.

    Returns:
        Tupla (DataFrame de métricas por k, run_id do MLflow).
    """
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type", name)
        mlflow.log_params(kwargs)
        mlflow.log_param("n_train_interactions", len(train_full))
        mlflow.log_param("n_train_users", train_full["visitorid"].nunique())
        if max_test_users is not None:
            mlflow.log_param("max_test_users", max_test_users)

        logger.info("Treinando %s | params=%s.", name, kwargs)
        recommender = build_recommender(name, **kwargs)
        recommender.fit(train_full)
        _log_model(recommender)

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
        return results, run.info.run_id


# Execução do modelo neural


def run_mlp(
    train_full: pd.DataFrame,
    test_eval: pd.DataFrame,
    k_values: list[int],
) -> tuple[pd.DataFrame, str] | None:
    """Treina e avalia o MLP dentro de um run do MLflow.

    Args:
        train_full: DataFrame de treino (filtrado e compartilhado entre
            todos os modelos).
        test_eval: DataFrame de teste (filtrado por cobertura do treino).
        k_values: Lista de cortes k para avaliação.

    Returns:
        Tupla (DataFrame de métricas por k, run_id do MLflow), ou ``None``
        se ``test_eval`` estiver vazio.
    """
    if test_eval.empty:
        logger.warning("test_eval vazio. Abortando avaliação do MLP.")
        return None

    with mlflow.start_run(run_name="mlp") as run:
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
            """Registra as losses da epoch no MLflow e imprime no terminal."""
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            print(
                f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
            )

        recommender = build_recommender(
            "mlp", config=MLP_CONFIG, checkpoint_dir=_CHECKPOINT_DIR
        )
        recommender.fit(train_full, epoch_callback=log_epoch)
        _log_model(recommender)
        mlflow.log_artifact(str(recommender.checkpoint_path))

        results = evaluate_recommender(recommender, test_eval, k_values=k_values)
        _log_results("mlp", results)
        _log_metrics_to_mlflow(results)
        return results, run.info.run_id


# Model Registry


def register_best_model(
    run_id: str,
    model_name: str = _MODEL_REGISTRY_NAME,
    artifact_path: str = "model",
) -> str:
    """Registra o modelo de um run no Model Registry e o promove a Production.

    Transição via Staging → Production usando a API clássica de stages.
    Nota: stages estão deprecados desde o MLflow 2.9 em favor de aliases
    (``set_registered_model_alias``), mas permanecem funcionais — usados
    aqui por corresponder à nomenclatura pedida no desafio.

    Args:
        run_id: ID do run do MLflow que produziu o melhor modelo (por NDCG).
        model_name: Nome a registrar no Model Registry.
        artifact_path: Subcaminho do artefato do modelo dentro do run.

    Returns:
        Número da versão registrada no Model Registry.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging",
    )
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info(
        "Modelo '%s' v%s promovido a Production (run_id=%s).",
        model_name,
        result.version,
        run_id,
    )
    return result.version


def main() -> None:
    """Avalia todos os baselines e o MLP, registrando o melhor no Model Registry.

    Prepara ``train_full``/``test_eval`` uma única vez e os reutiliza em
    todos os modelos. Ao final, seleciona o modelo com maior NDCG@10 entre
    todos os candidatos e o registra/promove no MLflow Model Registry.
    """
    set_global_seed(42)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)

    # Reforça o logger após a inicialização do MLflow/Alembic: a primeira
    # vez que o backend SQLite é usado, o Alembic carrega sua própria
    # configuração de logging com disable_existing_loggers=True, o que
    # desabilita silenciosamente loggers já existentes — incluindo este.
    logger.disabled = False
    logger.setLevel(logging.INFO)

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
    candidates: list[tuple[str, str, float]] = []  # (nome, run_id, ndcg@_SELECTION_K)

    for name, params, max_test_users in baselines:
        baseline_results, run_id = run_baseline(
            name,
            train_full,
            test_eval,
            k_values,
            max_test_users=max_test_users,
            **params,
        )
        all_results.extend(_results_to_records(name, baseline_results))
        candidates.append((name, run_id, _ndcg_at_selection_k(baseline_results)))

    mlp_out = run_mlp(train_full, test_eval, k_values)
    if mlp_out is not None:
        mlp_results, mlp_run_id = mlp_out
        all_results.extend(_results_to_records("mlp", mlp_results))
        candidates.append(("mlp", mlp_run_id, _ndcg_at_selection_k(mlp_results)))

    compare_models_metrics(all_results)

    best_name, best_run_id, best_ndcg = max(candidates, key=lambda c: c[2])
    logger.info(
        "Melhor modelo: %s | ndcg@%d=%.4f | run_id=%s",
        best_name,
        _SELECTION_K,
        best_ndcg,
        best_run_id,
    )
    version = register_best_model(run_id=best_run_id, model_name=_MODEL_REGISTRY_NAME)
    logger.info(
        "Registry: '%s' v%s (Production) ← modelo '%s'.",
        _MODEL_REGISTRY_NAME,
        version,
        best_name,
    )

    logger.info(
        "Avaliação concluída. Execute 'mlflow ui' para visualizar os experimentos."
    )


if __name__ == "__main__":
    main()
