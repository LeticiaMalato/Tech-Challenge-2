"""Testes unitários para as factories de recomendadores e pré-processadores."""

import pytest

from src.models.baseline.factory import build_recommender, list_available_recommenders
from src.models.baseline.item_knn_recommender import ItemKNNRecommender
from src.models.baseline.logistic_recommender import LogisticRecommender
from src.models.baseline.popularity_recommender import PopularityRecommender
from src.models.baseline.svd_recommender import SVDRecommender
from src.models.neural.recommender import MLPRecommender
from src.preprocessing.categories import CategoriesPreprocessor
from src.preprocessing.events import EventsPreprocessor
from src.preprocessing.factory import build_preprocessor
from src.preprocessing.item_properties import ItemPropertiesPreprocessor

# Recommender factory


def test_build_recommender_instancia_todos_os_tipos_registrados() -> None:
    """Cada nome registrado deve instanciar a classe concreta correspondente."""
    assert isinstance(build_recommender("popularity"), PopularityRecommender)
    assert isinstance(build_recommender("item_knn"), ItemKNNRecommender)
    assert isinstance(build_recommender("svd"), SVDRecommender)
    assert isinstance(build_recommender("logistic"), LogisticRecommender)
    assert isinstance(build_recommender("mlp"), MLPRecommender)


def test_build_recommender_repassa_hiperparametros() -> None:
    """Kwargs devem ser repassados ao construtor da classe concreta."""
    rec = build_recommender("svd", n_components=5, seed=7)
    assert rec._svd.n_components == 5


def test_build_recommender_nome_desconhecido_levanta_key_error() -> None:
    """Nome fora do registro deve levantar KeyError com mensagem informativa."""
    with pytest.raises(KeyError):
        build_recommender("modelo_inexistente")


def test_list_available_recommenders_retorna_todos_os_nomes() -> None:
    """Deve listar exatamente os 5 modelos registrados."""
    names = list_available_recommenders()
    assert set(names) == {"popularity", "item_knn", "svd", "logistic", "mlp"}


# Preprocessor factory


def test_build_preprocessor_instancia_tipos_corretos() -> None:
    """Cada fonte registrada deve instanciar o preprocessor correspondente."""
    assert isinstance(build_preprocessor("events"), EventsPreprocessor)
    assert isinstance(build_preprocessor("categories"), CategoriesPreprocessor)
    assert isinstance(build_preprocessor("item_properties"), ItemPropertiesPreprocessor)


def test_build_preprocessor_fonte_desconhecida_levanta_key_error() -> None:
    """Fonte fora do registro deve levantar KeyError."""
    with pytest.raises(KeyError):
        build_preprocessor("fonte_inexistente")
