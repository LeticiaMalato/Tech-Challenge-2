"""Factory para instanciação de recomendadores pelo nome registrado."""

from src.models.baseline.base import Recommender
from src.models.baseline.item_knn_recommender import ItemKNNRecommender
from src.models.baseline.logistic_recommender import LogisticRecommender
from src.models.baseline.popularity_recommender import PopularityRecommender
from src.models.baseline.svd_recommender import SVDRecommender
from src.models.neural.recommender import MLPRecommender

_REGISTRY: dict[str, type[Recommender]] = {
    "popularity": PopularityRecommender,
    "item_knn":   ItemKNNRecommender,
    "svd":        SVDRecommender,
    "logistic":   LogisticRecommender,
    "mlp":        MLPRecommender,  
}



def build_recommender(name: str, **kwargs: object) -> Recommender:
    """Instancia um recomendador pelo nome registrado.

    Implementa o padrão Factory: o chamador não precisa conhecer a classe
    concreta, apenas o nome e os hiperparâmetros desejados.

    Args:
        name: Chave do recomendador no registro (ex: ``"svd"``).
        **kwargs: Hiperparâmetros repassados ao construtor da classe.

    Returns:
        Instância do recomendador solicitado.

    Raises:
        KeyError: Se ``name`` não estiver no registro.

    Example:
        >>> rec = build_recommender("svd", n_components=64, seed=0)
        >>> rec.fit(train_df)
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"modelo desconhecido: {name!r}. "
            f"Disponíveis: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)


def list_available_recommenders() -> list[str]:
    """Retorna os nomes de todos os recomendadores registrados.

    Returns:
        Lista de strings com as chaves do registro de modelos.
    """
    return list(_REGISTRY.keys())