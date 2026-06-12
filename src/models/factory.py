from src.models.als_recommender import ALSRecommender
from src.models.base import Recommender
from src.models.bpr_mf_recommender import BPRMFRecommender
from src.models.item_knn_recommender import ItemKNNRecommender
from src.models.popularity_recommender import PopularityRecommender

_REGISTRY: dict[str, type[Recommender]] = {
    "popularity":  PopularityRecommender,
    "item_knn":    ItemKNNRecommender,
    "bpr_mf":      BPRMFRecommender,
    "als":         ALSRecommender,
}


def build_recommender(name: str, **kwargs) -> Recommender:

    if name not in _REGISTRY:
        raise KeyError(f"modelo desconhecido: {name!r}")
    return _REGISTRY[name](**kwargs)