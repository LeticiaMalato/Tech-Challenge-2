from src.models.baseline.als_recommender import ALSRecommender
from src.models.baseline.base import Recommender
from src.models.baseline.bpr_mf_recommender import BPRMFRecommender
from src.models.baseline.item_knn_recommender import ItemKNNRecommender
from src.models.baseline.popularity_recommender import PopularityRecommender

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