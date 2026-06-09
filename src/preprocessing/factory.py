"""Factory que instancia o pré-processador correto para cada fonte."""
from src.preprocessing.base import Preprocessor
from src.preprocessing.categories import CategoriesPreprocessor
from src.preprocessing.events import EventsPreprocessor
from src.preprocessing.item_properties import ItemPropertiesPreprocessor

_REGISTRY: dict[str, type[Preprocessor]] = {
    "events": EventsPreprocessor,
    "categories": CategoriesPreprocessor,
    "item_properties": ItemPropertiesPreprocessor,
}


def build_preprocessor(source: str) -> Preprocessor:
    #Retorna a instância do pré-processador correspondente à fonte.
    if source not in _REGISTRY:
        raise KeyError(f"fonte desconhecida: {source!r}")
    return _REGISTRY[source]()