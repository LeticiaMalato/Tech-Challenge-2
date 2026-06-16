"""Factory que instancia o pré-processador correto para cada fonte de dados."""
from src.preprocessing.base import Preprocessor
from src.preprocessing.categories import CategoriesPreprocessor
from src.preprocessing.events import EventsPreprocessor
from src.preprocessing.item_properties import ItemPropertiesPreprocessor

__all__ = ["build_preprocessor", "REGISTRY"]

REGISTRY: dict[str, type[Preprocessor]] = {
    "events": EventsPreprocessor,
    "categories": CategoriesPreprocessor,
    "item_properties": ItemPropertiesPreprocessor,
}


def build_preprocessor(source: str) -> Preprocessor:
    """Instancia e retorna o pré-processador correspondente à fonte.

    Args:
        source: Identificador da fonte de dados (ex: 'events', 'categories').

    Returns:
        Instância do Preprocessor concreto para a fonte solicitada.

    Raises:
        KeyError: Se a fonte não estiver registrada no REGISTRY.
    """
    if source not in REGISTRY:
        valid = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Fonte desconhecida: {source!r}. Válidas: {valid}")
    return REGISTRY[source]()