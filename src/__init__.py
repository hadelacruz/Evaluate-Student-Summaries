"""
src/__init__.py
Paquete de utilidades para CommonLit — Evaluate Student Summaries.

Los módulos se importan bajo demanda (lazy) para evitar que la ausencia
de dependencias opcionales rompa la importación del paquete completo.
"""
from . import config  # config no tiene dependencias externas

__all__ = ["config", "preprocessing", "features", "models", "visualization"]


def __getattr__(name: str):  # type: ignore[override]
    """Importación lazy de submódulos con dependencias externas."""
    if name in __all__:
        import importlib
        module = importlib.import_module(f".{name}", package=__name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module 'src' has no attribute {name!r}")
