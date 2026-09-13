"""
src/models.py
Entrenamiento y evaluación de modelos de regresión para CommonLit.

El problema es de REGRESIÓN: predecir las puntuaciones continuas
``content`` y ``wording`` de cada resumen de estudiante.

Modelos evaluados:
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- HistGradientBoosting Regressor (versión moderna y compatible con sklearn >= 1.0)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from . import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Catálogo de modelos
# ─────────────────────────────────────────────────────────────────────────────

def get_model_catalog() -> dict[str, Any]:
    """Devuelve un diccionario con los modelos a evaluar y sus hiperparámetros.

    Returns:
        Dict ``{nombre: instancia_sklearn}``.
    """
    return {
        "Ridge": Ridge(alpha=1.0, random_state=config.RANDOM_SEED),
        "Lasso": Lasso(alpha=0.01, random_state=config.RANDOM_SEED, max_iter=5000),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=config.RANDOM_SEED,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula métricas de regresión estándar.

    Args:
        y_true: Valores reales.
        y_pred: Valores predichos.

    Returns:
        Dict con ``rmse``, ``mae`` y ``r2``.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento y evaluación
# ─────────────────────────────────────────────────────────────────────────────

def train_evaluate_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_name: str = "content",
) -> pd.DataFrame:
    """Entrena todos los modelos del catálogo y devuelve sus métricas.

    Args:
        X_train: Matriz de features de entrenamiento.
        y_train: Vector de targets de entrenamiento.
        X_test: Matriz de features de prueba.
        y_test: Vector de targets de prueba.
        target_name: Nombre del target (``"content"`` o ``"wording"``).

    Returns:
        DataFrame con columnas ``[modelo, target, rmse, mae, r2]``.
    """
    catalog = get_model_catalog()
    records: list[dict] = []

    for name, model in catalog.items():
        logger.info("Entrenando %s → target=%s", name, target_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        records.append({"modelo": name, "target": target_name, **metrics})
        logger.info(
            "  %s | RMSE=%.4f | MAE=%.4f | R²=%.4f",
            name, metrics["rmse"], metrics["mae"], metrics["r2"],
        )

    return pd.DataFrame(records)


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide los datos en entrenamiento y prueba de forma reproducible.

    Args:
        X: Matriz de features.
        y: Vector de targets.
        test_size: Fracción del set de prueba (default: 0.2 = 20%).

    Returns:
        Tupla ``(X_train, X_test, y_train, y_test)``.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=config.RANDOM_SEED
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia de modelos
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: Any, name: str, target: str) -> Path:
    """Serializa un modelo entrenado con joblib.

    El modelo se guarda en ``models/{name}_{target}.pkl``.

    Args:
        model: Objeto sklearn entrenado.
        name: Nombre del algoritmo (e.g. ``"Ridge"``).
        target: Nombre del target (e.g. ``"content"``).

    Returns:
        Ruta donde se guardó el archivo.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / f"{name}_{target}.pkl"
    joblib.dump(model, path)
    logger.info("Modelo guardado en %s", path)
    return path


def load_model(name: str, target: str) -> Any:
    """Carga un modelo previamente serializado.

    Args:
        name: Nombre del algoritmo.
        target: Nombre del target.

    Returns:
        Objeto sklearn cargado.

    Raises:
        FileNotFoundError: Si el archivo .pkl no existe.
    """
    path = config.MODELS_DIR / f"{name}_{target}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {path}\n"
            "Ejecuta notebooks/02_modeling.ipynb primero para entrenar los modelos."
        )
    return joblib.load(path)


def list_trained_models() -> list[dict[str, str]]:
    """Lista los modelos entrenados disponibles en ``models/``.

    Returns:
        Lista de dicts con ``{"name": ..., "target": ..., "path": ...}``.
    """
    if not config.MODELS_DIR.exists():
        return []

    models = []
    for pkl in sorted(config.MODELS_DIR.glob("*.pkl")):
        parts = pkl.stem.rsplit("_", 1)
        if len(parts) == 2:
            models.append({"name": parts[0], "target": parts[1], "path": str(pkl)})
    return models
