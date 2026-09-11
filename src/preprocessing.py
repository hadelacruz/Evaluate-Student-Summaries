"""
src/preprocessing.py
Funciones de carga, validación y preprocesamiento de datos.

Cada función tiene una responsabilidad única (principio SRP) y devuelve
siempre un DataFrame nuevo para no mutar el original (inmutabilidad defensiva).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────────────────────────────────────

def load_summaries(split: str = "train") -> pd.DataFrame:
    """Carga el archivo de resúmenes de estudiantes.

    Args:
        split: ``"train"`` o ``"test"``.

    Returns:
        DataFrame con las columnas originales del CSV.

    Raises:
        FileNotFoundError: Si el archivo no existe en ``data/``.
        ValueError: Si ``split`` no es ``"train"`` o ``"test"``.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split debe ser 'train' o 'test', recibió: {split!r}")

    path = config.SUMMARIES_TRAIN if split == "train" else config.SUMMARIES_TEST
    _assert_exists(path)
    df = pd.read_csv(path)
    logger.info("Cargado %s con forma %s", path.name, df.shape)
    return df


def load_prompts(split: str = "train") -> pd.DataFrame:
    """Carga el archivo de prompts (textos fuente).

    Args:
        split: ``"train"`` o ``"test"``.

    Returns:
        DataFrame con las columnas originales del CSV.

    Raises:
        FileNotFoundError: Si el archivo no existe en ``data/``.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split debe ser 'train' o 'test', recibió: {split!r}")

    path = config.PROMPTS_TRAIN if split == "train" else config.PROMPTS_TEST
    _assert_exists(path)
    df = pd.read_csv(path)
    logger.info("Cargado %s con forma %s", path.name, df.shape)
    return df


def load_and_merge(split: str = "train") -> pd.DataFrame:
    """Carga y fusiona resúmenes con prompts por ``prompt_id``.

    Args:
        split: ``"train"`` o ``"test"``.

    Returns:
        DataFrame combinado con todas las columnas de ambas fuentes.
    """
    summaries = load_summaries(split)
    prompts = load_prompts(split)
    merged = summaries.merge(prompts, on=config.PROMPT_ID_COL, how="left")
    logger.info("Merge completado: forma final %s", merged.shape)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Validación de calidad
# ─────────────────────────────────────────────────────────────────────────────

def check_quality(df: pd.DataFrame) -> dict[str, object]:
    """Verifica nulos, duplicados e integridad referencial.

    Args:
        df: DataFrame a inspeccionar.

    Returns:
        Diccionario con resultados:
        ``{"missing": Series, "duplicates": int, "valid": bool}``
    """
    missing = df.isna().sum()
    duplicates = int(df.duplicated().sum())
    has_problems = missing.any() or duplicates > 0

    report = {
        "missing": missing,
        "duplicates": duplicates,
        "valid": not has_problems,
    }

    if has_problems:
        logger.warning(
            "Problemas de calidad: %d nulos totales, %d duplicados",
            int(missing.sum()),
            duplicates,
        )
    else:
        logger.info("Calidad de datos OK — sin nulos ni duplicados")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering básico
# ─────────────────────────────────────────────────────────────────────────────

def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas de longitud del texto del resumen.

    Columnas añadidas:
    - ``text_len_words``: número de palabras en el resumen.
    - ``text_len_chars``: número de caracteres en el resumen.

    Args:
        df: DataFrame que debe contener la columna ``text``.

    Returns:
        Copia del DataFrame con las nuevas columnas.
    """
    out = df.copy()
    out[config.TEXT_LEN_WORDS_COL] = out[config.TEXT_COL].str.split().str.len()
    out[config.TEXT_LEN_CHARS_COL] = out[config.TEXT_COL].str.len()
    return out


def add_length_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el ratio de longitud resumen / prompt_text.

    Columnas añadidas:
    - ``length_ratio_vs_prompt``: palabras del resumen / palabras del prompt.

    Args:
        df: DataFrame que debe contener ``text_len_words`` y ``prompt_text``.
            Llama primero a :func:`add_length_features`.

    Returns:
        Copia del DataFrame con la nueva columna.
    """
    if config.TEXT_LEN_WORDS_COL not in df.columns:
        df = add_length_features(df)

    out = df.copy()
    prompt_words = out[config.PROMPT_TEXT_COL].str.split().str.len()
    # Evitar división por cero
    out[config.LENGTH_RATIO_COL] = out[config.TEXT_LEN_WORDS_COL] / prompt_words.replace(0, pd.NA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _assert_exists(path: Path) -> None:
    """Lanza FileNotFoundError si el archivo no existe."""
    if not path.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {path}\n"
            "Asegúrate de haber descargado los datos desde Kaggle y colocarlos en data/."
        )
