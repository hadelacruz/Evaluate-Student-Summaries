"""
src/features.py
Extracción de features NLP para los modelos de ML.

Separa la lógica de feature engineering de la de entrenamiento,
siguiendo el principio de separación de responsabilidades (SRP).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from . import config
from .preprocessing import add_length_features, add_length_ratio

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_matrix(
    train_texts: pd.Series,
    test_texts: Optional[pd.Series] = None,
    max_features: int = config.TFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = config.TFIDF_NGRAM_RANGE,
) -> tuple[np.ndarray, np.ndarray | None, TfidfVectorizer]:
    """Construye la matriz TF-IDF a partir de textos.

    Ajusta el vectorizador sobre ``train_texts`` y opcionalmente lo aplica
    a ``test_texts`` (para evitar data leakage).

    Args:
        train_texts: Serie de textos de entrenamiento.
        test_texts: Serie de textos de prueba (opcional).
        max_features: Máximo número de features TF-IDF.
        ngram_range: Rango de n-gramas, por defecto unigramas y bigramas.

    Returns:
        Tupla ``(X_train, X_test, vectorizer)``.
        ``X_test`` es ``None`` si no se proporcionó ``test_texts``.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,       # log(1 + tf), reduce el efecto de palabras muy frecuentes
        strip_accents="unicode",
        analyzer="word",
        min_df=2,                # ignorar términos que aparecen en menos de 2 documentos
    )

    X_train = vectorizer.fit_transform(train_texts.fillna("")).toarray()
    logger.info("TF-IDF matriz de entrenamiento: %s", X_train.shape)

    X_test = None
    if test_texts is not None:
        X_test = vectorizer.transform(test_texts.fillna("")).toarray()
        logger.info("TF-IDF matriz de prueba: %s", X_test.shape)

    return X_train, X_test, vectorizer


# ─────────────────────────────────────────────────────────────────────────────
# Features numéricas
# ─────────────────────────────────────────────────────────────────────────────

def build_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye las features numéricas del resumen.

    Features incluidas:
    - ``text_len_words``: número de palabras.
    - ``text_len_chars``: número de caracteres.
    - ``length_ratio_vs_prompt``: ratio palabras resumen / prompt.

    Args:
        df: DataFrame con columnas ``text`` y ``prompt_text``.

    Returns:
        DataFrame con solo las columnas de features numéricas.
    """
    df = add_length_features(df)
    if config.PROMPT_TEXT_COL in df.columns:
        df = add_length_ratio(df)
        cols = [
            config.TEXT_LEN_WORDS_COL,
            config.TEXT_LEN_CHARS_COL,
            config.LENGTH_RATIO_COL,
        ]
    else:
        cols = [config.TEXT_LEN_WORDS_COL, config.TEXT_LEN_CHARS_COL]

    return df[cols].copy()


def scale_numeric_features(
    train_numeric: pd.DataFrame,
    test_numeric: Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, np.ndarray | None, StandardScaler]:
    """Escala features numéricas con StandardScaler.

    Ajusta sobre train y transforma test para evitar data leakage.

    Args:
        train_numeric: DataFrame de features numéricas de entrenamiento.
        test_numeric: DataFrame de features numéricas de prueba (opcional).

    Returns:
        Tupla ``(X_train_scaled, X_test_scaled, scaler)``.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_numeric.fillna(train_numeric.median()))

    X_test = None
    if test_numeric is not None:
        X_test = scaler.transform(test_numeric.fillna(train_numeric.median()))

    return X_train, X_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix combinada
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Combina TF-IDF y features numéricas en una sola matriz.

    Pipeline:
    1. TF-IDF sobre ``text``
    2. Features numéricas escaladas (longitudes + ratio)
    3. Concatenación horizontal

    Args:
        train_df: DataFrame de entrenamiento (ya mergeado con prompts).
        test_df: DataFrame de prueba opcional.

    Returns:
        Tupla ``(X_train, X_test)``. ``X_test`` es ``None`` si no se proveyó.
    """
    # TF-IDF
    tfidf_train, tfidf_test, _ = build_tfidf_matrix(
        train_df[config.TEXT_COL],
        test_df[config.TEXT_COL] if test_df is not None else None,
    )

    # Numéricas
    num_train = build_numeric_features(train_df)
    num_test = build_numeric_features(test_df) if test_df is not None else None
    num_train_s, num_test_s, _ = scale_numeric_features(num_train, num_test)

    # Concatenar
    X_train = np.hstack([tfidf_train, num_train_s])

    X_test = None
    if test_df is not None and tfidf_test is not None and num_test_s is not None:
        X_test = np.hstack([tfidf_test, num_test_s])

    logger.info("Feature matrix final: %s", X_train.shape)
    return X_train, X_test
