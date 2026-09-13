"""
src/visualization.py
Funciones de visualización reutilizables para EDA y resultados de modelos.

Todas las funciones devuelven la figura (matplotlib.figure.Figure) para que
puedan usarse tanto en notebooks como en la app Streamlit sin acoplamiento.

Principio DRY: cada función de plot se define una vez y se reutiliza en
notebooks/ y app/.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

# ── Estilo global consistente ─────────────────────────────────────────────────
PALETTE = "#4C72B0"
PALETTE_MULTI = sns.color_palette("tab10")
sns.set_theme(style="whitegrid", palette="tab10")


# ─────────────────────────────────────────────────────────────────────────────
# EDA — Distribuciones
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_distributions(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Histogramas de ``content``, ``wording`` y ``text_len_words``.

    Args:
        df: DataFrame con las columnas de scores y features derivadas.

    Returns:
        Figura de matplotlib con 3 subplots.
    """
    cols = [
        (config.CONTENT_COL, "Puntuación content"),
        (config.WORDING_COL, "Puntuación wording"),
        (config.TEXT_LEN_WORDS_COL, "Longitud del resumen (palabras)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, label) in zip(axes, cols):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=ax, color=PALETTE)
            ax.set_title(f"Distribución de {label}")
            ax.set_xlabel(label)
            ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    return fig


def plot_score_boxplots_by_prompt(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Diagramas de caja de ``content`` y ``wording`` por prompt.

    Args:
        df: DataFrame con ``prompt_title``, ``content`` y ``wording``.

    Returns:
        Figura de matplotlib con 2 subplots lado a lado.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, col in zip(axes, [config.CONTENT_COL, config.WORDING_COL]):
        if col in df.columns and config.PROMPT_TITLE_COL in df.columns:
            sns.boxplot(
                data=df,
                x=config.PROMPT_TITLE_COL,
                y=col,
                ax=ax,
                palette="Set2",
            )
            ax.set_title(f"Distribución de '{col}' por prompt")
            ax.set_xlabel("Prompt (texto fuente)")
            ax.set_ylabel(col)
            ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Mapa de calor de correlaciones entre variables numéricas.

    Args:
        df: DataFrame con al menos las columnas numéricas de interés.

    Returns:
        Figura de matplotlib con el heatmap.
    """
    numeric_cols = [
        c for c in [
            config.CONTENT_COL,
            config.WORDING_COL,
            config.TEXT_LEN_WORDS_COL,
            config.TEXT_LEN_CHARS_COL,
            config.LENGTH_RATIO_COL,
        ]
        if c in df.columns
    ]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
    )
    ax.set_title("Correlaciones entre variables numéricas")
    fig.tight_layout()
    return fig


def plot_scatter_length_vs_scores(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Diagramas de dispersión: longitud del resumen vs scores.

    Args:
        df: DataFrame con ``text_len_words``, ``content`` y ``wording``.

    Returns:
        Figura de matplotlib con 2 subplots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, score_col in zip(axes, [config.CONTENT_COL, config.WORDING_COL]):
        if all(c in df.columns for c in [config.TEXT_LEN_WORDS_COL, score_col]):
            corr_val = df[[config.TEXT_LEN_WORDS_COL, score_col]].corr().iloc[0, 1]
            sns.scatterplot(
                data=df,
                x=config.TEXT_LEN_WORDS_COL,
                y=score_col,
                alpha=0.3,
                ax=ax,
                color=PALETTE,
            )
            ax.set_title(f"Longitud vs {score_col} (r = {corr_val:.2f})")
            ax.set_xlabel("Palabras en el resumen")
            ax.set_ylabel(score_col)

    fig.tight_layout()
    return fig


def plot_prompt_frequency(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Gráfico de barras: número de resúmenes por prompt.

    Args:
        df: DataFrame con ``prompt_title``.

    Returns:
        Figura de matplotlib.
    """
    if config.PROMPT_TITLE_COL not in df.columns:
        raise ValueError(f"La columna '{config.PROMPT_TITLE_COL}' no está en el DataFrame.")

    freq = df[config.PROMPT_TITLE_COL].value_counts().reset_index()
    freq.columns = [config.PROMPT_TITLE_COL, "frecuencia"]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=config.PROMPT_TITLE_COL, y="frecuencia", data=freq, ax=ax, color=PALETTE)
    ax.set_xlabel("Prompt (texto fuente)")
    ax.set_ylabel("Número de resúmenes")
    ax.set_title("Resúmenes por prompt")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Resultados de modelos
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "rmse",
    target: str = "content",
) -> matplotlib.figure.Figure:
    """Gráfico de barras comparando modelos según una métrica.

    Args:
        results_df: DataFrame con columnas ``[modelo, target, rmse, mae, r2]``.
        metric: Métrica a graficar (``"rmse"``, ``"mae"`` o ``"r2"``).
        target: Filtrar por target (``"content"`` o ``"wording"``).

    Returns:
        Figura de matplotlib.
    """
    subset = results_df[results_df["target"] == target].copy()
    if subset.empty:
        raise ValueError(f"No hay resultados para target='{target}'")

    # Para RMSE y MAE: menor es mejor; para R²: mayor es mejor
    ascending = metric in ("rmse", "mae")
    subset = subset.sort_values(metric, ascending=ascending)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(subset["modelo"], subset[metric], color=PALETTE_MULTI[: len(subset)])
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Comparación de modelos — {metric.upper()} ({target})")
    ax.bar_label(bars, fmt="%.4f", padding=3)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Modelo",
    target: str = "content",
) -> matplotlib.figure.Figure:
    """Diagrama de dispersión real vs predicho.

    Args:
        y_true: Valores reales.
        y_pred: Valores predichos.
        model_name: Nombre del modelo para el título.
        target: Nombre del target.

    Returns:
        Figura de matplotlib.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, color=PALETTE, s=10)
    lim = [min(y_true.min(), y_pred.min()) - 0.1, max(y_true.max(), y_pred.max()) + 0.1]
    ax.plot(lim, lim, "r--", linewidth=1, label="Predicción perfecta")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"Real ({target})")
    ax.set_ylabel(f"Predicho ({target})")
    ax.set_title(f"{model_name} — Real vs Predicho")
    ax.legend()
    fig.tight_layout()
    return fig
