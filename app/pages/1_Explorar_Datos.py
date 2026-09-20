"""
app/pages/1_Explorar_Datos.py
Página 1: Exploración interactiva del dataset con gráficos Plotly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src import config
from src.preprocessing import load_and_merge, add_length_features, add_length_ratio

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Explorar Datos — CommonLit", page_icon="📊", layout="wide")

# Paleta de colores consistente con el proyecto
COLORS = px.colors.qualitative.Plotly
PROMPT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

st.title("📊 Exploración de Datos")
st.caption("Analiza distribuciones, estadísticos y correlaciones del dataset de entrenamiento.")
st.divider()


# ── Carga de datos ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando dataset...")
def load_data() -> pd.DataFrame:
    df = load_and_merge("train")
    df = add_length_features(df)
    df = add_length_ratio(df)
    return df


df = load_data()

# ── Sidebar — filtros ─────────────────────────────────────────────────────────
st.sidebar.header("Filtros")
all_prompts = ["Todos"] + sorted(df[config.PROMPT_TITLE_COL].unique().tolist())
selected_prompt = st.sidebar.selectbox("Prompt (texto fuente)", all_prompts)

if selected_prompt != "Todos":
    df_filtered = df[df[config.PROMPT_TITLE_COL] == selected_prompt]
else:
    df_filtered = df

st.sidebar.markdown(f"**Observaciones filtradas:** {len(df_filtered):,}")

# ─────────────────────────────────────────────────────────────────────────────
# Sección 1: Vista general
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Vista general del dataset", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total observaciones", f"{len(df_filtered):,}")
    col2.metric("Prompts únicos", df_filtered[config.PROMPT_ID_COL].nunique())
    col3.metric("Media content", f"{df_filtered[config.CONTENT_COL].mean():.3f}")
    col4.metric("Media wording", f"{df_filtered[config.WORDING_COL].mean():.3f}")

    st.dataframe(
        df_filtered[[
            config.STUDENT_ID_COL, config.PROMPT_TITLE_COL,
            config.TEXT_COL, config.CONTENT_COL, config.WORDING_COL,
            config.TEXT_LEN_WORDS_COL,
        ]].head(8),
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sección 2: Estadísticos descriptivos
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Estadísticos descriptivos — variables numéricas", expanded=True):
    numeric_cols = [
        config.CONTENT_COL, config.WORDING_COL,
        config.TEXT_LEN_WORDS_COL, config.TEXT_LEN_CHARS_COL,
    ]
    if config.LENGTH_RATIO_COL in df_filtered.columns:
        numeric_cols.append(config.LENGTH_RATIO_COL)

    st.dataframe(
        df_filtered[numeric_cols].describe().T.style.format("{:.3f}"),
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sección 3: Distribuciones interactivas
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Distribuciones de puntuaciones y longitud", expanded=True):
    col_left, col_right = st.columns(2)

    with col_left:
        # Histogramas de content y wording superpuestos
        fig = go.Figure()
        for col, color, name in [
            (config.CONTENT_COL, "#4C72B0", "content"),
            (config.WORDING_COL, "#DD8452", "wording"),
        ]:
            fig.add_trace(go.Histogram(
                x=df_filtered[col],
                name=name,
                opacity=0.7,
                nbinsx=40,
                marker_color=color,
            ))
        fig.update_layout(
            title="Distribución de puntuaciones (content vs wording)",
            xaxis_title="Puntuación",
            yaxis_title="Frecuencia",
            barmode="overlay",
            legend=dict(orientation="h", y=1.1),
            height=380,
        )
        st.plotly_chart(fig, width="stretch")

    with col_right:
        # Histograma de longitud con curva KDE simulada
        fig = px.histogram(
            df_filtered,
            x=config.TEXT_LEN_WORDS_COL,
            nbins=50,
            color_discrete_sequence=["#55A868"],
            title="Longitud del resumen (palabras)",
            labels={config.TEXT_LEN_WORDS_COL: "Número de palabras"},
        )
        fig.update_layout(yaxis_title="Frecuencia", height=380)
        st.plotly_chart(fig, width="stretch")

    # Scatter: longitud vs puntuaciones
    st.markdown("**Longitud del resumen vs puntuaciones** *(hover para ver detalles)*")
    col_s1, col_s2 = st.columns(2)

    for col, score_col, color in [
        (col_s1, config.CONTENT_COL, "#4C72B0"),
        (col_s2, config.WORDING_COL, "#DD8452"),
    ]:
        with col:
            corr_val = df_filtered[[config.TEXT_LEN_WORDS_COL, score_col]].corr().iloc[0, 1]
            fig = px.scatter(
                df_filtered.sample(min(2000, len(df_filtered)), random_state=42),
                x=config.TEXT_LEN_WORDS_COL,
                y=score_col,
                color=config.PROMPT_TITLE_COL if config.PROMPT_TITLE_COL in df_filtered.columns else None,
                opacity=0.4,
                trendline="ols",
                title=f"Longitud vs {score_col} (r = {corr_val:.2f})",
                labels={
                    config.TEXT_LEN_WORDS_COL: "Palabras en el resumen",
                    score_col: score_col,
                },
                color_discrete_sequence=PROMPT_COLORS,
            )
            fig.update_layout(height=350, showlegend=(selected_prompt == "Todos"))
            st.plotly_chart(fig, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# Sección 4: Boxplots por prompt
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Distribución por prompt (boxplots)", expanded=False):
    col_b1, col_b2 = st.columns(2)

    for col, score_col in [(col_b1, config.CONTENT_COL), (col_b2, config.WORDING_COL)]:
        with col:
            fig = px.box(
                df,
                x=config.PROMPT_TITLE_COL,
                y=score_col,
                color=config.PROMPT_TITLE_COL,
                title=f"'{score_col}' por prompt",
                color_discrete_sequence=PROMPT_COLORS,
                points="outliers",
            )
            fig.update_layout(
                xaxis_title="",
                showlegend=False,
                height=380,
                xaxis_tickangle=-20,
            )
            st.plotly_chart(fig, width="stretch")

    st.markdown("""
    **Interpretación:** Los resúmenes del mismo prompt tienen distribuciones distintas de `content` y `wording`,
    lo que sugiere que la dificultad del texto fuente influye en la calidad de los resúmenes.
    Los puntos fuera de los bigotes son outliers (resúmenes atípicamente buenos o malos).
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Sección 5: Mapa de correlaciones interactivo
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Mapa de correlaciones", expanded=False):
    corr_cols = [c for c in numeric_cols if c in df_filtered.columns]
    corr = df_filtered[corr_cols].corr().round(3)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlaciones entre variables numéricas",
        aspect="auto",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")

    st.markdown("""
    **Hallazgos clave:**
    - `text_len_words` y `content` tienen correlación fuerte positiva (**r ≈ 0.79**): los resúmenes más largos capturan mejor la idea principal.
    - `content` y `wording` tienen correlación moderada (**r ≈ 0.74**): un buen contenido tiende a ir de la mano con buen lenguaje.
    - `text_len_words` y `text_len_chars` tienen correlación casi perfecta (**r ≈ 0.99**) — redundantes; el modelo solo necesita una.
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Sección 6: Frecuencia por prompt interactiva
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Tabla de frecuencia por prompt", expanded=False):
    freq = df[config.PROMPT_TITLE_COL].value_counts().reset_index()
    freq.columns = ["Prompt", "Frecuencia"]
    freq["Proporción (%)"] = (freq["Frecuencia"] / len(df) * 100).round(2)

    col_t, col_c = st.columns([1, 2])
    with col_t:
        st.dataframe(freq, use_container_width=True)
    with col_c:
        fig = px.bar(
            freq,
            x="Prompt",
            y="Frecuencia",
            color="Prompt",
            text="Proporción (%)",
            title="Número de resúmenes por prompt",
            color_discrete_sequence=PROMPT_COLORS,
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(showlegend=False, height=340, xaxis_tickangle=-15)
        st.plotly_chart(fig, width="stretch")
