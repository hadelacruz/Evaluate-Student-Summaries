"""
app/app.py
Página de inicio de la aplicación Streamlit — CommonLit: Evaluate Student Summaries.

Lanzar con:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

# ── Agregar raíz del proyecto al path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Configuración global de la página ────────────────────────────────────────
st.set_page_config(
    page_title="CommonLit — Evaluate Student Summaries",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos personalizados ────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .card {
        background: #f0f4ff;
        border-left: 5px solid #4C72B0;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        background: #4C72B0;
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 12px;
        font-size: 0.85rem;
        margin-right: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Contenido ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📚 CommonLit — Evaluate Student Summaries</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Proyecto 2 · CC3084 Data Science · UVG Semestre II 2026 · '
    '<span class="badge">Reto #12</span>'
    '<span class="badge">NLP</span>'
    '<span class="badge">Regresión</span></p>',
    unsafe_allow_html=True,
)

st.divider()

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("¿Qué hace esta aplicación?")
    st.markdown("""
    Esta app interactiva permite explorar y evaluar modelos de Machine Learning
    entrenados para **predecir la calidad de resúmenes escritos por estudiantes**
    de los grados 3 a 12.

    Los modelos predicen dos puntuaciones:
    - **`content`** — qué tan bien el resumen captura la idea principal del texto fuente.
    - **`wording`** — calidad del lenguaje: claridad, precisión y fluidez.

    **Usa el menú de la izquierda para navegar entre las páginas.**
    """)

    st.info(
        "💡 **Competencia de Kaggle**: "
        "[CommonLit — Evaluate Student Summaries](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries)"
    )

with col2:
    st.subheader("Navegación")

    pages = [
        ("📊", "1 — Explorar Datos", "Analiza distribuciones, correlaciones y estadísticas descriptivas del dataset."),
        ("🤖", "2 — Resultados de Modelos", "Ingresa un resumen y obtén predicciones de todos los modelos."),
        ("⚖️", "3 — Comparación de Modelos", "Compara RMSE, MAE y R² de los 4 algoritmos de forma interactiva."),
    ]

    for icon, title, desc in pages:
        st.markdown(
            f'<div class="card"><strong>{icon} {title}</strong><br><small>{desc}</small></div>',
            unsafe_allow_html=True,
        )

st.divider()

st.subheader("Dataset")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Resúmenes de entrenamiento", "7,165")
metric_col2.metric("Prompts (textos fuente)", "4")
metric_col3.metric("Variables target", "2 (content, wording)")
metric_col4.metric("Modelos evaluados", "4")

st.divider()
st.caption(
    "Desarrollado por: "
    "[@hadelacruz](https://github.com/hadelacruz) · "
    "[@djuarez-2017510](https://github.com/djuarez-2017510) · "
    "[@GerardoFdez7](https://github.com/GerardoFdez7) · "
    "[@jruiz002](https://github.com/jruiz002) · "
    "[@nicollegordillo](https://github.com/nicollegordillo)"
)
