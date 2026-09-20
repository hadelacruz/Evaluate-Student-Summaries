"""
app/app.py — CommonLit: Evaluate Student Summaries
Página de inicio. Lanzar con: streamlit run app/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="CommonLit — Evaluate Student Summaries",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Encabezado
# ─────────────────────────────────────────────────────────────────────────────
st.title("📚 CommonLit — Evaluate Student Summaries")
st.markdown(
    "**Proyecto 2 · CC3084 Data Science · UVG Semestre II 2026** &nbsp;|&nbsp; "
    "`Reto #12` &nbsp;·&nbsp; `NLP` &nbsp;·&nbsp; `Regresión`"
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Descripción + Navegación
# ─────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.subheader("¿Qué hace esta aplicación?")
    st.markdown("""
    Esta app interactiva permite **explorar y evaluar modelos de Machine Learning**
    entrenados para predecir la calidad de resúmenes escritos por estudiantes de los grados 3 a 12.

    Los modelos predicen dos puntuaciones continuas:

    | Target | Descripción |
    |---|---|
    | `content` | Qué tan bien el resumen captura la idea principal del texto fuente |
    | `wording` | Calidad del lenguaje: claridad, precisión y fluidez |

    > 💡 **Competencia de Kaggle:** [CommonLit — Evaluate Student Summaries](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries)
    """)

with col2:
    st.subheader("Navegación")

    with st.container(border=True):
        st.markdown("**📊 1 — Explorar Datos**")
        st.caption("Analiza distribuciones, correlaciones y estadísticas descriptivas del dataset con filtros interactivos.")

    with st.container(border=True):
        st.markdown("**🤖 2 — Resultados de Modelos**")
        st.caption("Ingresa un resumen de texto libre y obtén predicciones de todos los modelos entrenados.")

    with st.container(border=True):
        st.markdown("**⚖️ 3 — Comparación de Modelos**")
        st.caption("Compara RMSE, MAE y R² de los 4 algoritmos. Muestra u oculta métricas desde el sidebar.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Métricas del dataset
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Dataset")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Resúmenes de entrenamiento", "7,165")
c2.metric("Prompts (textos fuente)", "4")
c3.metric("Variables target", "content, wording")
c4.metric("Modelos evaluados", "4")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Arquitectura del proyecto
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🏗️ Arquitectura del proyecto", expanded=False):
    st.markdown("""
    ```
    Evaluate-Student-Summaries/
    ├── data/               ← CSVs de Kaggle (summaries + prompts)
    ├── notebooks/
    │   ├── 01_eda.ipynb    ← Análisis Exploratorio de Datos
    │   └── 02_modeling.ipynb ← Entrenamiento y evaluación de modelos
    ├── src/                ← Paquete Python reutilizable
    │   ├── config.py       ← Rutas y constantes centralizadas
    │   ├── preprocessing.py
    │   ├── features.py     ← TF-IDF + features numéricas (sin data leakage)
    │   ├── models.py       ← Ridge, Lasso, RandomForest, HistGradientBoosting
    │   └── visualization.py
    ├── models/             ← Modelos serializados (.pkl)
    ├── reports/figures/    ← Figuras estáticas para el informe
    └── app/                ← Esta aplicación Streamlit
    ```

    **Flujo de datos:**
    `data/ → src/preprocessing → src/features → src/models → models/*.pkl → app/`
    """)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Autores
# ─────────────────────────────────────────────────────────────────────────────
st.caption(
    "Desarrollado por: "
    "[@hadelacruz](https://github.com/hadelacruz) · "
    "[@djuarez-2017510](https://github.com/djuarez-2017510) · "
    "[@GerardoFdez7](https://github.com/GerardoFdez7) · "
    "[@jruiz002](https://github.com/jruiz002) · "
    "[@nicollegordillo](https://github.com/nicollegordillo)"
)
