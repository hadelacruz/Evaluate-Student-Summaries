"""
app/pages/2_Resultados_Modelos.py
Página 2: Predicción interactiva de calidad de resúmenes.
Usa Plotly para todas las visualizaciones de resultados.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

from src import config

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Resultados de Modelos — CommonLit", page_icon="🤖", layout="wide")

COLORS = {"Ridge": "#4C72B0", "Lasso": "#DD8452", "RandomForest": "#55A868",
          "HistGradientBoosting": "#C44E52"}

st.title("🤖 Resultados de Modelos")
st.caption("Ingresa un resumen de estudiante y obtén predicciones de `content` y `wording`.")
st.divider()


# ── Carga de modelos ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelos...")
def load_artifacts():
    vec_path = config.MODELS_DIR / "tfidf_vectorizer.pkl"
    scaler_path = config.MODELS_DIR / "numeric_scaler.pkl"
    if not vec_path.exists() or not scaler_path.exists():
        return None

    try:
        artifacts = {
            "vectorizer": joblib.load(vec_path),
            "scaler": joblib.load(scaler_path),
            "models": {},
        }
    except Exception as e:
        st.error(f"Error cargando vectorizer/scaler: {e}")
        return None

    skipped = []
    for pkl in sorted(config.MODELS_DIR.glob("*.pkl")):
        if pkl.stem in ("tfidf_vectorizer", "numeric_scaler"):
            continue
        parts = pkl.stem.rsplit("_", 1)
        if len(parts) == 2:
            name, target = parts
            try:
                model = joblib.load(pkl)
                if name not in artifacts["models"]:
                    artifacts["models"][name] = {}
                artifacts["models"][name][target] = model
            except Exception:
                skipped.append(pkl.name)

    if skipped:
        st.warning(
            f"⚠️ Se omitieron {len(skipped)} modelo(s) incompatibles con la versión actual de sklearn: "
            f"`{'`, `'.join(skipped)}`. "
            "Vuelve a correr `notebooks/02_modeling.ipynb` para regenerarlos."
        )

    return artifacts if artifacts["models"] else None


def build_features(text: str, artifacts: dict) -> np.ndarray:
    """Construye el vector de features para un texto de entrada."""
    tfidf = artifacts["vectorizer"].transform([text]).toarray()
    words = len(text.split())
    chars = len(text)
    # Siempre 2 features numéricas: text_len_words, text_len_chars
    num_raw = np.array([[words, chars]], dtype=float)
    num_scaled = artifacts["scaler"].transform(num_raw)
    return np.hstack([tfidf, num_scaled])


# ── Verificar disponibilidad ──────────────────────────────────────────────────
artifacts = load_artifacts()

if artifacts is None:
    st.warning(
        "**Modelos no encontrados.**\n\n"
        "Ejecuta `notebooks/02_modeling.ipynb` para entrenar y guardar los modelos."
    )
    st.stop()

available_models = list(artifacts["models"].keys())

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Configuración")
model_selection = st.sidebar.radio(
    "Modelos a usar",
    options=["Todos los modelos"] + available_models,
    index=0,
)
show_metrics = st.sidebar.checkbox("Mostrar métricas de rendimiento", value=True)

# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("Ingresa el resumen del estudiante")

col_input, col_info = st.columns([3, 1])
with col_input:
    user_text = st.text_area(
        "Texto del resumen:",
        height=180,
        placeholder="Escribe o pega aquí el resumen del estudiante...",
    )
with col_info:
    st.markdown("**¿Qué se predice?**")
    st.markdown("""
    - **content**: Qué tan bien el resumen captura la idea principal del texto fuente.  
    *Escala: centrada en 0, ±σ = 1*

    - **wording**: Calidad del lenguaje: claridad, precisión y fluidez.  
    *Escala: centrada en 0, ±σ = 1*
    """)

predict_btn = st.button("🔍 Predecir", type="primary", disabled=(not user_text.strip()))

# ── Predicción ────────────────────────────────────────────────────────────────
if predict_btn and user_text.strip():
    with st.spinner("Procesando..."):
        features = build_features(user_text.strip(), artifacts)
        models_to_run = available_models if model_selection == "Todos los modelos" else [model_selection]

        results = []
        for model_name in models_to_run:
            row = {"Modelo": model_name}
            for target in config.TARGET_COLS:
                if target in artifacts["models"].get(model_name, {}):
                    pred = float(artifacts["models"][model_name][target].predict(features)[0])
                    row[target] = round(pred, 4)
                else:
                    row[target] = None
            results.append(row)

    results_df = pd.DataFrame(results)
    words = len(user_text.split())
    chars = len(user_text)

    st.divider()

    # ── Métricas rápidas ──────────────────────────────────────────────────────
    st.subheader("Resultados de predicción")

    info_cols = st.columns(3)
    info_cols[0].metric("Palabras en el texto", f"{words:,}")
    info_cols[1].metric("Caracteres", f"{chars:,}")
    info_cols[2].metric("Modelos evaluados", len(results_df))

    st.markdown("---")

    # ── Gauge chart por target ────────────────────────────────────────────────
    if model_selection == "Todos los modelos":
        # Promedio de todos los modelos como gauge
        avg_content = results_df["content"].dropna().mean()
        avg_wording = results_df["wording"].dropna().mean()

        gauge_cols = st.columns(2)
        for gcol, val, label, color in [
            (gauge_cols[0], avg_content, "content (promedio)", "#4C72B0"),
            (gauge_cols[1], avg_wording, "wording (promedio)", "#DD8452"),
        ]:
            with gcol:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=round(val, 3),
                    delta={"reference": 0, "valueformat": ".3f"},
                    title={"text": label, "font": {"size": 16}},
                    gauge={
                        "axis": {"range": [-3, 3], "tickformat": ".1f"},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [-3, -1], "color": "#ffcccc"},
                            {"range": [-1, 1], "color": "#fff3cc"},
                            {"range": [1, 3], "color": "#ccffcc"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 2},
                            "thickness": 0.75,
                            "value": 0,
                        },
                    },
                ))
                fig.update_layout(height=280)
                st.plotly_chart(fig, width="stretch")

        st.caption("Verde = por encima del promedio | Amarillo = promedio | Rojo = por debajo del promedio")
        st.markdown("---")

    # ── Gráfico de barras: predicción por modelo ──────────────────────────────
    col_c, col_w = st.columns(2)

    for col, target, color_seq in [
        (col_c, "content", ["#4C72B0", "#6b95d6", "#8db3e8", "#b0d0f5"]),
        (col_w, "wording", ["#DD8452", "#e8a070", "#f0bc94", "#f8d8bc"]),
    ]:
        with col:
            target_df = results_df[["Modelo", target]].dropna()
            fig = go.Figure()
            for i, row in target_df.iterrows():
                model_name = row["Modelo"]
                c = COLORS.get(model_name, "#888888")
                fig.add_trace(go.Bar(
                    name=model_name,
                    x=[model_name],
                    y=[row[target]],
                    marker_color=c,
                    text=[f"{row[target]:.3f}"],
                    textposition="outside",
                    hovertemplate=f"<b>{model_name}</b><br>{target}: %{{y:.4f}}<extra></extra>",
                ))
            # Línea de referencia en 0 (media de la distribución)
            fig.add_hline(y=0, line_dash="dash", line_color="gray",
                          annotation_text="Media del dataset (0)")
            fig.update_layout(
                title=f"Predicción de '{target}' por modelo",
                yaxis_title=f"Puntuación {target}",
                yaxis_range=[-3.5, 3.5],
                showlegend=False,
                height=380,
            )
            st.plotly_chart(fig, width="stretch")

    # ── Tabla detallada ───────────────────────────────────────────────────────
    st.markdown("**Tabla comparativa de predicciones:**")
    st.dataframe(
        results_df.style.format({
            c: "{:.4f}" for c in config.TARGET_COLS if c in results_df.columns
        }).background_gradient(
            subset=[c for c in config.TARGET_COLS if c in results_df.columns],
            cmap="RdYlGn",
        ),
        use_container_width=True,
    )

    # ── Métricas de rendimiento (ocultable) ───────────────────────────────────
    if show_metrics:
        with st.expander("Métricas de rendimiento de los modelos (datos de prueba)", expanded=False):
            results_path = ROOT / "models" / "results.csv"
            if results_path.exists():
                perf_df = pd.read_csv(results_path)

                for target in config.TARGET_COLS:
                    t_df = perf_df[perf_df["target"] == target].sort_values("rmse")
                    fig = px.bar(
                        t_df,
                        x="modelo",
                        y="rmse",
                        color="modelo",
                        text=t_df["rmse"].round(4),
                        title=f"RMSE en set de prueba — {target}",
                        color_discrete_sequence=list(COLORS.values()),
                    )
                    fig.update_traces(texttemplate="%{text}", textposition="outside")
                    fig.update_layout(showlegend=False, height=320)
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("Ejecuta `02_modeling.ipynb` y guarda `results.csv` para ver las métricas.")
