"""
app/pages/3_Comparacion_Modelos.py
Página 3: Comparación interactiva del rendimiento con Plotly.
Permite ocultar/mostrar métricas con checkboxes en el sidebar.
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
from plotly.subplots import make_subplots
import streamlit as st

from src import config

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Comparación de Modelos — CommonLit", page_icon="⚖️", layout="wide")

COLORS = {
    "Ridge": "#4C72B0",
    "Lasso": "#DD8452",
    "RandomForest": "#55A868",
    "HistGradientBoosting": "#C44E52",
}

st.title("⚖️ Comparación de Modelos")
st.caption("Compara el rendimiento de los 4 algoritmos con gráficas interactivas. Usa el sidebar para filtrar qué información ver.")
st.divider()


# ── Carga de resultados ───────────────────────────────────────────────────────
RESULTS_CSV = ROOT / "models" / "results.csv"


@st.cache_data(show_spinner="Cargando resultados...")
def load_results() -> pd.DataFrame:
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    # Datos de demostración hasta que se corra el notebook
    return pd.DataFrame({
        "modelo": ["Ridge", "Lasso", "RandomForest", "HistGradientBoosting"] * 2,
        "target": ["content"] * 4 + ["wording"] * 4,
        "rmse": [0.52, 0.58, 0.48, 0.45, 0.61, 0.67, 0.55, 0.52],
        "mae":  [0.41, 0.46, 0.38, 0.36, 0.49, 0.54, 0.43, 0.41],
        "r2":   [0.73, 0.66, 0.78, 0.81, 0.63, 0.55, 0.69, 0.72],
        "_demo": [True] * 8,
    })


results_df = load_results()
is_demo = "_demo" in results_df.columns

if is_demo:
    st.info("Mostrando datos de **demostración**. Ejecuta `notebooks/02_modeling.ipynb` para ver los valores reales.")

# ── Sidebar — controles ───────────────────────────────────────────────────────
st.sidebar.header("Controles de visualización")
show_rmse = st.sidebar.checkbox("Mostrar RMSE", value=True)
show_mae  = st.sidebar.checkbox("Mostrar MAE",  value=True)
show_r2   = st.sidebar.checkbox("Mostrar R²",   value=True)
st.sidebar.divider()
selected_target = st.sidebar.selectbox("Target a visualizar", config.TARGET_COLS, index=0)
show_both = st.sidebar.checkbox("Ver ambos targets simultáneamente", value=False)

# ─────────────────────────────────────────────────────────────────────────────
# Tabla de métricas con colores condicionales
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Tabla de métricas por modelo y target", expanded=True):
    display_cols = ["modelo", "target"]
    if show_rmse: display_cols.append("rmse")
    if show_mae:  display_cols.append("mae")
    if show_r2:   display_cols.append("r2")

    clean_df = results_df[[c for c in display_cols if c in results_df.columns]]
    styled = clean_df.sort_values(["target", "rmse"] if "rmse" in display_cols else ["target"]).style

    if "rmse" in display_cols:
        styled = styled.background_gradient(
            subset=[c for c in ["rmse", "mae"] if c in display_cols], cmap="RdYlGn_r"
        )
    if "r2" in display_cols:
        styled = styled.background_gradient(subset=["r2"], cmap="RdYlGn")

    fmt = {c: "{:.4f}" for c in ["rmse", "mae", "r2"] if c in display_cols}
    st.dataframe(styled.format(fmt), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Gráficas de métricas interactivas
# ─────────────────────────────────────────────────────────────────────────────
if not (show_rmse or show_mae or show_r2):
    st.warning("Selecciona al menos una métrica en el panel lateral.")
else:
    targets_to_show = config.TARGET_COLS if show_both else [selected_target]

    for target in targets_to_show:
        subset = results_df[results_df["target"] == target].copy()
        colors_list = [COLORS.get(m, "#888") for m in subset["modelo"]]

        st.markdown(f"### Target: `{target}`")

        metrics_cols = []
        if show_rmse: metrics_cols.append(("rmse", "RMSE", True, "Menor es mejor"))
        if show_mae:  metrics_cols.append(("mae",  "MAE",  True, "Menor es mejor"))
        if show_r2:   metrics_cols.append(("r2",   "R²",   False, "Mayor es mejor"))

        n = len(metrics_cols)
        cols = st.columns(n)

        for col, (metric, label, ascending, hint) in zip(cols, metrics_cols):
            with col:
                sorted_df = subset.sort_values(metric, ascending=ascending)
                fig = go.Figure()

                for i, row in sorted_df.iterrows():
                    c = COLORS.get(row["modelo"], "#888")
                    fig.add_trace(go.Bar(
                        name=row["modelo"],
                        x=[row["modelo"]],
                        y=[row[metric]],
                        marker_color=c,
                        marker_line_color="white",
                        marker_line_width=1.5,
                        text=[f"{row[metric]:.4f}"],
                        textposition="outside",
                        hovertemplate=(
                            f"<b>{row['modelo']}</b><br>"
                            f"{label}: <b>%{{y:.4f}}</b><extra></extra>"
                        ),
                    ))

                fig.update_layout(
                    title=f"{label} — {target}<br><sup>{hint}</sup>",
                    yaxis_title=label,
                    showlegend=False,
                    height=370,
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="rgba(0,0,0,0.1)"),
                )
                st.plotly_chart(fig, width="stretch")

        if len(targets_to_show) > 1:
            st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Radar chart comparativo
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Radar comparativo — perfil de cada modelo", expanded=False):
    st.markdown("""
    El radar muestra el **perfil de cada modelo** normalizado entre 0 y 1.  
    Para RMSE y MAE se invierte la escala (menor = mejor → más hacia afuera).
    """)

    radar_target = st.selectbox("Target para radar:", config.TARGET_COLS, key="radar_target")
    r_df = results_df[results_df["target"] == radar_target].copy()

    # Normalizar: RMSE y MAE invertidos (1 - norm), R² directo
    for col in ["rmse", "mae"]:
        r_df[col + "_norm"] = 1 - (r_df[col] - r_df[col].min()) / (r_df[col].max() - r_df[col].min() + 1e-9)
    r_df["r2_norm"] = (r_df["r2"] - r_df["r2"].min()) / (r_df["r2"].max() - r_df["r2"].min() + 1e-9)

    categories = ["RMSE (inv)", "MAE (inv)", "R²"]
    fig = go.Figure()

    for _, row in r_df.iterrows():
        values = [row["rmse_norm"], row["mae_norm"], row["r2_norm"]]
        values.append(values[0])  # cerrar el polígono
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["modelo"],
            line_color=COLORS.get(row["modelo"], "#888"),
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=420,
        title=f"Perfil normalizado de modelos — {radar_target}",
    )
    st.plotly_chart(fig, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# Comparación simultánea: content vs wording
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Analisis comparativo — ambos targets", expanded=False):
    if show_rmse:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f"RMSE — {t}" for t in config.TARGET_COLS])

        for col_idx, target in enumerate(config.TARGET_COLS, start=1):
            t_df = results_df[results_df["target"] == target].sort_values("rmse")
            for i, row in t_df.iterrows():
                c = COLORS.get(row["modelo"], "#888")
                fig.add_trace(go.Bar(
                    name=row["modelo"],
                    x=[row["modelo"]],
                    y=[row["rmse"]],
                    marker_color=c,
                    text=[f"{row['rmse']:.3f}"],
                    textposition="outside",
                    showlegend=(col_idx == 1),
                    hovertemplate=f"<b>{row['modelo']}</b><br>RMSE: %{{y:.4f}}<extra></extra>",
                ), row=1, col=col_idx)

        fig.update_layout(
            title="RMSE comparativo — content vs wording",
            barmode="group",
            height=400,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, width="stretch")

    # Scatter RMSE content vs wording por modelo
    pivot_rmse = results_df.pivot(index="modelo", columns="target", values="rmse").reset_index()
    if all(t in pivot_rmse.columns for t in config.TARGET_COLS):
        fig = px.scatter(
            pivot_rmse,
            x=config.TARGET_COLS[0],
            y=config.TARGET_COLS[1],
            text="modelo",
            color="modelo",
            color_discrete_map=COLORS,
            title="RMSE content vs RMSE wording por modelo",
            labels={config.TARGET_COLS[0]: f"RMSE {config.TARGET_COLS[0]}",
                    config.TARGET_COLS[1]: f"RMSE {config.TARGET_COLS[1]}"},
            size_max=20,
        )
        fig.update_traces(textposition="top center", marker_size=12)
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    # Ranking final
    st.markdown("**Ranking final — menor RMSE promedio entre targets:**")
    pivot = results_df.pivot(index="modelo", columns="target", values="rmse")
    pivot["RMSE promedio"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("RMSE promedio").reset_index()

    fig = px.bar(
        pivot,
        x="modelo",
        y="RMSE promedio",
        color="modelo",
        text=pivot["RMSE promedio"].round(4),
        color_discrete_map=COLORS,
        title="Ranking de modelos — RMSE promedio (content + wording)",
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(showlegend=False, height=360)
    st.plotly_chart(fig, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# Figuras estáticas del notebook
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Figuras estáticas generadas en el notebook", expanded=False):
    figures = {
        "RMSE por modelo": ROOT / "reports" / "figures" / "model_comparison_rmse.png",
        "R² por modelo": ROOT / "reports" / "figures" / "model_comparison_r2.png",
        "Real vs Predicho": ROOT / "reports" / "figures" / "actual_vs_predicted.png",
    }
    found_any = False
    img_cols = st.columns(len([p for p in figures.values() if p.exists()]) or 1)
    col_idx = 0
    for title, path in figures.items():
        if path.exists():
            found_any = True
            with img_cols[col_idx % len(img_cols)]:
                st.markdown(f"**{title}**")
                st.image(str(path), use_container_width=True)
            col_idx += 1
    if not found_any:
        st.info("Ejecuta `notebooks/02_modeling.ipynb` para generar las figuras estáticas en `reports/figures/`.")
