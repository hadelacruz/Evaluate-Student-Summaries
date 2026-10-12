"""
tune_models.py
Ajuste de hiperparámetros y comparación de tiempos de procesamiento.

- Usa la misma partición 80/20 (random_state=42) y el mismo preprocesamiento
  que train_models.py (TF-IDF 5,000 términos, unigramas y bigramas, sublinear_tf,
  más text_len_words y text_len_chars escaladas).
- Busca hiperparámetros con GridSearchCV (KFold de 3 particiones) SOLO sobre el
  conjunto de entrenamiento. El TF-IDF y el escalador se ajustan dentro de cada
  partición (Pipeline), para que la validación cruzada no tenga fuga de información.
- La configuración original de train_models.py está incluida en cada rejilla,
  para comparar contra ella.
- Mide el tiempo de entrenamiento y de predicción de la configuración original
  y de la mejor configuración, y el RMSE de entrenamiento y prueba.

Resultados:
    models/tuning_results.csv     resumen por modelo y variable objetivo
    models/tuning_cv_details.csv  RMSE de validación cruzada de cada configuración

Por defecto NO modifica los modelos que usa la app.

Ejecutar:
    python tune_models.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RANDOM_SEED = 42
TARGET_COLS = ["content", "wording"]
NUM_COLS = ["text_len_words", "text_len_chars"]

# ── Datos (igual que train_models.py) ─────────────────────────────────────────
summaries = pd.read_csv(DATA_DIR / "summaries_train.csv")
prompts = pd.read_csv(DATA_DIR / "prompts_train.csv")
df = summaries.merge(prompts[["prompt_id", "prompt_title", "prompt_text"]], on="prompt_id", how="left")
df = df.dropna(subset=["text"] + TARGET_COLS).copy()
df["text_len_words"] = df["text"].str.split().str.len()
df["text_len_chars"] = df["text"].str.len()
train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
print(f"Train: {len(train_df):,} | Test: {len(test_df):,}", flush=True)


def make_pipeline(model):
    """TF-IDF + escalado de longitudes + modelo, igual que train_models.py."""
    features = ColumnTransformer(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True), "text"),
            ("num", StandardScaler(), NUM_COLS),
        ],
        sparse_threshold=0,  # matriz densa, como en train_models.py
    )
    return Pipeline([("features", features), ("model", model)])


# ── Rejillas de búsqueda (la primera configuración es la original) ────────────
SEARCHES = {
    "Ridge": (
        Ridge(random_state=RANDOM_SEED),
        {"model__alpha": [1.0, 0.1, 0.3, 3.0, 10.0]},
        {"model__alpha": 1.0},
    ),
    "Lasso": (
        Lasso(random_state=RANDOM_SEED, max_iter=5000),
        {"model__alpha": [0.01, 0.0003, 0.001, 0.003]},
        {"model__alpha": 0.01},
    ),
    "RandomForest": (
        RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        [
            {"model__max_features": [1.0], "model__max_depth": [10]},
            {"model__max_features": [0.3], "model__max_depth": [10, 20]},
        ],
        {"model__max_features": 1.0, "model__max_depth": 10},
    ),
    "HistGradientBoosting": (
        HistGradientBoostingRegressor(learning_rate=0.1, random_state=RANDOM_SEED),
        {"model__max_iter": [100, 300], "model__max_depth": [4, 8]},
        {"model__max_iter": 100, "model__max_depth": 4},
    ),
}


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def fit_and_time(pipe, X_tr, y_tr, X_te, y_te):
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_te = pipe.predict(X_te)
    pred_s = time.perf_counter() - t0
    pred_tr = pipe.predict(X_tr)
    return {
        "fit_s": fit_s,
        "predict_s": pred_s,
        "ms_por_resumen": 1000 * pred_s / len(X_te),
        "rmse_train": rmse(y_tr, pred_tr),
        "rmse_test": rmse(y_te, pred_te),
        "mae_test": float(mean_absolute_error(y_te, pred_te)),
        "r2_test": float(r2_score(y_te, pred_te)),
    }


def fmt_params(params):
    return ", ".join(f"{k.replace('model__', '')}={v}" for k, v in params.items())


cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
X_tr, X_te = train_df[["text"] + NUM_COLS], test_df[["text"] + NUM_COLS]
summary_rows, detail_rows = [], []

for name, (model, grid, base_params) in SEARCHES.items():
    for target in TARGET_COLS:
        y_tr, y_te = train_df[target].values, test_df[target].values
        print(f"\n[{name}] target={target}", flush=True)

        # 1) Búsqueda con validación cruzada sobre entrenamiento
        search = GridSearchCV(
            make_pipeline(model), grid, cv=cv,
            scoring="neg_root_mean_squared_error", n_jobs=1, refit=False,
        )
        t0 = time.perf_counter()
        search.fit(X_tr, y_tr)
        search_s = time.perf_counter() - t0
        res = search.cv_results_
        for p, m, s in zip(res["params"], res["mean_test_score"], res["std_test_score"]):
            detail_rows.append({"modelo": name, "target": target, "parametros": fmt_params(p),
                                "cv_rmse": -m, "cv_rmse_std": s,
                                "original": p == base_params})
        best_params = search.best_params_
        cv_best = -search.best_score_
        cv_base = -res["mean_test_score"][res["params"].index(base_params)]
        print(f"  búsqueda: {search_s:.0f} s | mejor: {fmt_params(best_params)} "
              f"(CV RMSE {cv_best:.4f}; original {cv_base:.4f})", flush=True)

        # 2) Configuración original y mejor configuración, entrenadas en todo el train
        base = fit_and_time(make_pipeline(model).set_params(**base_params), X_tr, y_tr, X_te, y_te)
        if best_params == base_params:
            best = base
        else:
            best = fit_and_time(make_pipeline(model).set_params(**best_params), X_tr, y_tr, X_te, y_te)
        print(f"  original: test RMSE {base['rmse_test']:.4f} | ajustado: test RMSE {best['rmse_test']:.4f}", flush=True)

        row = {"modelo": name, "target": target,
               "params_original": fmt_params(base_params), "params_ajustados": fmt_params(best_params),
               "n_configuraciones": len(res["params"]), "tiempo_busqueda_s": search_s,
               "cv_rmse_original": cv_base, "cv_rmse_ajustado": cv_best}
        row.update({f"{k}_original": v for k, v in base.items()})
        row.update({f"{k}_ajustado": v for k, v in best.items()})
        summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
details = pd.DataFrame(detail_rows)
summary.round(4).to_csv(MODELS_DIR / "tuning_results.csv", index=False)
details.round(4).to_csv(MODELS_DIR / "tuning_cv_details.csv", index=False)
print("\nGuardado: models/tuning_results.csv y models/tuning_cv_details.csv")
print(summary[["modelo", "target", "params_ajustados", "rmse_test_original", "rmse_test_ajustado",
               "fit_s_ajustado", "ms_por_resumen_ajustado"]].round(4).to_string(index=False))
