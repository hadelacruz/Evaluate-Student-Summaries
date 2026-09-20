"""
train_models.py
Script standalone para entrenar y guardar todos los modelos.
Ejecutar con Python 3.11:
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 train_models.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(f"Python: {sys.version}")
import sklearn; print(f"sklearn: {sklearn.__version__}")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_SEED        = 42
TFIDF_MAX_FEATURES = 5000
TARGET_COLS        = ["content", "wording"]

# ── Carga de datos ────────────────────────────────────────────────────────────
print("\n[1/5] Cargando datos...")
summaries = pd.read_csv(DATA_DIR / "summaries_train.csv")
prompts   = pd.read_csv(DATA_DIR / "prompts_train.csv")
df = summaries.merge(prompts[["prompt_id", "prompt_title", "prompt_text"]], on="prompt_id", how="left")
df = df.dropna(subset=["text"] + TARGET_COLS).copy()
df["text_len_words"] = df["text"].str.split().str.len()
df["text_len_chars"] = df["text"].str.len()
print(f"  Dataset: {len(df):,} filas")

# ── Features ──────────────────────────────────────────────────────────────────
print("\n[2/5] Construyendo features...")
X_train_raw, X_test_raw = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

# TF-IDF — fit SOLO en train
vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train_raw["text"]).toarray()
X_test_tfidf  = vectorizer.transform(X_test_raw["text"]).toarray()

# Features numéricas
num_cols = ["text_len_words", "text_len_chars"]
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_raw[num_cols])
X_test_num  = scaler.transform(X_test_raw[num_cols])

X_train = np.hstack([X_train_tfidf, X_train_num])
X_test  = np.hstack([X_test_tfidf,  X_test_num])

print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

# Guardar vectorizer y scaler
joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
joblib.dump(scaler,     MODELS_DIR / "numeric_scaler.pkl")
print("  Vectorizer y scaler guardados.")

# ── Modelos ───────────────────────────────────────────────────────────────────
MODELS = {
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_SEED),
    "Lasso": Lasso(alpha=0.01, random_state=RANDOM_SEED, max_iter=5000),
    "RandomForest": RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.1, max_depth=4, random_state=RANDOM_SEED
    ),
}

# ── Entrenamiento y evaluación ────────────────────────────────────────────────
print("\n[3/5] Entrenando modelos...")
results = []
for model_name, model in MODELS.items():
    for target in TARGET_COLS:
        y_train = X_train_raw[target].values
        y_test  = X_test_raw[target].values

        print(f"  [{model_name}] target={target} ...", end=" ", flush=True)
        model_instance = type(model)(**model.get_params())
        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae  = float(mean_absolute_error(y_test, y_pred))
        r2   = float(r2_score(y_test, y_pred))
        print(f"RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

        pkl_path = MODELS_DIR / f"{model_name}_{target}.pkl"
        joblib.dump(model_instance, pkl_path)

        results.append({
            "modelo": model_name,
            "target": target,
            "rmse": round(rmse, 4),
            "mae":  round(mae, 4),
            "r2":   round(r2, 4),
        })

# ── Guardar resultados ────────────────────────────────────────────────────────
print("\n[4/5] Guardando resultados...")
results_df = pd.DataFrame(results)
results_df.to_csv(MODELS_DIR / "results.csv", index=False)
print(results_df.to_string(index=False))

# ── Verificación final ────────────────────────────────────────────────────────
print("\n[5/5] Verificando modelos guardados...")
for pkl in sorted(MODELS_DIR.glob("*.pkl")):
    obj = joblib.load(pkl)
    print(f"  ✅ {pkl.name}  ({type(obj).__name__})")

print("\n✅ Entrenamiento completo. Reinicia la app Streamlit para cargar los nuevos modelos.")
