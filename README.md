# 📚 CommonLit — Evaluate Student Summaries

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Kaggle](https://img.shields.io/badge/Competencia-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries)
[![UVG](https://img.shields.io/badge/Universidad-UVG-003580)](https://www.uvg.edu.gt/)

> **Proyecto 2 · CC3084 Data Science · UVG Semestre II 2026**  
> **Reto #12 — Procesamiento del Lenguaje Natural**

---

## 📖 Descripción del problema

La capacidad de resumir un texto es fundamental en la formación académica. Evaluar la calidad de esos resúmenes de forma manual a gran escala es costoso y poco escalable.

Este proyecto aborda el reto **CommonLit: Evaluate Student Summaries** de Kaggle: construir modelos de Machine Learning capaces de **predecir automáticamente la calidad de resúmenes escritos por estudiantes** de los grados 3 a 12. Los modelos predicen dos puntuaciones continuas:

| Target | Descripción |
|---|---|
| `content` | Qué tan bien el resumen representa la idea principal y los detalles del texto fuente |
| `wording` | Calidad del lenguaje: claridad, precisión y fluidez |

Es un problema de **regresión** con técnicas de **NLP** (TF-IDF, extracción de features de texto).

---

## 🏗️ Arquitectura del repositorio

```
Evaluate-Student-Summaries/
│
├── 📁 data/                          # Datos del dataset de Kaggle
│   ├── summaries_train.csv           # 7,165 resúmenes etiquetados
│   ├── prompts_train.csv             # 4 textos fuente (prompts)
│   ├── summaries_test.csv            # Set de prueba (sin etiquetas)
│   ├── prompts_test.csv              # Prompts del set de prueba
│   └── sample_submission.csv         # Formato de envío a Kaggle
│
├── 📁 notebooks/                     # Notebooks numerados por fase
│   ├── 01_eda.ipynb                  # Análisis Exploratorio de Datos (EDA)
│   └── 02_modeling.ipynb             # Entrenamiento y evaluación de modelos
│
├── 📁 src/                           # Paquete Python reutilizable
│   ├── __init__.py
│   ├── config.py                     # Rutas y constantes globales
│   ├── preprocessing.py              # Carga, validación y limpieza de datos
│   ├── features.py                   # Feature engineering NLP (TF-IDF, longitudes)
│   ├── models.py                     # Entrenamiento, evaluación y persistencia
│   └── visualization.py             # Funciones de gráficos reutilizables
│
├── 📁 models/                        # Modelos serializados (.pkl)
│   └── .gitkeep
│
├── 📁 reports/figures/               # Figuras estáticas generadas por notebooks
│   └── .gitkeep
│
├── 📁 app/                           # Aplicación Streamlit
│   ├── app.py                        # Página de inicio
│   └── pages/
│       ├── 1_Explorar_Datos.py       # EDA interactivo con filtros
│       ├── 2_Resultados_Modelos.py   # Predicción con texto libre
│       └── 3_Comparacion_Modelos.py  # Comparativa interactiva de algoritmos
│
├── .gitignore
├── requirements.txt
└── README.md
```

### Flujo de datos

```
data/CSVs
    │
    ▼
src/preprocessing.py   ─── load_and_merge(), add_length_features()
    │
    ▼
src/features.py        ─── build_tfidf_matrix(), build_numeric_features()
    │
    ▼
src/models.py          ─── train_evaluate_all(), save_model()
    │
    ▼
models/*.pkl           ◄── persisten los modelos entrenados
    │
    ▼
app/ (Streamlit)       ─── carga modelos y predice sobre texto nuevo
```

---

## ⚙️ Instalación y configuración

### Requisitos

- Python 3.10 o superior
- pip

### 1. Clonar el repositorio

```bash
git clone https://github.com/<tu-usuario>/Evaluate-Student-Summaries.git
cd Evaluate-Student-Summaries
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar datos de Kaggle

Los datos son parte de la competencia [CommonLit - Evaluate Student Summaries](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/data).

```bash
# Con la CLI de Kaggle (requiere API key configurada):
kaggle competitions download -c commonlit-evaluate-student-summaries
unzip commonlit-evaluate-student-summaries.zip -d data/
```

> **Nota:** Si descargás manualmente, coloca los 5 CSVs directamente en `data/`.

---

## 🚀 Cómo ejecutar

### Fase 1 — Análisis Exploratorio

```bash
jupyter notebook notebooks/01_eda.ipynb
```

El notebook incluye:
- Descripción de los datos y variables
- Limpieza y preprocesamiento
- Análisis estadístico descriptivo
- Histogramas, boxplots, correlaciones y diagramas de dispersión
- Hallazgos y conclusiones del EDA

### Fase 2 — Entrenamiento de modelos

```bash
jupyter notebook notebooks/02_modeling.ipynb
```

Este notebook:
1. Construye la matriz de features (TF-IDF + features numéricas)
2. Entrena 4 algoritmos: Ridge, Lasso, RandomForest, GradientBoosting
3. Evalúa con métricas RMSE, MAE y R²
4. Genera figuras estáticas en `reports/figures/`
5. Guarda los modelos en `models/*.pkl`

### Fase 3 — Aplicación interactiva

```bash
streamlit run app/app.py
```

La app se abre automáticamente en `http://localhost:8501`.

---

## 📦 Módulos Python (`src/`)

| Módulo | Funciones principales | Descripción |
|---|---|---|
| `config.py` | — | Rutas y constantes centralizadas |
| `preprocessing.py` | `load_and_merge()`, `check_quality()`, `add_length_features()` | Carga, validación y feature engineering básico |
| `features.py` | `build_tfidf_matrix()`, `build_feature_matrix()`, `scale_numeric_features()` | Feature engineering NLP, evita data leakage |
| `models.py` | `train_evaluate_all()`, `save_model()`, `load_model()` | Entrenamiento, métricas y persistencia joblib |
| `visualization.py` | `plot_score_distributions()`, `plot_model_comparison()` | Gráficos reutilizables (retornan `Figure`) |

### Uso rápido desde Python

```python
from src.preprocessing import load_and_merge, add_length_features
from src.features import build_feature_matrix
from src.models import train_evaluate_all, split_data
from src import config

# Cargar datos
df = load_and_merge("train")
df = add_length_features(df)

# Features
X, _ = build_feature_matrix(df)
y = df[config.CONTENT_COL].values

# Entrenar y evaluar
X_train, X_test, y_train, y_test = split_data(X, y)
results = train_evaluate_all(X_train, y_train, X_test, y_test, target_name="content")
print(results)
```

---

## 🤖 Modelos de ML

| Algoritmo | Tipo | Por qué se eligió |
|---|---|---|
| **Ridge Regression** | Lineal regularizado (L2) | Baseline robusto, maneja bien la alta dimensionalidad de TF-IDF |
| **Lasso Regression** | Lineal regularizado (L1) | Selección automática de features, interpretable |
| **Random Forest** | Ensemble (bagging) | Captura no-linealidades, robusto ante outliers |
| **Gradient Boosting** | Ensemble (boosting) | Alta precisión en tabular, estado del arte en regresión |

**Métricas de evaluación:** RMSE (métrica oficial de Kaggle), MAE y R².

---

## 🛠️ Buenas prácticas de ingeniería de software aplicadas

| # | Práctica | Descripción | Artefacto |
|---|---|---|---|
| 1 | **Separación de responsabilidades (SRP)** | Cada módulo tiene una responsabilidad única: carga, features, modelos, visualización | `src/*.py` |
| 2 | **Reproducibilidad** | Dependencias pinneadas con rangos de versión; semilla aleatoria centralizada en `config.py` | `requirements.txt`, `src/config.py` |
| 3 | **Control de versiones limpio** | `.gitignore` específico para DS: excluye `__pycache__`, `.ipynb_checkpoints`, `.env` | `.gitignore` |
| 4 | **Modularización y DRY** | Funciones de visualización definidas una vez en `src/visualization.py`, usadas en notebooks y la app | `src/visualization.py` |
| 5 | **Estructura de directorios estándar** | Convención `notebooks/`, `src/`, `models/`, `reports/`, `app/` | Directorio raíz |
| 6 | **Type hints y docstrings** | Toda función pública documenta args, returns y raises; type hints para IDE autocompletion | `src/*.py` |
| 7 | **Trazabilidad de datos** | Datos crudos en `data/`, modelos en `models/`, figuras en `reports/figures/` | Directorio raíz |
| 8 | **Nombrado de notebooks** | Prefijo numérico para orden de ejecución explícito (`01_eda`, `02_modeling`) | `notebooks/` |
| 9 | **Sin data leakage** | Vectorizer y scaler se ajustan solo sobre train; se aplican (transform) sobre test/nuevos datos | `src/features.py` |
| 10 | **UI desacoplada de la lógica** | App Streamlit carga modelos desde disco sin reproducir lógica de entrenamiento | `app/pages/` |
| 11 | **Logging** | Módulos usan `logging` estándar de Python en lugar de `print` | `src/preprocessing.py`, `src/models.py` |
| 12 | **Inmutabilidad defensiva** | Funciones de preprocessing devuelven `.copy()` del DataFrame, sin mutar el original | `src/preprocessing.py`, `src/features.py` |

---

## 👥 Autores

| GitHub | Contribución |
|---|---|
| [@hadelacruz](https://github.com/hadelacruz) | — |
| [@djuarez-2017510](https://github.com/djuarez-2017510) | — |
| [@GerardoFdez7](https://github.com/GerardoFdez7) | — |
| [@jruiz002](https://github.com/jruiz002) | — |
| [@nicollegordillo](https://github.com/nicollegordillo) | — |

---

## 📚 Referencias

- Kaggle Competition: [CommonLit — Evaluate Student Summaries](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries)
- [Kaggle Code of Conduct](https://www.kaggle.com/general/33266)
- [Google Colab + GitHub integration](https://medium.com/analytics-vidhya/how-to-use-google-colab-with-github-via-google-drive-68efb23a42d)
- Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)
- Streamlit documentation: [https://docs.streamlit.io](https://docs.streamlit.io)
