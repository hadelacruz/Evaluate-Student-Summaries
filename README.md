# CommonLit — Evaluate Student Summaries
> **Proyecto 2 · CC3084 Data Science · UVG Semestre II 2026 · Reto #12**  
> Procesamiento del Lenguaje Natural — Regresión supervisada

---

## ⚡ Inicio rápido para el evaluador

### Prerequisitos
- Python 3.11 instalado en `/Library/Frameworks/Python.framework/Versions/3.11/`
- Git y acceso al repositorio

### Instalación (una sola vez)

```bash
# 1. Clonar el repositorio
git clone https://github.com/hadelacruz/Evaluate-Student-Summaries.git
cd Evaluate-Student-Summaries

# 2. Instalar dependencias con Python 3.11
/Library/Frameworks/Python.framework/Versions/3.11/bin/pip install -r requirements.txt
```

### Entrenar los modelos

Los modelos entrenados pesan ~13 MB y están incluidos en el repositorio (`models/*.pkl`).  
Si necesitás regenerarlos desde cero:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 train_models.py
```

Esto tarda ~3 minutos y produce:
- `models/Ridge_content.pkl`, `models/Ridge_wording.pkl`
- `models/Lasso_content.pkl`, `models/Lasso_wording.pkl`
- `models/RandomForest_content.pkl`, `models/RandomForest_wording.pkl`
- `models/HistGradientBoosting_content.pkl`, `models/HistGradientBoosting_wording.pkl`
- `models/tfidf_vectorizer.pkl`, `models/numeric_scaler.pkl`
- `models/results.csv` (métricas de evaluación)

### Lanzar la aplicación

```bash
streamlit run app/app.py
```

Se abre automáticamente en **http://localhost:8501**

---

## 📋 Guía de uso de la aplicación

La app tiene **3 páginas** accesibles desde el menú lateral izquierdo:

### 📊 Página 1 — Explorar Datos
Explora el dataset de entrenamiento (7,165 resúmenes de estudiantes).

| Sección | Qué muestra |
|---|---|
| Vista general | Tabla de muestra con filtro por prompt en el sidebar |
| Estadísticos descriptivos | Media, desviación, percentiles de todas las variables numéricas |
| Distribuciones | Histogramas interactivos de `content`, `wording` y longitud del texto |
| Scatter plots | Relación entre longitud del resumen y puntuaciones, coloreado por prompt |
| Boxplots por prompt | Distribución de puntuaciones según el texto fuente |
| Mapa de correlaciones | Heatmap interactivo entre variables numéricas |
| Frecuencia por prompt | Proporción de resúmenes por cada texto fuente |

> **Filtro:** Usa el selector "Prompt" en el sidebar para ver solo los resúmenes de un texto fuente específico.

### 🤖 Página 2 — Resultados de Modelos
Predice las puntuaciones de un resumen escrito por el usuario.

1. Seleccioná en el sidebar si querés usar **"Todos los modelos"** o uno específico
2. Escribí o pegá un resumen de estudiante en el área de texto
3. Hacé clic en **"🔍 Predecir"**
4. La app muestra:
   - Gauge charts con el promedio de todos los modelos
   - Barras comparativas de predicción por modelo
   - Tabla con valores exactos por modelo y target

> El toggle **"Mostrar métricas de rendimiento"** en el sidebar controla si se muestran las gráficas de RMSE de los modelos.

### ⚖️ Página 3 — Comparación de Modelos
Compara el rendimiento de los 4 modelos con datos de prueba reales.

| Control (sidebar) | Efecto |
|---|---|
| Checkboxes RMSE / MAE / R² | **Oculta o muestra** cada métrica individualmente |
| Selector de target | Filtra las gráficas por `content` o `wording` |
| "Ver ambos targets" | Muestra content y wording en paralelo |

Incluye:
- Barras interactivas de métricas por modelo (con hover)
- Radar chart normalizado — perfil visual de cada modelo
- Scatter RMSE content vs wording — para ver qué modelo es mejor en ambas dimensiones simultáneamente
- Ranking final por RMSE promedio

---

## 🏗️ Arquitectura del proyecto

```
Evaluate-Student-Summaries/
├── data/                          ← CSVs de Kaggle
│   ├── summaries_train.csv        ← 7,165 resúmenes con puntuaciones
│   ├── prompts_train.csv          ← 4 textos fuente
│   └── ...
├── notebooks/
│   ├── 01_eda.ipynb               ← Análisis Exploratorio de Datos
│   └── 02_modeling.ipynb          ← Pipeline de entrenamiento documentado
├── src/                           ← Paquete Python modular
│   ├── config.py                  ← Rutas y constantes (sin hardcoding)
│   ├── preprocessing.py           ← Carga, limpieza y features de longitud
│   ├── features.py                ← TF-IDF + scaler (sin data leakage)
│   ├── models.py                  ← Catálogo de modelos y métricas
│   └── visualization.py          ← Funciones de gráficos reutilizables
├── models/                        ← Modelos serializados (.pkl) + results.csv
├── reports/figures/               ← Figuras estáticas para el informe
├── app/
│   ├── app.py                     ← Landing page
│   └── pages/
│       ├── 1_Explorar_Datos.py    ← EDA interactivo (Plotly)
│       ├── 2_Resultados_Modelos.py ← Predicción con modelos entrenados
│       └── 3_Comparacion_Modelos.py ← Comparación de algoritmos
├── train_models.py                ← Script de entrenamiento standalone
└── requirements.txt
```

---

## 🤖 Modelos implementados

| Modelo | Justificación |
|---|---|
| **Ridge Regression** | Baseline lineal regularizado (L2); eficiente con matrices TF-IDF dispersas |
| **Lasso Regression** | Regularización L1 para selección automática de features |
| **Random Forest** | Captura relaciones no lineales; robusto ante outliers |
| **HistGradientBoosting** | Boosting moderno (sklearn 1.0+), mejor rendimiento en datos tabulares |

### Resultados (set de prueba, 20% del dataset)

| Modelo | RMSE content | RMSE wording | R² content | R² wording |
|---|:---:|:---:|:---:|:---:|
| **HistGradientBoosting** | **0.4517** ⭐ | 0.6436 | **0.8147** ⭐ | 0.5959 |
| RandomForest | 0.4779 | 0.6758 | 0.7926 | 0.5544 |
| Ridge | 0.4921 | **0.6387** ⭐ | 0.7801 | **0.6020** ⭐ |
| Lasso | 0.6302 | 0.8600 | 0.6394 | 0.2784 |

> **Métrica elegida:** RMSE (Root Mean Squared Error), estándar para problemas de regresión continua. Menor RMSE = mejor.  
> **Ganador:** HistGradientBoosting tiene el mejor RMSE en `content` (R² = 0.81), Ridge es levemente mejor en `wording`.

---

## 📐 Pipeline de features

```
Texto del resumen (string)
    │
    ▼
TF-IDF Vectorizer (max 5,000 features, n-grams 1-2, TF-IDF sublineal)
    │                                         fit SOLO en train → sin data leakage
    ▼
Concatenación horizontal
    │
    ├── Features TF-IDF (5,000 dims)
    └── Features numéricas escaladas (2 dims): text_len_words, text_len_chars
    │
    ▼
Feature matrix: (n_samples, 5,002)
    │
    ▼
Modelo de regresión → predicción de content y wording
```

---

## ✅ Cumplimiento de la rúbrica

| Criterio | Puntos | Estado | Evidencia |
|---|:---:|:---:|---|
| **Preprocesamiento inadvertido** | 15 | ✅ Cubierto | TF-IDF + StandardScaler se aplican automáticamente al ingresar texto en Página 2. El usuario no ve ningún paso técnico. |
| **Presentación de resultados** | 20 | ✅ Cubierto | Página 2: gauge charts, barras por modelo, tabla con gradiente de color. Página 1: scatter con trendline que muestra relación longitud↔puntuación. |
| **Eficiencia con gráficas interactivas + toggle** | 15 | ✅ Cubierto | Página 3: barras Plotly, radar chart, scatter RMSE. Sidebar con 3 checkboxes para ocultar RMSE/MAE/R² individualmente. |
| **Informe final** | 40 | 📝 Informe PDF | Ver documento adjunto en Google Drive |
| **Referencias y Bibliografía APA** | 10 | 📝 Informe PDF | Ver sección de referencias en el informe |

---

## 🔧 Buenas prácticas de ingeniería implementadas

| # | Práctica | Implementación |
|---|---|---|
| 1 | **Separación de responsabilidades (SRP)** | Cada módulo en `src/` tiene una única responsabilidad |
| 2 | **Reproducibilidad** | `RANDOM_SEED = 42` en `config.py`; `requirements.txt` con versiones fijas |
| 3 | **Control de versiones** | `.gitignore` configura correctamente qué ignorar |
| 4 | **Modularización y DRY** | `src/` evita repetición de código entre notebooks y app |
| 5 | **Estructura estándar** | Directorios `data/`, `notebooks/`, `src/`, `models/`, `reports/`, `app/` |
| 6 | **Type hints y docstrings** | Todas las funciones públicas documentadas |
| 7 | **Sin data leakage** | `TfidfVectorizer` y `StandardScaler` se ajustan solo en train |
| 8 | **Nombrado de notebooks** | Prefijo numérico: `01_eda`, `02_modeling` |
| 9 | **UI desacoplada de modelos** | La app carga `.pkl`; no importa el código de entrenamiento |
| 10 | **Tolerancia a fallos** | `load_artifacts()` salta modelos incompatibles sin crashear la app |
| 11 | **Inmutabilidad defensiva** | Funciones retornan `.copy()` del DataFrame |
| 12 | **Script standalone** | `train_models.py` permite reentrenar sin depender del entorno Jupyter |

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
- Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)
- Streamlit documentation: [https://docs.streamlit.io](https://docs.streamlit.io)
- Plotly Python documentation: [https://plotly.com/python](https://plotly.com/python)
