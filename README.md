# CommonLit - Evaluate Student Summaries

Proyecto 2.

Reto seleccionado: **#12 - CommonLit: Evaluar resúmenes de estudiantes** (Procesamiento del Lenguaje Natural), basado en la competencia de Kaggle [CommonLit - Evaluate Student Summaries](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/data).

## Descripción

El objetivo de la competencia es evaluar la calidad de resúmenes escritos por estudiantes de los grados 3 a 12, midiendo qué tan bien representan la idea principal y los detalles de un texto fuente (`content`), así como la claridad, precisión y fluidez del lenguaje usado (`wording`).

Este repositorio contiene el análisis exploratorio de datos (EDA) realizado sobre el conjunto de entrenamiento de la competencia, como parte de la primera entrega del Proyecto 2.

## Estructura del repositorio

```
├── analisis_exploratorio.ipynb   # Notebook con el informe de análisis exploratorio
├── data/
│   ├── summaries_train.csv       # Resúmenes de entrenamiento (texto, content, wording)
│   ├── prompts_train.csv         # Textos fuente (prompts) de entrenamiento
│   ├── summaries_test.csv        # Ejemplo de formato del set de prueba (placeholder)
│   ├── prompts_test.csv          # Ejemplo de formato de prompts de prueba (placeholder)
│   └── sample_submission.csv     # Formato de envío de la competencia
└── README.md
```

## Contenido del notebook

1. Situación problemática
2. Problema científico
3. Objetivos
4. Descripción de los datos (y limpieza/preprocesamiento)
5. Análisis Exploratorio de Datos (EDA)
6. Hallazgos y conclusiones

## Cómo ejecutarlo

```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook analisis_exploratorio.ipynb
```

## Autores

- [@hadelacruz](https://github.com/hadelacruz)
- [@djuarez-2017510](https://github.com/djuarez-2017510)
- [@GerardoFdez7](https://github.com/GerardoFdez7)
- [@jruiz002](https://github.com/jruiz002)
- [@nicollegordillo](https://github.com/nicollegordillo)
