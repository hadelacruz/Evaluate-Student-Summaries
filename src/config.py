"""
src/config.py
Constantes globales y rutas del proyecto.

Todas las rutas se calculan relativamente a este archivo para que el proyecto
funcione independientemente de dónde esté instalado.
"""

from pathlib import Path

# ── Raíz del proyecto ─────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# ── Directorios principales ───────────────────────────────────────────────────
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"

# ── Archivos de datos ─────────────────────────────────────────────────────────
SUMMARIES_TRAIN: Path = DATA_DIR / "summaries_train.csv"
SUMMARIES_TEST: Path = DATA_DIR / "summaries_test.csv"
PROMPTS_TRAIN: Path = DATA_DIR / "prompts_train.csv"
PROMPTS_TEST: Path = DATA_DIR / "prompts_test.csv"
SAMPLE_SUBMISSION: Path = DATA_DIR / "sample_submission.csv"

# ── Columnas del dataset ──────────────────────────────────────────────────────
STUDENT_ID_COL: str = "student_id"
PROMPT_ID_COL: str = "prompt_id"
TEXT_COL: str = "text"
CONTENT_COL: str = "content"
WORDING_COL: str = "wording"
PROMPT_TEXT_COL: str = "prompt_text"
PROMPT_TITLE_COL: str = "prompt_title"
PROMPT_QUESTION_COL: str = "prompt_question"

# ── Features derivadas ────────────────────────────────────────────────────────
TEXT_LEN_WORDS_COL: str = "text_len_words"
TEXT_LEN_CHARS_COL: str = "text_len_chars"
LENGTH_RATIO_COL: str = "length_ratio_vs_prompt"

# ── Reproducibilidad ──────────────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ── Modelos a evaluar ─────────────────────────────────────────────────────────
TARGET_COLS: list[str] = [CONTENT_COL, WORDING_COL]

# ── TF-IDF ───────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES: int = 5_000
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)
