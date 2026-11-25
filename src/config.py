"""Configuración central del proyecto.

Todas las rutas, clases e hiperparámetros globales viven aquí para que los
scripts de cada fase de CRISP-DM usen exactamente los mismos valores.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_KAGGLE = RAW_DIR / "kaggle"      # Fruit Quality Classification (Kaggle)
RAW_PROPIAS = RAW_DIR / "propias"    # Fotos tomadas por el grupo (Google Drive)
INTERIM_DIR = DATA_DIR / "interim"   # Recortes individuales de mixed_quality
PROCESSED_DIR = DATA_DIR / "processed"

INDEX_CSV = PROCESSED_DIR / "index.csv"          # Todas las imágenes + etiquetas
SPLITS_CSV = PROCESSED_DIR / "splits.csv"        # index + columna split
FEATURES_NPZ = PROCESSED_DIR / "features.npz"    # Matriz X de características
ANOTACIONES_CSV = DATA_DIR / "anotaciones_propias.csv"  # Etiquetado manual
PROVENANCE_CSV = PROCESSED_DIR / "procedencia.csv"      # Marcas de imágenes generadas con IA
EXCLUDE_AI_GENERATED = True   # excluir del entrenamiento las imágenes con marca de IA

EXPERIMENTS_DIR = ROOT / "experiments"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
FIGURES_DIR = ROOT / "docs" / "figuras"

# Carpeta compartida de Drive con las fotos propias del grupo
DRIVE_FOLDER_ID = "1RpUBvQNExYSoU-HKXw2mgZkJHMnE57Wm"
KAGGLE_DATASET = "ryandpark/fruit-quality-classification"

# --------------------------------------------------------------------------
# Etiquetas
# --------------------------------------------------------------------------
# Clases de calidad (definición operativa en docs/datos.md)
QUALITY_CLASSES = ["good", "regular", "bad"]
# Palabras clave que aparecen en los nombres de carpeta -> clase
QUALITY_KEYWORDS = {
    "good": "good",
    "buena": "good",
    "regular": "regular",
    "medium": "regular",
    "bad": "bad",
    "mala": "bad",
    "rotten": "bad",
    "mixed": "mixed",   # imágenes con varias frutas: se segmentan aparte
}
SIZE_CLASSES = ["pequeno", "mediano", "grande"]
# Nombres de fruta equivalentes (errores de digitación y español -> inglés)
FRUIT_ALIASES = {
    "orage": "orange", "naranja": "orange", "manzana": "apple",
    "banano": "banana", "platano": "banana", "limon": "lime", "guayaba": "guava",
    "granada": "pomegranate", "lemon": "lime",
}
# Frutas reconocibles en el NOMBRE del archivo cuando la foto está suelta en la
# carpeta de calidad (p. ej. Bad_Quality_Fruits/grupoAPO_granadilla_mala_01.jpg).
# La granadilla solo existe en las fotos propias (no está en Kaggle).
KNOWN_FRUITS = {"apple", "banana", "guava", "lime", "orange", "pomegranate", "granadilla"}
# Máximo de imágenes por carpeta de Kaggle (Pomegranate_Good trae 5.940, el
# resto ~1.100). Se submuestrea de forma reproducible para no desbalancear.
KAGGLE_MAX_PER_FOLDER = 1200

# --------------------------------------------------------------------------
# Preprocesamiento
# --------------------------------------------------------------------------
MAX_SIDE = 512        # las fotos de celular (4000 px) se reducen a este lado mayor
IMG_SIZE = 128        # tamaño de entrada de la CNN y de los recortes procesados
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --------------------------------------------------------------------------
# Experimentos
# --------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
CV_FOLDS = 5
DUP_SIM_THRESHOLD = 0.95   # coseno entre miniaturas 32x32 para considerar dos fotos «la misma fruta»
