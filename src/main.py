"""Punto de entrada: ejecuta cada fase del pipeline CRISP-DM.

Uso (desde la raíz del repo):
    python -m src.main download      # baja Kaggle + fotos propias de Drive
    python -m src.main convert       # HEIC -> JPG, quita metadatos (GPS)
    python -m src.main provenance    # marca imágenes generadas con IA (C2PA)
    python -m src.main mixed         # recorta frutas de mixed_quality
    python -m src.main index         # construye data/processed/index.csv
    python -m src.main preprocess    # segmenta, recorta, tamaño, splits
    python -m src.main features      # matriz de características
    python -m src.main feature_space # PCA/LDA del espacio de características
    python -m src.main train         # modelos clásicos (GridSearchCV)
    python -m src.main train_cnn     # CNN desde cero (una configuración)
    python -m src.main search_cnn    # búsqueda de hiperparámetros de la CNN
    python -m src.main evaluate      # métricas de calidad en test + figuras
    python -m src.main intervals     # IC 95 % y prueba de McNemar entre modelos
    python -m src.main evaluate_size # fiabilidad de la estimación de tamaño
    python -m src.main all           # provenance -> evaluate_size (sin descargar)
"""
from __future__ import annotations

import sys

STEPS = ["download", "convert", "provenance", "mixed", "index", "preprocess", "features",
         "feature_space", "train", "train_cnn", "evaluate", "intervals", "evaluate_size"]
FIRST_ALL = STEPS.index("convert")  # `all` no vuelve a descargar


def run(step: str) -> None:
    if step == "download":
        from src.data.download import main
    elif step == "provenance":
        from src.data.provenance import main
    elif step == "convert":
        from src.data.convert import main
    elif step == "mixed":
        from src.data.segment_mixed import main
    elif step == "index":
        from src.data.dataset import build_index as main
    elif step == "preprocess":
        from src.data.preprocess import main
    elif step == "features":
        from src.features.extract import build_feature_matrix as main
    elif step == "feature_space":
        from src.evaluation.feature_space import main
    elif step == "train":
        from src.training.train_classical import main
    elif step == "train_cnn":
        from src.training.train_cnn import main
    elif step == "search_cnn":  # opcional: no entra en `all` (sustituye a train_cnn)
        from src.training.search_cnn import main
    elif step == "evaluate":
        from src.evaluation.evaluate import main
    elif step == "intervals":
        from src.evaluation.intervals import main
    elif step == "evaluate_size":
        from src.evaluation.evaluate_size import main
    else:
        raise SystemExit(f"Paso desconocido: {step}. Opciones: {STEPS + ['all']}")
    print(f"\n######## {step} ########")
    main()


if __name__ == "__main__":
    pedidos = sys.argv[1:] or ["all"]
    steps = [st for s in pedidos for st in (STEPS[FIRST_ALL:] if s == "all" else [s])]
    if len(steps) == 1:
        run(steps[0])
    else:
        # Cada paso corre en su propio proceso. PyTorch y XGBoost/scikit-learn
        # traen cada uno su copia de OpenMP; cargadas en el mismo proceso se
        # pueden bloquear mutuamente (en macOS la búsqueda de la CNN se quedaba
        # colgada para siempre después de entrenar XGBoost). Además, si un paso
        # falla, check=True detiene la cadena.
        import subprocess

        for st in steps:
            subprocess.run([sys.executable, "-u", "-m", "src.main", st], check=True)
