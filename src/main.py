"""Punto de entrada: ejecuta cada fase del pipeline CRISP-DM.

Uso (desde la raíz del repo):
    python -m src.main download      # baja Kaggle + fotos propias de Drive
    python -m src.main convert       # HEIC -> JPG, quita metadatos (GPS)
    python -m src.main provenance    # marca imágenes generadas con IA (C2PA)
    python -m src.main mixed         # recorta frutas de mixed_quality
    python -m src.main index         # construye data/processed/index.csv
    python -m src.main preprocess    # segmenta, recorta, tamaño, splits
    python -m src.main features      # matriz de características
    python -m src.main train         # modelos clásicos (GridSearchCV)
    python -m src.main train_cnn     # CNN desde cero
    python -m src.main evaluate      # métricas en test + figuras
    python -m src.main all           # provenance -> evaluate (sin descargar)
"""
from __future__ import annotations

import sys

STEPS = ["download", "convert", "provenance", "mixed", "index", "preprocess", "features",
         "train", "train_cnn", "evaluate"]
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
    elif step == "train":
        from src.training.train_classical import main
    elif step == "train_cnn":
        from src.training.train_cnn import main
    elif step == "evaluate":
        from src.evaluation.evaluate import main
    else:
        raise SystemExit(f"Paso desconocido: {step}. Opciones: {STEPS + ['all']}")
    print(f"\n######## {step} ########")
    main()


if __name__ == "__main__":
    steps = sys.argv[1:] or ["all"]
    for s in steps:
        if s == "all":
            for st in STEPS[FIRST_ALL:]:
                run(st)
        else:
            run(s)
