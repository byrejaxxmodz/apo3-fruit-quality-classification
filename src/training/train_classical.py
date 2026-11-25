"""Entrenamiento de modelos clásicos con búsqueda en rejilla (fase: Modelado).

- Se usan solo train + val (test queda intacto hasta la evaluación final).
- GridSearchCV con StratifiedGroupKFold(k=5): para cada combinación de
  hiperparámetros se promedia la métrica en los 5 pliegues y se elige la mejor.
- Métrica de selección: F1 macro = (1/K) sum_k F1_k. Se prefiere a la
  exactitud porque con clases desbalanceadas un modelo que siempre dice
  "good" puede tener exactitud alta y F1 de "regular" igual a 0.
"""
from __future__ import annotations

import argparse
import json
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from src import config
from src.models.classical import SAMPLE_WEIGHTED, get_models


def load_features():
    d = np.load(config.FEATURES_NPZ, allow_pickle=True)
    group = d["group"] if "group" in d.files else np.arange(len(d["y"]))
    return d["X"], d["y"], d["split"], d["source"], group


def main(models: Optional[Iterable[str]] = None) -> pd.DataFrame:
    X, y, split, _, group = load_features()
    trval = split != "test"
    Xtr, ytr, gtr = X[trval], y[trval], group[trval]
    min_class = np.bincount(ytr).min()
    # StratifiedGroupKFold: las fotos casi idénticas de una misma fruta nunca
    # quedan repartidas entre el pliegue de entrenamiento y el de validación
    cv = StratifiedGroupKFold(n_splits=max(2, min(config.CV_FOLDS, min_class)),
                              shuffle=True, random_state=config.SEED)

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, (pipe, grid) in get_models().items():
        if models and name not in models:
            continue
        print(f"[train] {name}: {np.prod([len(v) for v in grid.values()])} combinaciones x {cv.n_splits} pliegues")
        gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1,
                          return_train_score=True, refit=True)
        # GridSearchCV reparte los pesos por muestra entre los pliegues
        fit_params = ({"clf__sample_weight": compute_sample_weight("balanced", ytr)}
                      if name in SAMPLE_WEIGHTED else {})
        gs.fit(Xtr, ytr, groups=gtr, **fit_params)
        best_model = gs.best_estimator_
        if name == "svm":  # la app necesita probabilidades: reentrenar con Platt
            best_model = best_model.set_params(clf__probability=True).fit(Xtr, ytr)
        joblib.dump(best_model, config.CHECKPOINTS_DIR / f"{name}.joblib")
        pd.DataFrame(gs.cv_results_).to_csv(config.RESULTS_DIR / f"cv_{name}.csv", index=False)
        best = gs.cv_results_
        i = gs.best_index_
        summary.append({
            "modelo": name,
            "mejores_hiperparametros": json.dumps(gs.best_params_),
            "f1_macro_cv": best["mean_test_score"][i],
            "f1_macro_cv_std": best["std_test_score"][i],
            "f1_macro_train": best["mean_train_score"][i],   # train >> cv => sobreajuste
        })
        print(f"         mejor F1 cv = {gs.best_score_:.3f}  {gs.best_params_}")

    res = pd.DataFrame(summary)
    out = config.RESULTS_DIR / "resumen_cv_clasicos.csv"
    if models and out.exists():  # entrenamiento parcial: conservar las filas de los demás modelos
        prev = pd.read_csv(out)
        res = pd.concat([prev[~prev["modelo"].isin(res["modelo"])], res], ignore_index=True)
    res.to_csv(out, index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", help="entrenar solo estos modelos (p. ej. xgboost)")
    main(ap.parse_args().models)
