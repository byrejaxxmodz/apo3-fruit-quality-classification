"""Evaluación final en el conjunto de prueba (fase CRISP-DM: Evaluación).

Genera en experiments/results/ y docs/figuras/:
  - comparacion_modelos.csv : exactitud, precisión, recall y F1 macro en test
                              de cada modelo + la línea base (clase mayoritaria)
  - f1_por_fuente.csv       : F1 en fotos de Kaggle vs. fotos propias
                              (mide si el modelo generaliza a nuestro dominio)
  - cm_<modelo>.svg         : matriz de confusión normalizada (vectorial)
  - errores_<modelo>.csv    : imágenes mal clasificadas, para análisis de errores
  - curvas_cnn.svg          : pérdida y F1 por época (diagnóstico de sobreajuste)

Definiciones (por clase k):
  precision_k = TP_k / (TP_k + FP_k)     recall_k = TP_k / (TP_k + FN_k)
  F1_k = 2 P_k R_k / (P_k + R_k)         F1_macro = (1/K) sum_k F1_k
"""
from __future__ import annotations

from typing import Iterable, Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)

from src import config


def _metrics(name, y, p):
    return {
        "modelo": name,
        "accuracy": accuracy_score(y, p),
        "precision_macro": precision_score(y, p, average="macro", zero_division=0),
        "recall_macro": recall_score(y, p, average="macro", zero_division=0),
        "f1_macro": f1_score(y, p, average="macro", zero_division=0),
    }


def _plot_cm(name, y, p):
    labels = list(range(len(config.QUALITY_CLASSES)))
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, p, labels=labels, display_labels=config.QUALITY_CLASSES, normalize="true",
        cmap="Blues", ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"cm_{name}.svg")
    plt.close(fig)


def predict_cnn(df_test: pd.DataFrame) -> np.ndarray:
    import cv2
    import torch

    from src.models.cnn import SmallCNN
    from src.training.train_cnn import to_tensor

    # Este proceso ya cargó scikit-learn y XGBoost, que traen su propia copia de
    # OpenMP; si PyTorch abre su región paralela con la suya, el proceso se
    # bloquea para siempre (pasa en macOS). Con un solo hilo no hay región
    # paralela, y para inferir 2.000 recortes de 128x128 sobra.
    torch.set_num_threads(1)
    ckpt = torch.load(config.CHECKPOINTS_DIR / "cnn.pt", map_location="cpu")
    widths = tuple(ckpt.get("widths", (32, 64, 128)))  # la búsqueda puede ajustar el ancho
    model = SmallCNN(len(config.QUALITY_CLASSES), widths=widths, dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    preds = []
    with torch.no_grad():  # por lotes: todo el test de una vez ocupa varios GB de activaciones
        for i in range(0, len(df_test), 256):
            x = torch.stack([to_tensor(cv2.imread(str(config.ROOT / c)))
                             for c in df_test["crop"].iloc[i:i + 256]])
            preds.append(model(x).argmax(1).numpy())
    return np.concatenate(preds)


def plot_cnn_curves() -> None:
    f = config.LOGS_DIR / "cnn_history.csv"
    if not f.exists():
        return
    h = pd.read_csv(f)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(h.epoch, h.loss)
    ax[0].set(title="Pérdida de entrenamiento", xlabel="época")
    ax[1].plot(h.epoch, h.f1_train, label="train")
    ax[1].plot(h.epoch, h.f1_val, label="val")
    ax[1].set(title="F1 macro", xlabel="época")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "curvas_cnn.svg")
    plt.close(fig)


def _check_same_partition(df_te: pd.DataFrame, base_pred: np.ndarray, y_te: np.ndarray) -> None:
    """Antes de mezclar resultados nuevos con los ya guardados, verifica que el
    conjunto de test sea el mismo: los errores de la línea base (que lista todas
    las imágenes de test que no son de la clase mayoritaria) deben coincidir."""
    prev = config.RESULTS_DIR / "errores_baseline_mayoritaria.csv"
    if not prev.exists():
        return
    old = set(pd.read_csv(prev)["path"])
    new = set(df_te.loc[base_pred != y_te, "path"])
    if old != new:
        raise SystemExit(
            f"La partición de test cambió ({len(old ^ new)} imágenes distintas): los resultados "
            "guardados no son comparables. Ejecuta la evaluación completa (sin --models).")


def _merge_previous(nuevo: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """Une las filas nuevas con las de los modelos que no se reevaluaron."""
    ruta = config.RESULTS_DIR / nombre
    if not ruta.exists():
        return nuevo
    prev = pd.read_csv(ruta)
    return pd.concat([prev[~prev["modelo"].isin(nuevo["modelo"])], nuevo], ignore_index=True)


def main(models: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Evalúa todos los modelos entrenados o, con `models`, solo esos y los
    agrega a las tablas existentes (reemplazando sus filas anteriores)."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.SPLITS_CSV)
    d = np.load(config.FEATURES_NPZ, allow_pickle=True)
    X, y, split = d["X"], d["y"], d["split"]
    te = split == "test"
    df_te = df[te].reset_index(drop=True)

    preds = {}
    base = DummyClassifier(strategy="most_frequent").fit(X[~te], y[~te])
    preds["baseline_mayoritaria"] = base.predict(X[te])
    if models:  # la línea base solo sirve para verificar la partición; sus filas no se tocan
        _check_same_partition(df_te, preds.pop("baseline_mayoritaria"), y[te])
    for ck in sorted(config.CHECKPOINTS_DIR.glob("*.joblib")):
        if not models or ck.stem in models:
            preds[ck.stem] = joblib.load(ck).predict(X[te])
    if (config.CHECKPOINTS_DIR / "cnn.pt").exists() and (not models or "cnn" in models):
        preds["cnn"] = predict_cnn(df_te)

    rows, by_source = [], []
    for name, p in preds.items():
        rows.append(_metrics(name, y[te], p))
        _plot_cm(name, y[te], p)
        for src in df_te["source"].unique():
            m = (df_te["source"] == src).to_numpy()
            yt, pt = y[te][m], p[m]
            # Solo se promedian las clases presentes en esa fuente (Kaggle no
            # tiene «regular»); predecir «regular» allí sigue contando como error
            labels = np.unique(yt)
            row = {"modelo": name, "fuente": src, "n": int(m.sum()),
                   "accuracy": accuracy_score(yt, pt),
                   "f1_macro": f1_score(yt, pt, labels=labels, average="macro", zero_division=0)}
            for k in labels:
                row[f"recall_{config.QUALITY_CLASSES[k]}"] = recall_score(
                    yt, pt, labels=[k], average="macro", zero_division=0)
            by_source.append(row)
        err = df_te.loc[p != y[te], ["path", "source", "fruit", "quality"]].copy()
        err["prediccion"] = [config.QUALITY_CLASSES[i] for i in p[p != y[te]]]
        err.to_csv(config.RESULTS_DIR / f"errores_{name}.csv", index=False)
        print(f"\n=== {name} ===")
        print(classification_report(y[te], p, labels=range(len(config.QUALITY_CLASSES)),
                                    target_names=config.QUALITY_CLASSES, zero_division=0))

    res = pd.DataFrame(rows)
    by_source = pd.DataFrame(by_source)
    if models:  # evaluación parcial: conservar las filas de los demás modelos
        res = _merge_previous(res, "comparacion_modelos.csv")
        by_source = _merge_previous(by_source, "f1_por_fuente.csv")
    res = res.sort_values("f1_macro", ascending=False)
    res.to_csv(config.RESULTS_DIR / "comparacion_modelos.csv", index=False)
    by_source.to_csv(config.RESULTS_DIR / "f1_por_fuente.csv", index=False)
    if not models or "cnn" in models:
        plot_cnn_curves()
    print(res.to_string(index=False))
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", help="evaluar solo estos modelos y agregarlos a las tablas")
    main(ap.parse_args().models)
