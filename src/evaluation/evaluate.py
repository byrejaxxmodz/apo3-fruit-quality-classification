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

    ckpt = torch.load(config.CHECKPOINTS_DIR / "cnn.pt", map_location="cpu")
    widths = tuple(ckpt.get("widths", (32, 64, 128)))  # la búsqueda puede ajustar el ancho
    model = SmallCNN(len(config.QUALITY_CLASSES), widths=widths, dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    x = torch.stack([to_tensor(cv2.imread(str(config.ROOT / c))) for c in df_test["crop"]])
    with torch.no_grad():
        return model(x).argmax(1).numpy()


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


def main() -> pd.DataFrame:
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
    for ck in sorted(config.CHECKPOINTS_DIR.glob("*.joblib")):
        preds[ck.stem] = joblib.load(ck).predict(X[te])
    if (config.CHECKPOINTS_DIR / "cnn.pt").exists():
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

    res = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    res.to_csv(config.RESULTS_DIR / "comparacion_modelos.csv", index=False)
    pd.DataFrame(by_source).to_csv(config.RESULTS_DIR / "f1_por_fuente.csv", index=False)
    plot_cnn_curves()
    print(res.to_string(index=False))
    return res


if __name__ == "__main__":
    main()
