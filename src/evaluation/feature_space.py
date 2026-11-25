"""Visualización del espacio de características con PCA y LDA (fases: Datos y Evaluación).

Cada fruta es un punto x en R^62 (características manuales de src/features/).
No se puede «ver» un espacio de 62 dimensiones, así que se proyecta a 2D con
dos técnicas de la Unidad 3 del curso:

PCA (no supervisado): busca las direcciones de MÁXIMA VARIANZA.
    1. Estandarizar con media y desviación de train: z = (x - mu) / sigma
       (si no, las características con números grandes dominan la varianza).
    2. Covarianza: Sigma = Z^T Z / (n - 1)
    3. Vectores propios: Sigma v_j = lambda_j v_j, con lambda_1 >= lambda_2 >= ...
    4. Proyección: t = Z [v_1 v_2]. Varianza explicada por la componente j:
       lambda_j / sum_i lambda_i.

LDA (supervisado, criterio de Fisher): busca las direcciones que SEPARAN las
clases, maximizando
    J(w) = (w^T S_B w) / (w^T S_W w),
con S_B la dispersión entre clases y S_W la dispersión dentro de cada clase.
Con K = 3 clases hay como máximo K - 1 = 2 direcciones discriminantes.

Además se mide el DESPLAZAMIENTO DE DOMINIO: si un clasificador lineal sencillo
distingue «foto de Kaggle» de «foto propia» casi sin errores, las dos fuentes
viven en zonas distintas del espacio. Como la clase *regular* solo existe en
las fotos propias, eso permite el atajo «foto propia => regular» (ver
docs/datos.md).

Salidas en experiments/results/ y docs/figuras/:
  - espacio_caracteristicas.csv : varianza explicada y detector de dominio
  - pca_cargas.csv              : características con más peso en PC1 y PC2
  - pca_caracteristicas.svg     : PCA 2D coloreado por clase y por fuente
  - lda_caracteristicas.svg     : LDA 2D coloreado por clase y por fuente
  - pca_varianza_acumulada.svg  : cuántas componentes hacen falta
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config

COLORES_CLASE = {"good": "#2ca02c", "regular": "#ff7f0e", "bad": "#d62728"}
COLORES_FUENTE = {"kaggle": "#1f77b4", "propias": "#e377c2"}


def _scatter(ax, P, etiquetas, colores, titulo, xlabel, ylabel, encima=None):
    """Dibuja primero las etiquetas comunes y al final `encima` (las propias,
    que son pocas y quedarían tapadas)."""
    orden = [k for k in colores if k != encima] + ([encima] if encima else [])
    for k in orden:
        m = etiquetas == k
        if not m.any():
            continue
        grande = k == encima
        # rasterized: los ~14 500 puntos se guardan como imagen dentro del SVG
        # (si no, el archivo tiene ~30 000 líneas); ejes, texto y leyenda siguen vectoriales
        ax.scatter(P[m, 0], P[m, 1], s=9 if grande else 3, alpha=0.8 if grande else 0.25,
                   c=colores[k], label=f"{k} (n={int(m.sum())})", linewidths=0, rasterized=True)
    ax.set(title=titulo, xlabel=xlabel, ylabel=ylabel)
    ax.legend(markerscale=3, fontsize=8, loc="best")


def domain_detector(Z, source, y, groups) -> dict:
    """¿Qué tan fácil es saber de qué fuente viene una foto mirando sus
    características? Regresión logística con validación cruzada agrupada.

    Se mide dos veces: con todas las fotos y SOLO con las clases que existen en
    ambas fuentes (good y bad), para separar el efecto de la fuente del de la
    clase regular, que solo está en las fotos propias."""
    out = {}
    for nombre, m in [("todas", np.ones(len(y), bool)),
                      ("solo_good_bad", y != config.QUALITY_CLASSES.index("regular"))]:
        s = (source[m] == "propias").astype(int)
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=config.SEED)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
        prob = cross_val_predict(clf, Z[m], s, groups=groups[m], cv=cv, method="predict_proba")[:, 1]
        out[f"dominio_auc_{nombre}"] = roc_auc_score(s, prob)
        out[f"dominio_bal_acc_{nombre}"] = balanced_accuracy_score(s, prob > 0.5)
    return out


def main() -> pd.DataFrame:
    d = np.load(config.FEATURES_NPZ, allow_pickle=True)
    X, y, split, source = d["X"], d["y"], d["split"], d["source"]
    names = d["names"]
    groups = d["group"] if "group" in d.files else np.arange(len(y))
    clase = np.array(config.QUALITY_CLASSES)[y]
    tr = split == "train"

    # Paso 1: estandarizar con las estadísticas de TRAIN (sin fuga hacia val/test)
    scaler = StandardScaler().fit(X[tr])
    Z = scaler.transform(X)

    # PCA ajustado en train y aplicado a todo
    pca = PCA(random_state=config.SEED).fit(Z[tr])
    P = pca.transform(Z)[:, :2]
    evr = pca.explained_variance_ratio_
    acum = np.cumsum(evr)

    # LDA ajustado en train
    lda = LinearDiscriminantAnalysis(n_components=2).fit(Z[tr], y[tr])
    L = lda.transform(Z)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for nombre, Q, ejes in [
        ("pca", P, (f"PC1 ({evr[0]:.0%} de la varianza)", f"PC2 ({evr[1]:.0%})")),
        ("lda", L, ("LD1", "LD2")),
    ]:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
        _scatter(a1, Q, clase, COLORES_CLASE, f"{nombre.upper()}: por clase de calidad", *ejes,
                 encima="regular")
        _scatter(a2, Q, source, COLORES_FUENTE, f"{nombre.upper()}: por fuente", *ejes,
                 encima="propias")
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / f"{nombre}_caracteristicas.svg", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(np.arange(1, len(acum) + 1), acum, marker=".")
    for meta in (0.90, 0.95):
        k = int(np.searchsorted(acum, meta) + 1)
        ax.axhline(meta, ls="--", lw=0.8, c="gray")
        ax.annotate(f"{meta:.0%}: {k} comp.", (k, meta), textcoords="offset points",
                    xytext=(-70, 3), fontsize=8)
    ax.set(xlabel="número de componentes", ylabel="varianza explicada acumulada",
           title="PCA sobre las 62 características")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "pca_varianza_acumulada.svg")
    plt.close(fig)

    # Cargas: qué características pesan más en cada componente (|v_j|)
    filas = []
    for j in range(2):
        v = pca.components_[j]
        for i in np.argsort(-np.abs(v))[:10]:
            filas.append({"componente": f"PC{j + 1}", "caracteristica": names[i], "carga": v[i]})
    pd.DataFrame(filas).to_csv(config.RESULTS_DIR / "pca_cargas.csv", index=False)

    resumen = {
        "n_imagenes": len(y), "dim": X.shape[1],
        "varianza_pc1": evr[0], "varianza_pc2": evr[1], "varianza_pc1_pc2": acum[1],
        "componentes_90": int(np.searchsorted(acum, 0.90) + 1),
        "componentes_95": int(np.searchsorted(acum, 0.95) + 1),
        "lda_varianza_ld1": lda.explained_variance_ratio_[0],
        **domain_detector(Z, source, y, groups),
    }
    res = pd.DataFrame([resumen])
    res.to_csv(config.RESULTS_DIR / "espacio_caracteristicas.csv", index=False)
    print(res.T.round(3).to_string(header=False))
    return res


if __name__ == "__main__":
    main()
