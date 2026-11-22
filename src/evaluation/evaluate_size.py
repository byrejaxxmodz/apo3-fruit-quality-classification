"""Evaluación de la estimación de tamaño (fase CRISP-DM: Evaluación).

La consigna pide clasificar el tamaño de la fruta (pequeño/mediano/grande). El
tamaño se estima con el **diámetro equivalente** del contorno segmentado,

    d_eq = sqrt(4 A / pi),   d_norm = d_eq / min(alto, ancho),

y la clase se asigna por terciles de `d_norm` por (fuente, fruta) en
`src/data/preprocess.py`.

¿Por qué este módulo NO reporta «exactitud»? Porque las etiquetas de tamaño se
derivan de `d_norm`: medir un clasificador contra sus propios terciles daría
100 % de forma trivial y no significaría nada. Además, el grupo todavía no
anotó `size_manual`, así que no hay una verdad humana contra la cual comparar.

Lo que sí se puede medir sin etiquetas manuales es la **fiabilidad** de la
estimación: si fotografiamos la misma fruta física desde otro ángulo, ¿el
sistema le asigna el mismo tamaño? Los grupos de casi duplicados
(`near_duplicate_groups`) son justamente eso —ráfagas de una misma fruta—, así
que funcionan como una prueba test–retest. Reportamos:

  - cobertura de la segmentación (el tamaño solo es válido si `seg_ok`),
  - distribución de `d_eq`/`d_norm` por fuente y fruta,
  - consistencia intra-grupo: acuerdo por pares de la clase de tamaño dentro de
    cada grupo, el coeficiente de variación de `d_norm`, y un acuerdo ajustado
    al azar estilo kappa (observado vs. una línea base por permutación).

Salidas en experiments/results/ y docs/figuras/:
  - tamano_distribucion.csv : conteos de clase y estadísticos de d_eq por fuente/fruta
  - tamano_fiabilidad.csv   : métricas de consistencia test–retest
  - tamano_d_norm.svg       : distribución de d_norm por fuente (diagnóstico)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

RNG = np.random.default_rng(config.SEED)
N_PERM = 200  # permutaciones para la línea base de azar


def _load() -> pd.DataFrame:
    df = pd.read_csv(config.SPLITS_CSV)
    # d_eq en píxeles: d_norm se normalizó por el lado menor de la imagen
    df["d_eq_px"] = df["d_norm"] * df[["height", "width"]].min(axis=1)
    return df


def distribucion(df: pd.DataFrame) -> pd.DataFrame:
    """Conteo de clases de tamaño y estadísticos de d_eq por fuente y fruta."""
    rows = []
    for (src, fruit), g in df.groupby(["source", "fruit"]):
        vc = g["size"].value_counts()
        rows.append({
            "fuente": src, "fruta": fruit, "n": len(g),
            "seg_ok": int(g["seg_ok"].sum()),
            "pequeno": int(vc.get("pequeno", 0)),
            "mediano": int(vc.get("mediano", 0)),
            "grande": int(vc.get("grande", 0)),
            "d_eq_px_mediana": round(float(g["d_eq_px"].median()), 1),
            "d_norm_mediana": round(float(g["d_norm"].median()), 3),
            "d_norm_cv": round(float(g["d_norm"].std() / g["d_norm"].mean()), 3),
        })
    return pd.DataFrame(rows)


def _pairwise_agreement(labels: np.ndarray, groups: np.ndarray) -> float:
    """Fracción de pares (i, j) del mismo grupo con la misma etiqueta de tamaño.

    Es el acuerdo observado p_o de una medida test–retest: 1.0 = la misma fruta
    siempre recibe el mismo tamaño; 0.33 = como lanzar un dado de 3 caras.
    """
    same, total = 0, 0
    order = np.argsort(groups, kind="stable")
    g_sorted, l_sorted = groups[order], labels[order]
    # recorre bloques contiguos de un mismo grupo
    start = 0
    for end in range(1, len(g_sorted) + 1):
        if end == len(g_sorted) or g_sorted[end] != g_sorted[start]:
            block = l_sorted[start:end]
            n = len(block)
            if n >= 2:
                for k, c in zip(*np.unique(block, return_counts=True)):
                    same += c * (c - 1) // 2
                total += n * (n - 1) // 2
            start = end
    return same / total if total else float("nan")


def fiabilidad(df: pd.DataFrame) -> pd.DataFrame:
    """Consistencia test–retest usando los grupos de casi duplicados."""
    rows = []
    for src in ["propias", "kaggle", "todas"]:
        sub = df if src == "todas" else df[df["source"] == src]
        # solo sirven los grupos con >= 2 miembros bien segmentados
        sub = sub[sub["seg_ok"]]
        gsz = sub.groupby("group").transform("size")
        multi = sub[gsz >= 2]
        if multi.empty:
            continue
        labels = multi["size"].to_numpy()
        groups = multi["group"].to_numpy()
        fruit = multi["fruit"].to_numpy()

        p_o = _pairwise_agreement(labels, groups)

        # Línea base de azar: barajar las etiquetas DENTRO de cada fruta (conserva
        # la frecuencia de cada clase) y recomputar el acuerdo por pares.
        chance = []
        for _ in range(N_PERM):
            perm = labels.copy()
            for fr in np.unique(fruit):
                m = fruit == fr
                perm[m] = RNG.permutation(perm[m])
            chance.append(_pairwise_agreement(perm, groups))
        p_e = float(np.mean(chance))
        kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else float("nan")

        # dispersión física: CV de d_norm dentro de cada grupo (mediana entre grupos)
        cv = (multi.groupby("group")["d_norm"]
              .apply(lambda x: x.std(ddof=0) / x.mean() if len(x) >= 2 else np.nan)
              .median())

        rows.append({
            "fuente": src,
            "n_grupos_multi": int(multi["group"].nunique()),
            "n_imagenes": int(len(multi)),
            "acuerdo_observado": round(p_o, 3),
            "acuerdo_azar": round(p_e, 3),
            "kappa_ajustado": round(kappa, 3),
            "cv_dnorm_intragrupo_mediana": round(float(cv), 3),
        })
    return pd.DataFrame(rows)


def _plot_dnorm(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    data, labels = [], []
    for src, g in df[df["seg_ok"]].groupby("source"):
        data.append(g["d_norm"].to_numpy())
        labels.append(f"{src}\n(n={len(g)})")
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("d_norm = d_eq / lado menor")
    ax.set_title("Diámetro equivalente normalizado por fuente")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "tamano_d_norm.svg")
    plt.close(fig)


def main() -> pd.DataFrame:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load()

    dist = distribucion(df)
    dist.to_csv(config.RESULTS_DIR / "tamano_distribucion.csv", index=False)

    rel = fiabilidad(df)
    rel.to_csv(config.RESULTS_DIR / "tamano_fiabilidad.csv", index=False)

    _plot_dnorm(df)

    seg_rate = df["seg_ok"].mean()
    print(f"Cobertura de segmentación (tamaño válido): {seg_rate:.1%} "
          f"({int(df['seg_ok'].sum())}/{len(df)})")
    print("\n=== Distribución de tamaño (primeras filas) ===")
    print(dist.to_string(index=False))
    print("\n=== Fiabilidad test–retest (grupos de casi duplicados) ===")
    print(rel.to_string(index=False))
    print("\nLectura: acuerdo_observado = una misma fruta recibe el mismo tamaño "
          "en fotos distintas; kappa_ajustado descuenta el azar. "
          "Sin objeto de referencia ni distancia fija, el tamaño en píxeles "
          "solo es comparable dentro de la misma fruta y fuente.")
    return rel


if __name__ == "__main__":
    main()
