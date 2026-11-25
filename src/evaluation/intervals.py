"""Intervalos de confianza y comparación estadística de modelos (fase: Evaluación).

¿Por qué hace falta? Una métrica puntual sobre pocas imágenes engaña. En las
fotos propias de test solo hay 5 frutas *bad*, así que un acierto más o menos
mueve su recall en 0.20. La guía de transfer learning del curso (Unidad 3) pide
expresar las diferencias entre modelos en unidades de su error de muestreo;
este módulo lo hace para todos los modelos y fuentes.

Fundamentos (z = 1.96 para el 95 %):

1. Error estándar binomial (Wald) de una proporción p = k / n:
       SE(p) = sqrt(p (1 - p) / n)
2. Intervalo de Wilson al 95 %. Es mejor que p ± z·SE cuando n es pequeño o p
   está cerca de 0 o de 1 (con 5/5 aciertos, Wald daría el intervalo [1, 1]):
       centro = (p + z²/2n) / (1 + z²/n)
       radio  = z / (1 + z²/n) · sqrt(p (1 - p) / n + z² / 4n²)
3. F1 macro: su varianza no tiene fórmula cerrada, así que se usa bootstrap
   percentil (B remuestreos con reemplazo del conjunto de test). Si se conoce el
   grupo de casi duplicados de cada imagen, se remuestrean GRUPOS (bootstrap por
   conglomerados): varias fotos de la misma fruta no son independientes, y
   remuestrearlas por separado daría intervalos demasiado estrechos.
4. Prueba exacta de McNemar para comparar dos modelos sobre las MISMAS imágenes.
   Solo importan los desacuerdos: b = errores solo de A, c = errores solo de B.
       H0: ambos modelos se equivocan igual  =>  b ~ Binomial(b + c, 1/2)
       p = min(1, 2 · P(X <= min(b, c)))
   Como se hacen muchas comparaciones a la vez, los p-valores se corrigen con
   Holm (controla la probabilidad de al menos un falso positivo).

Entradas: los errores_<modelo>.csv de evaluate.py y, si existe, splits.csv.
Sin splits.csv (por ejemplo, en un clon sin las imágenes), el conjunto de test
se reconstruye: la línea base mayoritaria predice siempre «good», así que su
archivo de errores lista TODAS las imágenes de test que no son «good». Las
«good» se completan hasta el n de cada fuente de f1_por_fuente.csv. Antes de
reportar, el módulo verifica que el F1 reconstruido coincida con el de
evaluate.py.

Salidas en experiments/results/ y docs/figuras/:
  - intervalos_confianza.csv : métricas con su IC 95 % por modelo y fuente
  - mcnemar.csv              : comparación por pares de modelos
  - ic_por_fuente.svg        : F1 macro y recall por clase con sus IC
"""
from __future__ import annotations

from itertools import combinations
from math import comb, sqrt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

Z = 1.96
N_BOOT = 2000
META_F1 = 0.80          # meta de negocio: F1 macro en fotos propias (docs/crisp_dm.md)
BASELINE = "baseline_mayoritaria"
CLASSES = config.QUALITY_CLASSES
K = len(CLASSES)


# --------------------------------------------------------------------------
# Fórmulas
# --------------------------------------------------------------------------
def wald_se(k: int, n: int) -> float:
    """Error estándar binomial sqrt(p(1-p)/n) de la proporción k/n."""
    if n == 0:
        return float("nan")
    p = k / n
    return sqrt(p * (1 - p) / n)


def wilson(k: int, n: int, z: float = Z) -> tuple:
    """Intervalo de Wilson para la proporción k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    radio = z / den * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centro - radio), min(1.0, centro + radio)


def mcnemar_exact(b: int, c: int) -> float:
    """p-valor bilateral exacto de McNemar con b y c desacuerdos."""
    n = b + c
    if n == 0:
        return 1.0
    cola = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * cola)


def holm(pvals) -> np.ndarray:
    """Corrección de Holm-Bonferroni para m comparaciones."""
    p = np.asarray(pvals, float)
    m = len(p)
    orden = np.argsort(p)
    ajust = np.empty(m)
    acum = 0.0
    for rango, i in enumerate(orden):
        acum = max(acum, min(1.0, (m - rango) * p[i]))
        ajust[i] = acum
    return ajust


def f1_macro_cm(cm: np.ndarray, labels) -> float:
    """F1 macro a partir de una matriz de confusión (filas = verdad)."""
    f1 = []
    for k in labels:
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        den = 2 * tp + fp + fn
        f1.append(2 * tp / den if den else 0.0)
    return float(np.mean(f1))


# --------------------------------------------------------------------------
# Conjunto de test y predicciones
# --------------------------------------------------------------------------
def load_errors() -> dict:
    return {p.stem.removeprefix("errores_"): pd.read_csv(p)
            for p in sorted(config.RESULTS_DIR.glob("errores_*.csv"))}


def load_test_set(errors: dict) -> tuple:
    """Devuelve (test, modo): una fila por imagen de test con path, source,
    quality y group."""
    cols = ["path", "source", "quality"]
    if config.SPLITS_CSV.exists():
        df = pd.read_csv(config.SPLITS_CSV)
        test = df.loc[df["split"] == "test", cols + ["group"]].reset_index(drop=True)
        test["group"] = test["group"].astype(str)
        return test, "splits.csv (bootstrap por grupos de casi duplicados)"

    base = errors[BASELINE]
    if not (base["prediccion"] == "good").all():
        raise SystemExit("La línea base no predice siempre «good»: no se puede reconstruir el test.")
    known_good = pd.concat([e.loc[e["quality"] == "good", cols] for e in errors.values()])
    test = pd.concat([base[cols], known_good]).drop_duplicates("path")
    n_fuente = (pd.read_csv(config.RESULTS_DIR / "f1_por_fuente.csv")
                .query("modelo == @BASELINE").set_index("fuente")["n"])
    relleno = []
    for fuente, n in n_fuente.items():
        faltan = int(n) - int((test["source"] == fuente).sum())
        relleno += [{"path": f"_good_bien_clasificada_{fuente}_{i}", "source": fuente,
                     "quality": "good"} for i in range(faltan)]
    test = pd.concat([test, pd.DataFrame(relleno)], ignore_index=True)
    test["group"] = test["path"]   # sin grupos: cada imagen es su propio grupo
    return test, "reconstruido desde errores_*.csv (bootstrap por imagen)"


def predictions(test: pd.DataFrame, err: pd.DataFrame) -> np.ndarray:
    """Predicción por imagen: la verdad, salvo donde el modelo se equivocó."""
    pred = test.set_index("path")["quality"].copy()
    faltan = set(err["path"]) - set(pred.index)
    if faltan:
        raise SystemExit(f"{len(faltan)} errores no están en el conjunto de test (¿otra partición?)")
    pred.loc[err["path"].to_numpy()] = err["prediccion"].to_numpy()
    return pred.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()


# --------------------------------------------------------------------------
# Métricas con intervalos
# --------------------------------------------------------------------------
def bootstrap_f1(y, p, groups, labels, rng) -> np.ndarray:
    """F1 macro de N_BOOT remuestreos de grupos con reemplazo.

    Cada grupo aporta su propia matriz de confusión; un remuestreo es una suma
    ponderada de esas matrices (el peso es cuántas veces salió el grupo)."""
    codes, g = np.unique(groups, return_inverse=True)
    gcm = np.zeros((len(codes), K, K))
    np.add.at(gcm, (g, y, p), 1)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pesos = np.bincount(rng.integers(len(codes), size=len(codes)), minlength=len(codes))
        out[b] = f1_macro_cm(np.tensordot(pesos, gcm, axes=1), labels)
    return out


def metrics_with_ci(name, fuente, y, p, groups, rng) -> dict:
    labels = np.unique(y)          # igual que evaluate.py: solo las clases presentes
    cm = np.zeros((K, K))
    np.add.at(cm, (y, p), 1)
    n, aciertos = len(y), int((y == p).sum())
    boot = bootstrap_f1(y, p, groups, labels, rng)
    lo_acc, hi_acc = wilson(aciertos, n)
    row = {
        "modelo": name, "fuente": fuente, "n": n, "n_grupos": len(np.unique(groups)),
        "accuracy": aciertos / n, "accuracy_se": wald_se(aciertos, n),
        "accuracy_ic_inf": lo_acc, "accuracy_ic_sup": hi_acc,
        "f1_macro": f1_macro_cm(cm, labels),
        "f1_macro_ic_inf": np.percentile(boot, 2.5), "f1_macro_ic_sup": np.percentile(boot, 97.5),
        "f1_macro_boot_se": boot.std(ddof=1),
        f"prob_f1_ge_{META_F1:.2f}": float((boot >= META_F1).mean()),
    }
    for k in labels:
        nk, tp = int((y == k).sum()), int(cm[k, k])
        lo, hi = wilson(tp, nk)
        c = CLASSES[k]
        row.update({f"n_{c}": nk, f"recall_{c}": tp / nk,
                    f"recall_{c}_ic_inf": lo, f"recall_{c}_ic_sup": hi})
    return row


def mcnemar_table(test, preds) -> pd.DataFrame:
    y = test["quality"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    filas = []
    for fuente in ["global", "kaggle", "propias"]:
        m = np.ones(len(test), bool) if fuente == "global" else (test["source"] == fuente).to_numpy()
        bloque = []
        for a, b in combinations([k for k in preds if k != BASELINE], 2):
            mal_a, mal_b = preds[a][m] != y[m], preds[b][m] != y[m]
            solo_a, solo_b = int((mal_a & ~mal_b).sum()), int((mal_b & ~mal_a).sum())
            bloque.append({"fuente": fuente, "modelo_a": a, "modelo_b": b, "n": int(m.sum()),
                           "errores_a": int(mal_a.sum()), "errores_b": int(mal_b.sum()),
                           "solo_a_falla": solo_a, "solo_b_falla": solo_b,
                           "p_valor": mcnemar_exact(solo_a, solo_b)})
        bloque = pd.DataFrame(bloque)
        bloque["p_holm"] = holm(bloque["p_valor"])
        bloque["diferencia_significativa"] = bloque["p_holm"] < 0.05
        filas.append(bloque)
    return pd.concat(filas, ignore_index=True)


# --------------------------------------------------------------------------
# Figura
# --------------------------------------------------------------------------
def plot_intervals(res: pd.DataFrame) -> None:
    orden = (res.query("fuente == 'propias'").sort_values("f1_macro")["modelo"].tolist())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    estilos = {"kaggle": ("#1f77b4", -0.15), "propias": ("#d62728", 0.15)}
    for fuente, (color, dy) in estilos.items():
        sub = res[res["fuente"] == fuente].set_index("modelo").loc[orden]
        pos = np.arange(len(orden)) + dy
        ax1.errorbar(sub["f1_macro"], pos,
                     xerr=[sub["f1_macro"] - sub["f1_macro_ic_inf"],
                           sub["f1_macro_ic_sup"] - sub["f1_macro"]],
                     fmt="o", color=color, capsize=3, label=fuente)
    ax1.axvline(META_F1, ls="--", c="gray", lw=1, label=f"meta ({META_F1:.2f})")
    ax1.set(title="F1 macro en test (IC 95 %, bootstrap)", xlabel="F1 macro", xlim=(0, 1.02))
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)

    prop = res[res["fuente"] == "propias"].set_index("modelo").loc[orden]
    colores = {"good": "#2ca02c", "regular": "#ff7f0e", "bad": "#7f7f7f"}
    for j, c in enumerate(CLASSES):
        if f"recall_{c}" not in prop:
            continue
        pos = np.arange(len(orden)) + (j - 1) * 0.22
        ax2.errorbar(prop[f"recall_{c}"], pos,
                     xerr=[prop[f"recall_{c}"] - prop[f"recall_{c}_ic_inf"],
                           prop[f"recall_{c}_ic_sup"] - prop[f"recall_{c}"]],
                     fmt="s", ms=4, color=colores[c], capsize=2,
                     label=f"{c} (n={int(prop[f'n_{c}'].iloc[0])})")
    ax2.set(title="Recall por clase en fotos propias (IC 95 %, Wilson)", xlabel="recall",
            xlim=(-0.02, 1.02))
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
    ax1.set_yticks(range(len(orden)), orden)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "ic_por_fuente.svg")
    plt.close(fig)


def main() -> pd.DataFrame:
    errors = load_errors()
    test, modo = load_test_set(errors)
    print(f"[intervalos] conjunto de test: {len(test)} imágenes, modo = {modo}")
    y = test["quality"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    preds = {name: predictions(test, err) for name, err in errors.items()}

    rng = np.random.default_rng(config.SEED)
    rows = []
    for name, p in preds.items():
        for fuente in ["global", "kaggle", "propias"]:
            m = np.ones(len(test), bool) if fuente == "global" else (test["source"] == fuente).to_numpy()
            rows.append(metrics_with_ci(name, fuente, y[m], p[m], test["group"].to_numpy()[m], rng))
    res = pd.DataFrame(rows)

    # Verificación: el F1 puntual debe coincidir con el de evaluate.py
    ref = pd.read_csv(config.RESULTS_DIR / "f1_por_fuente.csv").set_index(["modelo", "fuente"])["f1_macro"]
    chk = res.set_index(["modelo", "fuente"])["f1_macro"]
    comunes = chk.index.intersection(ref.index)
    dif = (chk.loc[comunes] - ref.loc[comunes]).abs().max()
    if dif > 1e-9:
        raise SystemExit(f"El F1 reconstruido difiere del de evaluate.py (máx. {dif:.3g})")
    print(f"[intervalos] F1 reconstruido = F1 de evaluate.py en {len(comunes)} casos")

    res.insert(2, "modo", modo)
    res.to_csv(config.RESULTS_DIR / "intervalos_confianza.csv", index=False)
    mc = mcnemar_table(test, preds)
    mc.to_csv(config.RESULTS_DIR / "mcnemar.csv", index=False)
    plot_intervals(res[res["modelo"] != BASELINE])

    cols = ["modelo", "fuente", "f1_macro", "f1_macro_ic_inf", "f1_macro_ic_sup"]
    print(res[cols].round(3).to_string(index=False))
    print(mc.query("fuente == 'propias'")[["modelo_a", "modelo_b", "solo_a_falla",
                                            "solo_b_falla", "p_valor", "p_holm"]]
          .round(4).to_string(index=False))
    return res


if __name__ == "__main__":
    main()
