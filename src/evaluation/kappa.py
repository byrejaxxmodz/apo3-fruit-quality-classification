"""Acuerdo entre anotadores (kappa de Cohen y de Fleiss) sobre las fotos propias.

Las etiquetas de calidad de las fotos propias salen del nombre de la carpeta en
la que cada integrante guardó la foto. Para saber qué tan repetible es ese
criterio, varios integrantes vuelven a etiquetar a ciegas una misma muestra y se
mide cuánto coinciden descontando el acuerdo que se daría por azar.

Uso:
    python -m src.evaluation.kappa muestra     # crea la página de anotación
    python -m src.evaluation.kappa calcular    # lee las anotaciones y calcula kappa

`muestra` escribe:
    data/interim/kappa/anotar_calidad.html   página única (las fotos van dentro);
                                              se comparte por WhatsApp o Drive
    data/kappa/clave.csv                     id de cada foto, ruta y etiqueta de carpeta

Cada anotador abre la página, etiqueta las fotos sin ver la carpeta y descarga
`kappa_<nombre>.csv`. Esos archivos se copian en `data/kappa/` y se corre
`calcular`, que escribe `experiments/results/kappa_anotadores.csv`.
"""
from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.config import INTERIM_DIR, DATA_DIR, RESULTS_DIR, ROOT, SPLITS_CSV
from src.data.preprocess import is_augmented_copy

CLASES = ["good", "regular", "bad"]
KAPPA_DIR = DATA_DIR / "kappa"
CLAVE_CSV = KAPPA_DIR / "clave.csv"
HTML_OUT = INTERIM_DIR / "kappa" / "anotar_calidad.html"
RESULT_CSV = RESULTS_DIR / "kappa_anotadores.csv"


# --------------------------------------------------------------------------
# Estadística
# --------------------------------------------------------------------------
def cohen_kappa(a, b, labels=CLASES) -> float:
    """κ = (p_o − p_e) / (1 − p_e) para dos anotadores."""
    a, b = np.asarray(a), np.asarray(b)
    p_o = np.mean(a == b)
    p_e = sum(np.mean(a == k) * np.mean(b == k) for k in labels)
    return float((p_o - p_e) / (1 - p_e)) if p_e < 1 else float("nan")


def fleiss_kappa(etiquetas: np.ndarray, labels=CLASES) -> float:
    """Kappa de Fleiss. `etiquetas` es (n_fotos, n_anotadores)."""
    n, m = etiquetas.shape
    cuentas = np.stack([(etiquetas == k).sum(axis=1) for k in labels], axis=1)  # n × K
    p_i = (np.sum(cuentas**2, axis=1) - m) / (m * (m - 1))
    p_bar = p_i.mean()
    p_k = cuentas.sum(axis=0) / (n * m)
    p_e = np.sum(p_k**2)
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else float("nan")


def bootstrap_ci(fn, *cols, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    """IC 95 % percentil remuestreando fotos (filas)."""
    rng = np.random.default_rng(seed)
    n = len(cols[0])
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = fn(*(c[idx] for c in cols))
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def interpretar(k: float) -> str:
    """Escala de Landis y Koch (1977)."""
    if k < 0:
        return "peor que el azar"
    for lim, txt in [(0.20, "leve"), (0.40, "aceptable"), (0.60, "moderado"),
                     (0.80, "sustancial"), (1.01, "casi perfecto")]:
        if k <= lim:
            return txt
    return "casi perfecto"


# --------------------------------------------------------------------------
# Muestra y página de anotación
# --------------------------------------------------------------------------
def elegir_muestra(splits: pd.DataFrame, por_clase: int, seed: int) -> pd.DataFrame:
    """Una foto por fruta física (grupo), repartida entre frutas dentro de cada clase."""
    rng = np.random.default_rng(seed)
    p = splits[(splits.source == "propias") & (~splits.ai_generated.astype(bool))]
    filas = []
    for q in CLASES:
        grupos = {f: list(rng.permutation(g.group.unique()))
                  for f, g in p[p.quality == q].groupby("fruit")}
        elegidos = []
        for f in itertools.cycle(sorted(grupos)):
            if len(elegidos) == por_clase or not any(grupos.values()):
                break
            if grupos[f]:
                elegidos.append(grupos[f].pop())
        for g in elegidos:
            fotos = p[p.group == g]
            originales = fotos[~fotos.path.map(is_augmented_copy)]
            fotos = originales if len(originales) else fotos  # mejor la toma original que una copia rotada
            filas.append(fotos.iloc[rng.integers(len(fotos))])
    m = pd.DataFrame(filas).sample(frac=1, random_state=seed).reset_index(drop=True)
    m.insert(0, "id", [f"{i + 1:03d}" for i in range(len(m))])
    return m[["id", "path", "fruit", "quality"]].rename(columns={"quality": "carpeta"})


def _miniatura_b64(path: Path, lado: int = 640) -> str:
    """JPEG reducido y sin EXIF (sin GPS ni modelo del teléfono)."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((lado, lado))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def escribir_html(muestra: pd.DataFrame, out: Path) -> None:
    fotos = [{"id": r.id, "src": "data:image/jpeg;base64," + _miniatura_b64(ROOT / r.path)}
             for r in muestra.itertuples()]
    plantilla = (Path(__file__).parent / "kappa_plantilla.html").read_text(encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(plantilla.replace("/*FOTOS*/[]", json.dumps(fotos)), encoding="utf-8")


def cmd_muestra(por_clase: int, seed: int) -> None:
    splits = pd.read_csv(SPLITS_CSV)
    muestra = elegir_muestra(splits, por_clase, seed)
    KAPPA_DIR.mkdir(parents=True, exist_ok=True)
    muestra.to_csv(CLAVE_CSV, index=False)
    escribir_html(muestra, HTML_OUT)
    print(muestra.groupby(["carpeta", "fruit"]).size().unstack(fill_value=0))
    print(f"\n{len(muestra)} fotos. Clave: {CLAVE_CSV.relative_to(ROOT)}")
    print(f"Página para compartir ({HTML_OUT.stat().st_size / 1e6:.1f} MB): {HTML_OUT.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# Cálculo
# --------------------------------------------------------------------------
def cargar_anotaciones() -> pd.DataFrame:
    """Tabla ancha: una fila por foto, una columna por anotador más `carpeta`."""
    clave = pd.read_csv(CLAVE_CSV, dtype={"id": str})
    tabla = clave.set_index("id")[["carpeta"]]
    for f in sorted(KAPPA_DIR.glob("kappa_*.csv")):
        a = pd.read_csv(f, dtype={"id": str})
        nombre = str(a.anotador.iloc[0]).strip() or f.stem.removeprefix("kappa_")
        tabla[nombre] = a.drop_duplicates("id", keep="last").set_index("id").etiqueta
    return tabla


def cmd_calcular() -> None:
    tabla = cargar_anotaciones()
    anotadores = [c for c in tabla.columns if c != "carpeta"]
    if not anotadores:
        raise SystemExit(f"No hay archivos kappa_*.csv en {KAPPA_DIR.relative_to(ROOT)}")
    filas = []
    for a, b in itertools.combinations(["carpeta"] + anotadores, 2):
        t = tabla[[a, b]].dropna()
        x, y = t[a].to_numpy(), t[b].to_numpy()
        k = cohen_kappa(x, y)
        lo, hi = bootstrap_ci(cohen_kappa, x, y)
        filas.append({"comparacion": f"{a} vs {b}", "tipo": "Cohen", "n_fotos": len(t),
                      "acuerdo_observado": round(float(np.mean(x == y)), 3),
                      "kappa": round(k, 3), "ic95_inf": round(lo, 3), "ic95_sup": round(hi, 3),
                      "interpretacion": interpretar(k)})
    if len(anotadores) >= 3:
        t = tabla[anotadores].dropna().to_numpy()
        k = fleiss_kappa(t)
        lo, hi = bootstrap_ci(fleiss_kappa, t)
        filas.append({"comparacion": " + ".join(anotadores), "tipo": "Fleiss", "n_fotos": len(t),
                      "acuerdo_observado": round(float(np.mean([len(set(r)) == 1 for r in t])), 3),
                      "kappa": round(k, 3), "ic95_inf": round(lo, 3), "ic95_sup": round(hi, 3),
                      "interpretacion": interpretar(k)})
    res = pd.DataFrame(filas)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULT_CSV, index=False)
    print(res.to_string(index=False))
    print(f"\nGuardado en {RESULT_CSV.relative_to(ROOT)}")
    # Dónde se reparten los desacuerdos con la etiqueta de carpeta
    for a in anotadores:
        print(f"\n{a} (filas) frente a la carpeta (columnas):")
        print(pd.crosstab(tabla[a], tabla["carpeta"]).reindex(index=CLASES, columns=CLASES, fill_value=0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("muestra")
    m.add_argument("--por-clase", type=int, default=20)
    m.add_argument("--seed", type=int, default=42)
    sub.add_parser("calcular")
    args = ap.parse_args()
    if args.cmd == "muestra":
        cmd_muestra(args.por_clase, args.seed)
    else:
        cmd_calcular()


if __name__ == "__main__":
    main()
