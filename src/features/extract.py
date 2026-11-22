"""Extracción manual de características para los modelos clásicos y el MLP.

Cada recorte (fondo en negro) se describe con un vector x de dimensión fija:

| Bloque            | Dim | Qué captura                                      |
|-------------------|-----|--------------------------------------------------|
| Histograma HSV    | 34  | color / madurez (H: 18, S: 8, V: 8 bins)         |
| Momentos Lab      | 9   | media, desviación y asimetría por canal L, a, b  |
| LBP uniforme      | 10  | micro-textura (arrugas, moho, piel lisa)         |
| GLCM              | 4   | contraste, homogeneidad, energía, correlación    |
| Defectos          | 2   | % píxeles oscuros y % píxeles "marrones"         |
| Forma             | 3   | circularidad, solidez, relación de aspecto       |

Total: 62 características. Todas se calculan SOLO sobre los píxeles de la
máscara de la fruta. El tamaño (d_norm) no se incluye a propósito: la calidad
no debería depender de si la fruta es grande o pequeña.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from src import config

H_BINS, S_BINS, V_BINS = 18, 8, 8
LBP_P, LBP_R = 8, 1


def feature_names() -> List[str]:
    names = [f"h{i}" for i in range(H_BINS)] + [f"s{i}" for i in range(S_BINS)] + [f"v{i}" for i in range(V_BINS)]
    names += [f"{st}_{c}" for c in "Lab" for st in ("mean", "std", "skew")]
    names += [f"lbp{i}" for i in range(LBP_P + 2)]
    names += ["glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation"]
    names += ["dark_ratio", "brown_ratio", "circularity", "solidity", "aspect_ratio"]
    return names


def _hist(values: np.ndarray, bins: int, rng) -> np.ndarray:
    h, _ = np.histogram(values, bins=bins, range=rng)
    return h / max(h.sum(), 1)  # normalizado: distribución de probabilidad


def _moments(x: np.ndarray) -> List[float]:
    mu, sd = float(x.mean()), float(x.std())
    skew = float(((x - mu) ** 3).mean() / (sd**3 + 1e-8))
    return [mu, sd, skew]


def extract(crop_bgr: np.ndarray, circularity: float = 0.0, solidity: float = 0.0,
            aspect_ratio: float = 1.0) -> np.ndarray:
    mask = crop_bgr.max(axis=2) > 0
    if mask.sum() < 20:  # recorte vacío: se usa toda la imagen
        mask = np.ones(mask.shape, bool)

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0][mask], hsv[..., 1][mask], hsv[..., 2][mask]
    f = [_hist(h, H_BINS, (0, 180)), _hist(s, S_BINS, (0, 256)), _hist(v, V_BINS, (0, 256))]

    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    f.append(np.array(sum((_moments(lab[..., c][mask]) for c in range(3)), [])))

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")
    f.append(_hist(lbp[mask], LBP_P + 2, (0, LBP_P + 2)))

    # GLCM sobre la caja de la fruta, cuantizada a 32 niveles
    ys, xs = np.where(mask)
    g = (gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1] // 8).astype(np.uint8)
    glcm = graycomatrix(g, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    f.append(np.array([graycoprops(glcm, p).mean() for p in
                       ("contrast", "homogeneity", "energy", "correlation")]))

    # Defectos: píxeles mucho más oscuros que la mediana (golpes, podredumbre)
    # y píxeles de tono marrón (H 5-25 con V medio-bajo)
    med_v = np.median(v) + 1e-6
    dark = float((v < 0.5 * med_v).mean())
    brown = float(((h >= 5) & (h <= 25) & (v < 0.6 * 255) & (s > 50)).mean())
    f.append(np.array([dark, brown, circularity, solidity, aspect_ratio]))

    return np.concatenate(f).astype(np.float32)


def build_feature_matrix() -> None:
    df = pd.read_csv(config.SPLITS_CSV)
    X = np.stack([
        extract(cv2.imread(str(config.ROOT / r.crop)), r.circularity, r.solidity, r.aspect_ratio)
        for r in df.itertuples()
    ])
    y = df["quality"].map({c: i for i, c in enumerate(config.QUALITY_CLASSES)}).to_numpy()
    np.savez_compressed(config.FEATURES_NPZ, X=X, y=y, split=df["split"].to_numpy(),
                        source=df["source"].to_numpy(), group=df["group"].to_numpy(),
                        names=np.array(feature_names()))
    print(f"[features] X={X.shape} -> {config.FEATURES_NPZ}")


if __name__ == "__main__":
    build_feature_matrix()
