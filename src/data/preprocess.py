"""Preparación de los datos (fase CRISP-DM: Preparación).

Para cada imagen etiquetada del índice:
  1. Lee y reduce la resolución (las fotos de celular llegan a 4000 px).
  2. Segmenta la fruta y mide su tamaño (src/data/segmentation.py).
  3. Guarda un recorte cuadrado IMG_SIZE x IMG_SIZE con fondo enmascarado.
  4. Asigna la clase de tamaño y la partición train/val/test.

Clase de tamaño: si el grupo anotó `size_manual`, se usa esa etiqueta. Si no,
se usan los terciles del diámetro normalizado POR FRUTA (una manzana grande
no mide lo mismo que una naranja grande):
    pequeno  si d_norm <= P33(fruta)
    mediano  si P33 < d_norm <= P66
    grande   si d_norm >  P66

Partición: estratificada por calidad y AGRUPADA por fotos casi idénticas
(ráfagas de la misma fruta), para que ninguna fruta física aparezca a la vez
en train y en test. Ver near_duplicate_groups().
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data.segmentation import crop_square, load_image, segment


def size_class_from_terciles(df: pd.DataFrame) -> pd.Series:
    """Terciles por (fuente, fruta): las imágenes de Kaggle vienen recortadas a
    256x256 y la fruta ocupa más cuadro que en nuestras fotos, así que su
    d_norm no es comparable con el de las fotos propias."""
    out = pd.Series(index=df.index, dtype=object)
    for _, g in df.groupby(["source", "fruit"]):
        q33, q66 = g["d_norm"].quantile([1 / 3, 2 / 3])
        out[g.index] = np.select(
            [g["d_norm"] <= q33, g["d_norm"] <= q66], ["pequeno", "mediano"], "grande"
        )
    return out


def _embedding(crop_path) -> np.ndarray:
    """Vector 32x32x3 centrado y normalizado: similitud coseno = correlación."""
    g = cv2.resize(cv2.imread(str(crop_path)), (32, 32)).astype(np.float32).ravel()
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-8)


def near_duplicate_groups(df: pd.DataFrame, thr: float = config.DUP_SIM_THRESHOLD) -> pd.Series:
    """Agrupa fotos casi idénticas (ráfagas de la misma fruta).

    Dos recortes i, j de la misma carpeta (fruta, calidad, fuente) quedan en el
    mismo grupo si cos(e_i, e_j) > thr. Los grupos son las componentes conexas
    del grafo de similitud (union-find). Sin esto, fotos consecutivas de la
    misma fruta caen en train y test a la vez y el test sobreestima el
    desempeño (fuga de datos): con el split aleatorio, 53 de 91 imágenes de
    test tenían un «gemelo» en train.
    """
    parent = list(range(len(df)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    emb = np.stack([_embedding(config.ROOT / c) for c in df["crop"]])
    for _, g in df.groupby(["source", "fruit", "quality"]):
        idx = g.index.to_numpy()
        sim = emb[idx] @ emb[idx].T
        ii, jj = np.where(np.triu(sim, 1) > thr)
        for a, b in zip(idx[ii], idx[jj]):
            parent[find(a)] = find(b)
    roots = [find(i) for i in range(len(df))]
    return pd.Series(pd.factorize(roots)[0], index=df.index)


def grouped_stratified_split(df: pd.DataFrame, seed: int = config.SEED) -> pd.Series:
    """Partición estratificada por calidad y AGRUPADA: un grupo de casi
    duplicados nunca queda repartido entre train, val y test."""
    from sklearn.model_selection import StratifiedGroupKFold

    y = df["quality"].to_numpy()
    groups = df["group"].to_numpy()
    k_test = max(2, round(1 / config.TEST_SIZE))
    _, te = next(StratifiedGroupKFold(k_test, shuffle=True, random_state=seed).split(df, y, groups))
    rest = np.setdiff1d(np.arange(len(df)), te)
    k_val = max(2, round((1 - config.TEST_SIZE) / config.VAL_SIZE))
    _, va_rel = next(StratifiedGroupKFold(k_val, shuffle=True, random_state=seed)
                     .split(rest, y[rest], groups[rest]))
    split = pd.Series("train", index=df.index)
    split.iloc[rest[va_rel]] = "val"
    split.iloc[te] = "test"
    return split


def main() -> pd.DataFrame:
    df = pd.read_csv(config.INDEX_CSV)
    df = df[df["quality"].isin(config.QUALITY_CLASSES)].reset_index(drop=True)
    if df.empty:
        raise SystemExit("No hay imágenes etiquetadas. Ejecuta primero download e index.")

    crops_dir = config.PROCESSED_DIR / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    for i, row in df.iterrows():
        img = load_image(config.ROOT / row["path"])
        seg = segment(img)
        crop = crop_square(img, seg)
        crop_path = crops_dir / f"{i:05d}.png"
        cv2.imwrite(str(crop_path), crop)
        metrics.append(
            {
                "crop": str(crop_path.relative_to(config.ROOT)),
                "d_norm": seg.d_norm,
                "circularity": seg.circularity,
                "solidity": seg.solidity,
                "aspect_ratio": seg.aspect_ratio,
                "seg_ok": seg.ok,
                "height": img.shape[0],
                "width": img.shape[1],
            }
        )
    df = pd.concat([df, pd.DataFrame(metrics)], axis=1)

    df["size"] = size_class_from_terciles(df)
    if "size_manual" in df.columns:
        manual = df["size_manual"].where(df["size_manual"].isin(config.SIZE_CLASSES))
        df["size"] = manual.fillna(df["size"])

    df["group"] = near_duplicate_groups(df)
    df["split"] = grouped_stratified_split(df)
    df.to_csv(config.SPLITS_CSV, index=False)
    print(f"[preprocess] {df['group'].nunique()} grupos de fotos casi idénticas "
          f"(umbral coseno {config.DUP_SIM_THRESHOLD}) para {len(df)} imágenes")

    n_bad = int((~df["seg_ok"]).sum())
    print(f"[preprocess] {len(df)} recortes en {crops_dir}")
    print(f"[preprocess] segmentaciones dudosas: {n_bad} (revisar en el notebook de EDA)")
    print(pd.crosstab(df["split"], df["quality"]))
    return df


if __name__ == "__main__":
    main()
