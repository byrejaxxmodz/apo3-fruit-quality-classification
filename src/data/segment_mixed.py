"""Recorta cada fruta de las imágenes de la carpeta mixed_quality (Kaggle).

La consigna pide segmentar individualmente cada fruta de las fotos con varias
frutas. Los recortes quedan en:

    data/interim/mixed_crops/sin_etiquetar/<fruta>_mixed/<img>_<k>.png

Para usarlos en entrenamiento, el grupo los revisa y los MUEVE a mano a
    data/interim/mixed_crops/<Good|Regular|Bad>_Quality_Fruits/<fruta>_<calidad>/
y vuelve a ejecutar `python -m src.main index`.
"""
from __future__ import annotations

import cv2

from src import config
from src.data.dataset import parse_fruit, parse_quality
from src.data.segmentation import crop_square, load_image, segment_all


def main(min_frac: float = 0.01, size: int = 256) -> int:
    out_root = config.INTERIM_DIR / "mixed_crops" / "sin_etiquetar"
    n = 0
    for p in sorted(config.RAW_KAGGLE.rglob("*")):
        if p.suffix.lower() not in config.IMAGE_EXTS:
            continue
        rel = p.relative_to(config.RAW_KAGGLE)
        if parse_quality(rel) != "mixed":
            continue
        img = load_image(p, max_side=1024)
        fruit = parse_fruit(rel)
        out_dir = out_root / f"{fruit}_mixed"
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, seg in enumerate(segment_all(img, min_frac=min_frac)):
            crop = crop_square(img, seg, size=size, mask_bg=False)
            cv2.imwrite(str(out_dir / f"{p.stem}_{k}.png"), crop)
            n += 1
    print(f"[mixed] {n} recortes guardados en {out_root}")
    return n


if __name__ == "__main__":
    main()
