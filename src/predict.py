"""Inferencia sobre una imagen nueva: calidad + tamaño (fase: Despliegue).

Reutiliza exactamente el mismo preprocesamiento del entrenamiento
(segmentación -> recorte con fondo negro -> características / tensor), para
evitar diferencias entre entrenamiento y producción.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data.segmentation import Segmentation, crop_square, resize_max_side, segment
from src.features.extract import extract


@dataclass
class Prediction:
    quality: str
    probs: Dict[str, float]
    size: str
    d_norm: float
    seg: Segmentation
    crop: np.ndarray


@lru_cache(maxsize=1)
def size_thresholds() -> Dict[str, tuple]:
    """Terciles de d_norm calculados en entrenamiento (por fruta y global)."""
    if not config.SPLITS_CSV.exists():
        return {"_global": (0.35, 0.55)}
    df = pd.read_csv(config.SPLITS_CSV)
    df = df[df["split"] == "train"]
    # La app recibe fotos de celular: se usan los umbrales de las fotos propias
    if (df["source"] == "propias").any():
        df = df[df["source"] == "propias"]
    th = {f: tuple(g["d_norm"].quantile([1 / 3, 2 / 3])) for f, g in df.groupby("fruit")}
    th["_global"] = tuple(df["d_norm"].quantile([1 / 3, 2 / 3]))
    return th


def size_class(d_norm: float, fruit: str = "_global") -> str:
    q33, q66 = size_thresholds().get(fruit, size_thresholds()["_global"])
    return "pequeno" if d_norm <= q33 else ("mediano" if d_norm <= q66 else "grande")


@lru_cache(maxsize=4)
def load_model(name: str):
    if name == "cnn":
        import torch

        from src.models.cnn import SmallCNN
        ckpt = torch.load(config.CHECKPOINTS_DIR / "cnn.pt", map_location="cpu")
        m = SmallCNN(len(config.QUALITY_CLASSES), dropout=ckpt["dropout"])
        m.load_state_dict(ckpt["state_dict"])
        return m.eval()
    return joblib.load(config.CHECKPOINTS_DIR / f"{name}.joblib")


def available_models():
    names = [p.stem for p in sorted(config.CHECKPOINTS_DIR.glob("*.joblib"))]
    if (config.CHECKPOINTS_DIR / "cnn.pt").exists():
        names.append("cnn")
    return names


def predict(img_bgr: np.ndarray, model_name: str, fruit: str = "_global") -> Prediction:
    img = resize_max_side(img_bgr)
    seg = segment(img)
    crop = crop_square(img, seg)
    model = load_model(model_name)
    if model_name == "cnn":
        import torch

        from src.training.train_cnn import to_tensor
        with torch.no_grad():
            p = torch.softmax(model(to_tensor(crop)[None]), 1)[0].numpy()
    else:
        x = extract(crop, seg.circularity, seg.solidity, seg.aspect_ratio)[None]
        p = model.predict_proba(x)[0]
    probs = {c: float(v) for c, v in zip(config.QUALITY_CLASSES, p)}
    return Prediction(
        quality=max(probs, key=probs.get), probs=probs,
        size=size_class(seg.d_norm, fruit), d_norm=seg.d_norm, seg=seg, crop=crop,
    )
