"""Pruebas unitarias de las piezas deterministas del pipeline."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.dataset import parse_fruit, parse_quality
from src.data.segmentation import crop_square, segment
from src.features.extract import extract, feature_names


def synthetic_fruit(radius=80, size=400, color=(30, 60, 200)):
    """Círculo rojo sobre fondo blanco: fruta ideal con área conocida."""
    img = np.full((size, size, 3), 235, np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, color, -1)
    return img


@pytest.mark.parametrize("path,quality,fruit", [
    ("Good_Quality_Fruits/Apple_good/1.jpg", "good", "apple"),
    ("Regular_Quality_Fruits/banana_regular/x.jpg", "regular", "banana"),
    ("Bad Quality_Fruits/Banana_Bad/x.png", "bad", "banana"),
    ("Mixed Qualit_Fruits/Orange/x.png", "mixed", "orange"),
    ("Bad_Quality_Fruits/Orage_Bad/x.jpg", "bad", "orange"),  # alias
    # fotos sueltas en la carpeta de calidad: la fruta sale del nombre del archivo
    ("Bad_Quality_Fruits/grupoAPO_granadilla_mala_01.jpg", "bad", "granadilla"),
    ("Regular_Quality_Fruits/grupoAPO_guayaba_regular_03.jpg", "regular", "guava"),
    ("Bad_Quality_Fruits/foto_sin_fruta.jpg", "bad", "desconocida"),
])
def test_parse_labels(path, quality, fruit):
    p = Path(path)
    assert parse_quality(p) == quality
    assert parse_fruit(p) == fruit


def test_segmentation_recovers_diameter():
    seg = segment(synthetic_fruit(radius=80))
    assert seg.ok
    # d_eq debe aproximar el diámetro real (160 px) con un error < 3 %
    assert abs(seg.d_eq - 160) / 160 < 0.03
    assert seg.circularity > 0.85


def test_size_monotonic():
    small = segment(synthetic_fruit(radius=40)).d_norm
    big = segment(synthetic_fruit(radius=120)).d_norm
    assert small < big


def test_crop_and_features_shape():
    img = synthetic_fruit()
    seg = segment(img)
    crop = crop_square(img, seg)
    assert crop.shape == (128, 128, 3)
    assert crop[0, 0].sum() == 0  # el fondo queda en negro
    x = extract(crop, seg.circularity, seg.solidity, seg.aspect_ratio)
    assert x.shape == (len(feature_names()),)
    assert np.isfinite(x).all()


def test_cnn_forward():
    torch = pytest.importorskip("torch")
    from src.models.cnn import SmallCNN
    out = SmallCNN(3)(torch.zeros(2, 3, 128, 128))
    assert out.shape == (2, 3)
