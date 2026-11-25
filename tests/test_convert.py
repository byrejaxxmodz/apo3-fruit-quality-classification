"""Pruebas de la conversión HEIC -> JPG (src/data/convert.py)."""
import pytest
from PIL import Image

pytest.importorskip("pillow_heif")
from src.data import convert  # noqa: E402

convert._register_heif()


def _heic(path):
    Image.new("RGB", (32, 32), (200, 30, 30)).save(path, format="HEIF")
    return path


def test_heic_to_jpg_es_idempotente(tmp_path):
    heic = _heic(tmp_path / "IMG_1.HEIC")
    primera = convert.heic_to_jpg(heic)
    segunda = convert.heic_to_jpg(heic)   # antes creaba IMG_1_heic.jpg (foto duplicada)
    assert primera == segunda == tmp_path / "IMG_1.jpg"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["IMG_1.HEIC", "IMG_1.jpg"]


def test_heic_to_jpg_no_pisa_otra_foto_con_el_mismo_nombre(tmp_path):
    exif = Image.Exif()
    exif[0x010F] = "Apple"   # una foto original de cámara trae EXIF
    Image.new("RGB", (32, 32)).save(tmp_path / "IMG_1.jpg", exif=exif)
    heic = _heic(tmp_path / "IMG_1.HEIC")
    out = convert.heic_to_jpg(heic)
    assert out == tmp_path / "IMG_1_heic.jpg"
    assert convert.heic_to_jpg(heic) == out   # y tampoco se duplica en la segunda corrida
