"""Normaliza el formato de las fotos propias.

- La mayoría de las fotos del grupo vienen en HEIC (formato del iPhone), que
  OpenCV no puede leer. Se convierten a JPG con pillow-heif.
- Se aplica la orientación EXIF (si no, la foto puede quedar girada) y se
  guarda SIN metadatos. Esto también quita la ubicación GPS y el modelo del
  teléfono: es una medida de privacidad (ver docs/etica_impactos.md).
- A los archivos sin extensión se les agrega la extensión de su formato real.

Los HEIC originales se conservan; el índice solo lee los formatos de
config.IMAGE_EXTS, así que no aparecen duplicados.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from src import config

HEIC_EXTS = {".heic", ".heif"}


def _register_heif() -> None:
    from pillow_heif import register_heif_opener
    register_heif_opener()


def fix_missing_extension(p: Path) -> Path:
    with Image.open(p) as im:
        ext = {"JPEG": ".jpg", "PNG": ".png", "HEIF": ".heic", "WEBP": ".webp"}.get(im.format, "")
    if not ext:
        return p
    new = p.with_name(p.name + ext)
    p.rename(new)
    return new


def heic_to_jpg(p: Path, quality: int = 92) -> Path:
    out = p.with_suffix(".jpg")
    if out.exists():
        out = p.with_name(f"{p.stem}_heic.jpg")
    if out.exists():
        return out
    with Image.open(p) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.save(out, "JPEG", quality=quality)  # sin exif=... -> sin metadatos
    return out


def main(root: Path = config.RAW_PROPIAS) -> None:
    _register_heif()
    n_ext = n_heic = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix == "":
            p = fix_missing_extension(p)
            n_ext += 1
        if p.suffix.lower() in HEIC_EXTS:
            heic_to_jpg(p)
            n_heic += 1
    print(f"[convert] {n_heic} HEIC convertidos a JPG, {n_ext} extensiones corregidas")


if __name__ == "__main__":
    main()
