"""Descarga de los datos crudos (fase CRISP-DM: Comprensión de los datos).

Fuentes:
  1. Kaggle  - Fruit Quality Classification (ryandpark). Requiere credenciales
     de Kaggle (~/.kaggle/kaggle.json o variables KAGGLE_USERNAME/KAGGLE_KEY).
  2. Drive   - Carpeta "proyectoAPO" con las fotos tomadas por el grupo.
     La carpeta debe estar compartida como "Cualquier persona con el enlace".

Si alguna descarga automática falla, se pueden copiar las carpetas a mano en
data/raw/kaggle/ y data/raw/propias/ respetando la estructura original.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from src import config


def download_kaggle(dest: Path = config.RAW_KAGGLE) -> Path:
    """Descarga el dataset de Kaggle (~3,3 GB, 19.526 imágenes, licencia CC0).

    Primero intenta con kagglehub (usa ~/.kaggle/kaggle.json si existe). Si no
    hay credenciales, usa la API pública de descarga, que para este dataset
    funciona sin cuenta.
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub  # import diferido: solo se necesita para esta función

        src_path = Path(kagglehub.dataset_download(config.KAGGLE_DATASET))
        shutil.copytree(src_path, dest, dirs_exist_ok=True)
    except Exception as e:  # sin credenciales u otro error de kagglehub
        import urllib.request
        import zipfile

        print(f"[kaggle] kagglehub falló ({e.__class__.__name__}); descarga directa sin cuenta...")
        url = f"https://www.kaggle.com/api/v1/datasets/download/{config.KAGGLE_DATASET}"
        zip_path = dest / "archive.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():  # evita rutas absolutas o con '..'
                if name.startswith("/") or ".." in Path(name).parts:
                    raise ValueError(f"Ruta insegura en el zip: {name}")
            z.extractall(dest)
        zip_path.unlink()
    print(f"[kaggle] dataset en {dest}")
    return dest


def download_drive(dest: Path = config.RAW_PROPIAS) -> Path:
    """Descarga la carpeta compartida de Google Drive con gdown."""
    import gdown

    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{config.DRIVE_FOLDER_ID}"
    gdown.download_folder(url, output=str(dest), quiet=False, remaining_ok=True)
    print(f"[drive] fotos propias descargadas en {dest}")
    return dest


def main(kaggle: bool = True, drive: bool = True) -> None:
    if drive:
        download_drive()
    if kaggle:
        download_kaggle()


if __name__ == "__main__":
    main()
