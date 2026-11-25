"""Construcción del índice de imágenes a partir de la estructura de carpetas.

Tanto Kaggle como la carpeta de Drive del grupo organizan las imágenes así:

    <Calidad>_Quality_Fruits/<Fruta>_<calidad>/imagen.jpg
    p. ej.  Good_Quality_Fruits/Apple_good/1794858713032.jpg
            Regular_Quality_Fruits/banana_regular/xxx.jpg
            Bad Quality_Fruits/Banana_Bad/xxx.png        (Kaggle)

La calidad se deduce de las palabras clave de la ruta (config.QUALITY_KEYWORDS)
y la fruta del primer token del nombre de la subcarpeta. Si el archivo
data/anotaciones_propias.csv existe, sus etiquetas manuales tienen prioridad.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config


def parse_quality(path: Path) -> Optional[str]:
    """Devuelve 'good' | 'regular' | 'bad' | 'mixed' o None según la ruta."""
    # Se recorre de la carpeta más interna a la más externa
    for part in reversed(path.parent.parts):
        tokens = re.split(r"[\s_\-]+", part.lower())
        for tok in tokens:
            for key, label in config.QUALITY_KEYWORDS.items():
                if tok.startswith(key):
                    return label
    return None


def parse_fruit(path: Path) -> str:
    """Primer token de la subcarpeta: 'Apple_good' -> 'apple'.

    Si la foto está suelta en la carpeta de calidad (sin subcarpeta de fruta),
    la fruta se busca en el nombre del archivo:
    'Bad_Quality_Fruits/grupoAPO_granadilla_mala_01.jpg' -> 'granadilla'."""
    leaf = path.parent.name
    token = re.split(r"[\s_\-]+", leaf.strip())[0].lower()
    if token not in config.QUALITY_KEYWORDS:
        return config.FRUIT_ALIASES.get(token, token)
    for tok in re.split(r"[\s_\-]+", path.stem.lower()):  # carpeta sin fruta en el nombre
        fruta = config.FRUIT_ALIASES.get(tok, tok)
        if fruta in config.KNOWN_FRUITS:
            return fruta
    return "desconocida"


def scan_folder(root: Path, source: str) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in config.IMAGE_EXTS or p.name.startswith("."):
            continue
        rel = p.relative_to(root)  # solo se analizan carpetas dentro de la fuente
        rows.append(
            {
                "path": str(p.relative_to(config.ROOT)),
                "source": source,
                "fruit": parse_fruit(rel),
                "quality": parse_quality(rel),
            }
        )
    return pd.DataFrame(rows, columns=["path", "source", "fruit", "quality"])


def apply_manual_annotations(df: pd.DataFrame, csv: Path = config.ANOTACIONES_CSV) -> pd.DataFrame:
    """Sobrescribe calidad/fruta/tamaño con el etiquetado manual del grupo."""
    if not csv.exists():
        return df
    ann = pd.read_csv(csv)
    ann = ann.dropna(subset=["path"]).set_index("path")
    for col in ("quality", "fruit", "size_manual"):
        if col not in ann.columns:
            continue
        if col not in df.columns:
            df[col] = None
        m = df["path"].isin(ann.index)
        vals = df.loc[m, "path"].map(ann[col])
        df.loc[m & vals.notna(), col] = vals[vals.notna()]
    return df


def write_annotation_template(df: pd.DataFrame, csv: Path = config.ANOTACIONES_CSV) -> None:
    """Crea la plantilla de anotación manual de las fotos propias o, si ya
    existe, le agrega al final las fotos nuevas sin tocar las filas anteriores
    (que pueden tener anotaciones hechas a mano)."""
    own = df[df["source"] == "propias"][["path", "fruit", "quality"]].copy()
    own["size_manual"] = ""   # pequeno / mediano / grande
    own["defectos"] = ""      # golpe, mancha, podredumbre, ninguno...
    own["anotador"] = ""
    if not csv.exists():
        own.to_csv(csv, index=False)
        print(f"[dataset] plantilla de anotación creada: {csv} ({len(own)} filas)")
        return
    prev = pd.read_csv(csv, dtype=str, keep_default_na=False)
    nuevas = own[~own["path"].isin(prev["path"])]
    if len(nuevas):
        pd.concat([prev, nuevas.astype(str)], ignore_index=True).to_csv(csv, index=False)
        print(f"[dataset] {len(nuevas)} fotos nuevas agregadas a {csv.name} para anotar")


def build_index() -> pd.DataFrame:
    kaggle = scan_folder(config.RAW_KAGGLE, "kaggle")
    if len(kaggle):
        folder = kaggle["path"].map(lambda s: str(Path(s).parent))
        kaggle = (kaggle.groupby(folder, group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), config.KAGGLE_MAX_PER_FOLDER),
                                            random_state=config.SEED))
                  .sort_values("path").reset_index(drop=True))
    frames = [
        kaggle,
        scan_folder(config.RAW_PROPIAS, "propias"),
        scan_folder(config.INTERIM_DIR / "mixed_crops", "kaggle_mixed"),
    ]
    df = pd.concat(frames, ignore_index=True)

    # Procedencia: excluir imágenes generadas con IA (ver src/data/provenance.py)
    if config.PROVENANCE_CSV.exists():
        prov = pd.read_csv(config.PROVENANCE_CSV).set_index("path")["ai_generated"]
        df["ai_generated"] = df["path"].map(prov).fillna(False).astype(bool)
    else:
        print("[dataset] AVISO: no se ha ejecutado el paso 'provenance'")
        df["ai_generated"] = False
    n_ai = int(df["ai_generated"].sum())
    if config.EXCLUDE_AI_GENERATED and n_ai:
        print(f"[dataset] se excluyen {n_ai} imágenes con marcas de IA")
        df = df[~df["ai_generated"]].reset_index(drop=True)

    write_annotation_template(df)
    df = apply_manual_annotations(df)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.INDEX_CSV, index=False)
    print(f"[dataset] {len(df)} imágenes indexadas -> {config.INDEX_CSV}")
    if len(df):
        print(pd.crosstab([df["source"], df["fruit"]], df["quality"].fillna("?")))
    return df


if __name__ == "__main__":
    build_index()
