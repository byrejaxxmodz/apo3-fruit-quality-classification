"""Verificación de procedencia: detecta imágenes generadas con IA.

Generadores como ChatGPT/DALL·E, Gemini o Firefly incrustan un manifiesto
C2PA ("Content Credentials") o la etiqueta IPTC `trainedAlgorithmicMedia`
en el archivo. Este script busca esas marcas en los bytes de cada imagen
cruda y genera data/processed/procedencia.csv.

Por qué importa: la consigna pide fotos REALES de frutas. Una imagen sintética
no muestra un defecto real, introduce un sesgo (el modelo puede aprender el
«estilo» del generador en vez de la calidad) y presentarla como recolección
propia sería un problema de honestidad académica. Por defecto el índice las
excluye (config.EXCLUDE_AI_GENERATED).

Limitación: si a la imagen le quitaron los metadatos (captura de pantalla,
re-guardado, WhatsApp), la marca desaparece. Que no haya marca NO garantiza
que la foto sea real: hay que complementar con una revisión visual.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src import config

MARKERS = re.compile(
    rb"trainedAlgorithmicMedia|compositeWithTrainedAlgorithmicMedia|OpenAI|ChatGPT|DALL.?E|"
    rb"Gemini|SynthID|Google Imagen|Adobe Firefly|Midjourney|Stable Diffusion"
)


def scan_file(p: Path) -> str:
    """Devuelve las marcas encontradas separadas por coma ('' si ninguna)."""
    data = p.read_bytes()
    found = sorted({m.decode(errors="ignore") for m in MARKERS.findall(data)})
    return ",".join(found)


def main() -> pd.DataFrame:
    rows = []
    for root in (config.RAW_PROPIAS, config.RAW_KAGGLE):
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.name.startswith(".") or p.name == ".gitkeep":
                continue
            marks = scan_file(p)
            # Un JPG convertido desde HEIC perdió los metadatos: revisar el original
            for ext in (".heic", ".HEIC", ".heif", ".HEIF"):
                orig = p.with_suffix(ext)
                if p.suffix.lower() == ".jpg" and orig.exists():
                    marks = ",".join(filter(None, [marks, scan_file(orig)]))
            rows.append({"path": str(p.relative_to(config.ROOT)), "ai_markers": marks,
                         "ai_generated": bool(marks)})
    df = pd.DataFrame(rows, columns=["path", "ai_markers", "ai_generated"])
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROVENANCE_CSV, index=False)
    n = int(df["ai_generated"].sum()) if len(df) else 0
    print(f"[procedencia] {n} de {len(df)} archivos con marcas de IA -> {config.PROVENANCE_CSV}")
    if n:
        flagged = df[df.ai_generated].copy()
        flagged["carpeta"] = flagged["path"].map(lambda s: str(Path(s).parent))
        print(flagged.groupby("carpeta").size().to_string())
    return df


if __name__ == "__main__":
    main()
