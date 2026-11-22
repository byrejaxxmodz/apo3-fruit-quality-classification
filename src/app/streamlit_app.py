"""Interfaz gráfica de despliegue.

Ejecutar desde la raíz del repo:
    streamlit run src/app/streamlit_app.py

Permite subir una foto o tomarla con la cámara y muestra:
  - clase de calidad y probabilidad de cada clase
  - tamaño estimado (pequeño / mediano / grande) y diámetro normalizado
  - la segmentación usada, para que el usuario vea en qué se basó el modelo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # permite `import src`

import cv2
import numpy as np
import streamlit as st

from src.data.segmentation import overlay
from src.predict import available_models, predict

st.set_page_config(page_title="Calidad de frutas", page_icon="🍎", layout="centered")
st.title("Clasificador de calidad de frutas")
st.caption("Proyecto final · Algoritmos y Programación III · Universidad Icesi")

models = available_models()
if not models:
    st.error("No hay modelos entrenados. Ejecuta `python -m src.main all` primero.")
    st.stop()

with st.sidebar:
    model_name = st.selectbox("Modelo", models, index=models.index("cnn") if "cnn" in models else 0)
    fruit = st.selectbox("Fruta (para el umbral de tamaño)",
                         ["_global", "apple", "banana", "orange", "lime", "guava", "pomegranate"])
    st.markdown("**Recomendaciones de captura**: una sola fruta, fondo liso y claro, buena luz.")

source = st.radio("Origen de la imagen", ["Subir archivo", "Cámara"], horizontal=True)
file = st.file_uploader("Imagen", type=["jpg", "jpeg", "png", "heic", "heif"]) \
    if source == "Subir archivo" else st.camera_input("Toma una foto")


def decode(data: bytes):
    """JPG/PNG con OpenCV; HEIC (iPhone) con pillow-heif."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    import io

    from PIL import Image, ImageOps
    from pillow_heif import register_heif_opener
    register_heif_opener()
    pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


if file is not None:
    try:
        img = decode(file.getvalue())
    except Exception:
        st.error("No se pudo leer la imagen. Usa JPG, PNG o HEIC.")
        st.stop()
    pred = predict(img, model_name, fruit)

    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(overlay(cv2.resize(img, pred.seg.mask.shape[::-1]), pred.seg),
                          cv2.COLOR_BGR2RGB), caption="Segmentación")
    c2.image(cv2.cvtColor(pred.crop, cv2.COLOR_BGR2RGB), caption="Entrada del modelo")

    st.subheader(f"Calidad: **{pred.quality.upper()}**")
    st.bar_chart(pred.probs)
    st.subheader(f"Tamaño: **{pred.size}**  (d_norm = {pred.d_norm:.3f})")
    if not pred.seg.ok:
        st.warning("La segmentación es dudosa: revisa el fondo o la iluminación.")
