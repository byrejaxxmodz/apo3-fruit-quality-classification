"""Interfaz gráfica de despliegue.

Ejecutar desde la raíz del repo, con el entorno del proyecto:
    python -m streamlit run src/app/streamlit_app.py

Cuatro formas de dar la imagen: subir una foto, una foto de ejemplo, tomar una
foto con la cámara del navegador o la cámara en vivo (clasifica mientras la
fruta está frente a la cámara del computador o del iPhone). Para cada imagen muestra:
  - la clase de calidad y la probabilidad de cada clase
  - el tamaño estimado (pequeño / mediano / grande) y el diámetro normalizado
  - la segmentación usada, para que el usuario vea en qué se basó el modelo
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # permite `import src`

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.app.camara import abrir, listar_camaras
from src.data.segmentation import overlay
from src.predict import available_models, predict

EJEMPLOS = Path(__file__).parent / "ejemplos"

NOMBRES_MODELO = {
    "svm": "SVM (recomendado)",
    "knn": "KNN",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "random_forest": "Random Forest",
    "logreg": "Regresión logística",
    "cnn": "CNN desde cero",
}
FRUTAS = {"Cualquiera": "_global", "Manzana": "apple", "Banano": "banana", "Granadilla": "granadilla",
          "Guayaba": "guava", "Lima": "lime", "Naranja": "orange"}
CLASES = {"good": ("Buena", "green"), "regular": ("Regular", "orange"), "bad": ("Mala", "red")}
TAMANOS = {"pequeno": "pequeño", "mediano": "mediano", "grande": "grande"}
CUADROS_POR_SEGUNDO = 5  # en vivo: suficiente para ver el resultado sin saturar el equipo


@st.cache_data(ttl=20)
def camaras_disponibles() -> list:
    """Nombres de las cámaras (vía ffmpeg); sin ffmpeg, solo la cámara 0 de OpenCV."""
    return listar_camaras() or [0]


@st.cache_data
def f1_fotos_propias() -> dict:
    """F1 macro de cada modelo en las fotos propias de prueba (para mostrarlo al elegir)."""
    ruta = config.RESULTS_DIR / "f1_por_fuente.csv"
    if not ruta.exists():
        return {}
    df = pd.read_csv(ruta)
    return df[df["fuente"] == "propias"].set_index("modelo")["f1_macro"].to_dict()


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


def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def tarjeta_resultado(pred, contenedor):
    """Calidad, probabilidades en orden good/regular/bad y tamaño."""
    nombre, color = CLASES[pred.quality]
    with contenedor.container(border=True):
        st.markdown(f"#### Calidad: :{color}[**{nombre}**] ({pred.quality})")
        for clase, (etiqueta, _) in CLASES.items():
            p = float(pred.probs.get(clase, 0.0))
            st.progress(p, text=f"{etiqueta} ({clase}): {p:.0%}")
        st.markdown(f"#### Tamaño: **{TAMANOS.get(pred.size, pred.size)}**")
        st.caption(f"Diámetro normalizado: {pred.d_norm:.3f}. Es relativo a la foto, no son centímetros.")
        if not pred.seg.ok:
            st.warning("La segmentación es dudosa: revisa el fondo o la iluminación.")


def mostrar(img, model_name, fruit):
    pred = predict(img, model_name, fruit)
    alto, ancho = img.shape[:2]
    if max(alto, ancho) < 1000:
        st.warning(f"La foto es pequeña ({ancho}×{alto}). Nuestras fotos de entrenamiento venían del "
                   "celular en alta resolución, y el resultado puede cambiar con fotos reducidas o "
                   "comprimidas (WhatsApp, capturas de pantalla, internet). Si puedes, usa la foto original.")
    izq, der = st.columns([3, 2], gap="large")
    with izq:
        c1, c2 = st.columns([3, 2])
        c1.image(rgb(overlay(cv2.resize(img, pred.seg.mask.shape[::-1]), pred.seg)),
                 caption="Segmentación (contorno verde)", width="stretch")
        c2.image(rgb(pred.crop), caption="Lo que ve el modelo", width="stretch")
    tarjeta_resultado(pred, der)
    st.caption("El sistema es un apoyo: revisa la máscara y las probabilidades antes de decidir.")


def camara_en_vivo(model_name, fruit):
    camara = st.selectbox("Cámara", camaras_disponibles(),
                          format_func=lambda c: "Cámara del computador" if c == 0 else c)
    st.caption("¿No aparece el iPhone? Bloquéalo y déjalo quieto cerca del Mac (o conéctalo por cable), con "
               "la misma cuenta de Apple y Wi-Fi y Bluetooth encendidos. Si lo desbloqueas, macOS lo suelta.")
    if st.button("Buscar cámaras de nuevo"):
        camaras_disponibles.clear()
        st.rerun()
    encendida = st.toggle("Encender la cámara")
    st.caption("Pon una sola fruta frente a la cámara, sobre un fondo liso y con buena luz.")
    if not encendida:
        return
    with st.spinner("Encendiendo la cámara…"):
        cap = abrir(camara)
        abierta = cap.isOpened()
    if not abierta:
        detalle = cap.error() if hasattr(cap, "error") else ""
        cap.release()
        st.error("No se pudo abrir esa cámara. Si es el iPhone, revisa que esté cerca del Mac y desbloquéalo "
                 "una vez. En macOS, da permiso de cámara a la Terminal (o a VS Code) en Ajustes del "
                 "Sistema > Privacidad y seguridad > Cámara." + (f"\n\nDetalle: {detalle}" if detalle else ""))
        return
    izq, der = st.columns([3, 2], gap="large")
    cuadro = izq.empty()
    resultado = der.empty()
    try:
        while True:
            inicio = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                st.error("La cámara dejó de enviar imágenes.")
                break
            t0 = time.perf_counter()
            pred = predict(frame, model_name, fruit)
            ms = (time.perf_counter() - t0) * 1000
            vista = overlay(cv2.resize(frame, pred.seg.mask.shape[::-1]), pred.seg)
            nombre, _ = CLASES[pred.quality]
            texto = f"{nombre} {float(pred.probs[pred.quality]):.0%} | {TAMANOS.get(pred.size, pred.size)}"
            cv2.putText(vista, texto, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(vista, texto, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cuadro.image(rgb(vista), caption=f"En vivo · {ms:.0f} ms por cuadro", width="stretch")
            tarjeta_resultado(pred, resultado)
            # Sin este límite el ciclo clasifica y envía cuadros tan rápido como puede, y traba el
            # computador y el navegador (más aún compartiendo pantalla en una videollamada).
            time.sleep(max(0.0, 1 / CUADROS_POR_SEGUNDO - (time.perf_counter() - inicio)))
    finally:
        cap.release()


st.set_page_config(page_title="Calidad de frutas", page_icon="🍎", layout="wide")
st.title("Clasificador de calidad y tamaño de frutas")
st.caption("Proyecto final · Algoritmos y Programación III · Universidad Icesi")

models = available_models()
if not models:
    st.error("No hay modelos entrenados. Ejecuta `python -m src.main all` primero.")
    st.stop()
orden = [m for m in NOMBRES_MODELO if m in models] + [m for m in models if m not in NOMBRES_MODELO]
f1 = f1_fotos_propias()

with st.sidebar:
    model_name = st.selectbox("Modelo", orden, index=orden.index("svm") if "svm" in orden else 0,
                              format_func=lambda m: NOMBRES_MODELO.get(m, m))
    if model_name in f1:
        st.caption(f"F1 macro en nuestras fotos de prueba: {f1[model_name]:.2f}")
    fruit = FRUTAS[st.selectbox("Fruta (para el umbral de tamaño)", list(FRUTAS))]
    st.markdown("**Para mejores resultados:** una sola fruta, fondo liso y claro, buena luz.")

origen = st.radio("Imagen", ["Subir foto", "Foto de ejemplo", "Tomar foto", "Cámara en vivo"],
                  horizontal=True)

if origen == "Cámara en vivo":
    camara_en_vivo(model_name, fruit)
else:
    data = None
    if origen == "Subir foto":
        archivo = st.file_uploader("Foto de una fruta", type=["jpg", "jpeg", "png", "heic", "heif"])
        data = archivo.getvalue() if archivo else None
    elif origen == "Foto de ejemplo":
        ejemplos = sorted(EJEMPLOS.glob("*.jpg"))
        if ejemplos:
            elegido = st.selectbox("Ejemplo (fotos de prueba, que el modelo no vio al entrenar)", ejemplos,
                                   format_func=lambda p: p.stem.replace("_", " "))
            data = elegido.read_bytes()
        else:
            st.info("No hay fotos de ejemplo en src/app/ejemplos/.")
    else:
        foto = st.camera_input("Toma una foto")
        data = foto.getvalue() if foto else None

    if data is not None:
        try:
            img = decode(data)
        except Exception:
            st.error("No se pudo leer la imagen. Usa JPG, PNG o HEIC.")
            st.stop()
        mostrar(img, model_name, fruit)
