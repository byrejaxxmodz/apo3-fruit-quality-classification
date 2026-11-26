"""Lectura de cámaras para el modo «Cámara en vivo».

En macOS reciente, OpenCV no ve el iPhone como cámara (Cámara de Continuidad):
la API que usa no lista ese tipo de dispositivo. ffmpeg sí lo ve, así que, si
ffmpeg está instalado, las cámaras se listan y se leen con él; si no, se usa
OpenCV con la cámara del computador.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading

import cv2
import numpy as np

FFMPEG = shutil.which("ffmpeg")


def listar_camaras() -> list[str]:
    """Nombres de las cámaras de video que ve ffmpeg (sin pantallas ni la vista de escritorio)."""
    if not FFMPEG or sys.platform != "darwin":
        return []
    salida = subprocess.run([FFMPEG, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                            capture_output=True, text=True, timeout=15).stderr
    video = salida.split("audio devices")[0]
    nombres = re.findall(r"\] \[\d+\] (.+)$", video, flags=re.M)
    return [n.strip() for n in nombres if "Capture screen" not in n and "Desk View" not in n]


class CamaraFfmpeg:
    """Lee una cámara con ffmpeg (MJPEG por una tubería) y guarda solo el cuadro más reciente.

    Un hilo lee la tubería sin parar, así el video no se atrasa aunque la
    clasificación vaya más lenta que la cámara.
    """

    def __init__(self, nombre: str, fps: int = 10, ancho: int = 1280):
        self.proceso = subprocess.Popen(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "avfoundation", "-framerate", "30",
             "-i", f"{nombre}:none", "-vf", f"fps={fps},scale={ancho}:-2",
             "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        self.ultimo = None
        self.nuevo = threading.Event()
        self.vivo = True
        threading.Thread(target=self._leer, daemon=True).start()

    def _leer(self):
        buffer = b""
        while self.vivo:
            datos = self.proceso.stdout.read1(65536)
            if not datos:
                break
            buffer += datos
            # Cada JPEG va de FFD8 a FFD9; se decodifica solo el último completo.
            fin = buffer.rfind(b"\xff\xd9")
            if fin == -1:
                continue
            inicio = buffer.rfind(b"\xff\xd8", 0, fin)
            if inicio != -1:
                img = cv2.imdecode(np.frombuffer(buffer[inicio:fin + 2], np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    self.ultimo = img
                    self.nuevo.set()
            buffer = buffer[fin + 2:]
        self.vivo = False
        self.nuevo.set()

    def isOpened(self) -> bool:
        """Espera el primer cuadro (la cámara del iPhone tarda un par de segundos en encender)."""
        self.nuevo.wait(timeout=10)
        return self.ultimo is not None

    def read(self):
        self.nuevo.wait(timeout=5)
        self.nuevo.clear()
        if self.ultimo is None or not self.vivo:
            return False, None
        return True, self.ultimo.copy()

    def error(self) -> str:
        if self.proceso.poll() is None:
            return ""
        return self.proceso.stderr.read().decode(errors="ignore").strip()[-300:]

    def release(self):
        self.vivo = False
        if self.proceso.poll() is None:
            self.proceso.terminate()
            try:
                self.proceso.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proceso.kill()


def abrir(camara: str | int):
    """`camara` es un nombre de ffmpeg o un índice de OpenCV."""
    if isinstance(camara, str):
        return CamaraFfmpeg(camara)
    cap = cv2.VideoCapture(camara)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap
