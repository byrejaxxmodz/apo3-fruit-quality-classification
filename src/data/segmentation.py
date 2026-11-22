"""Segmentación fruta / fondo y medición del tamaño.

Supuesto de la consigna: una fruta sobre fondo simple y uniforme. Entonces:

1. Se pasa a HSV. Las frutas son saturadas (S alta) y el fondo es neutro
   (blanco, gris, negro) -> umbral de Otsu sobre S separa ambas clases
   maximizando la varianza entre clases:
       t* = argmax_t  w0(t) w1(t) [mu0(t) - mu1(t)]^2
2. Apertura + cierre morfológico eliminan ruido y rellenan brillos.
3. Se conserva el contorno de mayor área (la fruta).

Tamaño: diámetro equivalente del círculo con la misma área A del contorno,
    d_eq = sqrt(4 A / pi),
normalizado por el lado menor de la imagen para que no dependa de la
resolución:  d_norm = d_eq / min(H, W)  en [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src import config


@dataclass
class Segmentation:
    mask: np.ndarray                 # uint8 {0,255}, mismo tamaño que la imagen
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: float                      # píxeles
    d_eq: float                      # diámetro equivalente en píxeles
    d_norm: float                    # diámetro normalizado [0,1]
    circularity: float               # 4 pi A / P^2   (1 = círculo perfecto)
    solidity: float                  # A / área de la envolvente convexa
    aspect_ratio: float              # w / h del rectángulo mínimo rotado
    ok: bool                         # False si la segmentación es dudosa


def load_image(path, max_side: int = config.MAX_SIDE) -> np.ndarray:
    """Lee una imagen en BGR y reduce su lado mayor a `max_side`."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return resize_max_side(img, max_side)


def resize_max_side(img: np.ndarray, max_side: int = config.MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def foreground_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Máscara binaria del primer plano (todas las regiones candidatas)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    _, m_s = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Respaldo para frutas poco saturadas sobre fondo claro: diferencia con el
    # color del borde (se asume que el borde de la foto es fondo).
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    dist = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m_d = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = cv2.bitwise_or(m_s, m_d)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    return mask


def _measure(contour: np.ndarray, shape) -> Segmentation:
    h_img, w_img = shape[:2]
    mask = np.zeros((h_img, w_img), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    area = float(cv2.contourArea(contour))
    perim = float(cv2.arcLength(contour, True)) or 1.0
    hull_area = float(cv2.contourArea(cv2.convexHull(contour))) or 1.0
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    d_eq = float(np.sqrt(4 * area / np.pi))
    frac = area / (h_img * w_img)
    return Segmentation(
        mask=mask,
        bbox=tuple(int(v) for v in cv2.boundingRect(contour)),
        area=area,
        d_eq=d_eq,
        d_norm=d_eq / min(h_img, w_img),
        circularity=4 * np.pi * area / perim**2,
        solidity=area / hull_area,
        aspect_ratio=max(rw, rh) / max(min(rw, rh), 1.0),
        ok=0.02 < frac < 0.95,   # ni ruido ni "todo es fruta"
    )


def _region_score(contour: np.ndarray, sat: np.ndarray) -> float:
    """Puntaje de «parece fruta» de una región candidata.

    score = A · (S̄/255)² · (1 − b)²
      A  : área de la región
      S̄  : saturación media dentro de la región (las frutas son saturadas;
           paredes, pisos y sombras no)
      b  : fracción del contorno que toca el borde de la imagen (la fruta
           suele estar completa dentro de la foto; el fondo toca los bordes)
    """
    h, w = sat.shape
    m = np.zeros((h, w), np.uint8)
    cv2.drawContours(m, [contour], -1, 255, cv2.FILLED)
    area = float((m > 0).sum())
    if area == 0:
        return 0.0
    s_mean = float(sat[m > 0].mean()) / 255.0
    pts = contour.reshape(-1, 2)
    on_border = ((pts[:, 0] <= 2) | (pts[:, 0] >= w - 3) | (pts[:, 1] <= 2) | (pts[:, 1] >= h - 3)).mean()
    return area * s_mean**2 * (1.0 - float(on_border)) ** 2


def segment(img_bgr: np.ndarray) -> Segmentation:
    """Segmenta la fruta principal.

    1. Máscara candidata (saturación OR diferencia con el color del borde).
    2. Entre las regiones candidatas se elige la de mayor `_region_score`, no
       la de mayor área: con fondos no uniformes (pared + piso) la región más
       grande suele ser el fondo.
    3. Refinamiento: dentro de la región se descartan los píxeles poco
       saturados del contorno exterior (sombra) y se rellenan los huecos
       interiores (brillos y manchas de la propia fruta).
    """
    h, w = img_bgr.shape[:2]
    sat = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1], (5, 5), 0)
    mask = foreground_mask(img_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 0.005 * h * w]
    if not contours:  # sin primer plano: se usa la imagen completa
        full = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]])
        seg = _measure(full, img_bgr.shape)
        seg.ok = False
        return seg
    best = max(contours, key=lambda c: _region_score(c, sat))

    def _touch(c):
        p = c.reshape(-1, 2)
        return ((p[:, 0] <= 2) | (p[:, 0] >= w - 3) | (p[:, 1] <= 2) | (p[:, 1] >= h - 3)).mean()

    region = np.zeros((h, w), np.uint8)
    cv2.drawContours(region, [best], -1, 255, cv2.FILLED)

    # Segunda pasada: si la región toca el borde, la fruta quedó pegada a un
    # parche de fondo. Se separan con un umbral de Otsu sobre la saturación
    # (la fruta es más saturada que pared/piso/papel) y se re-puntúa.
    if _touch(best) > 0.02:
        t_g, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sub = cv2.bitwise_and(region, (sat > t_g).astype(np.uint8) * 255)
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=2)
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        cs2, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cs2 = [c for c in cs2 if cv2.contourArea(c) > 0.005 * h * w]
        if cs2:
            cand = max(cs2, key=lambda c: _region_score(c, sat))
            if _touch(cand) < _touch(best):
                best = cand
                region = np.zeros((h, w), np.uint8)
                cv2.drawContours(region, [best], -1, 255, cv2.FILLED)

    # Refinamiento: quitar sombra (baja saturación) y rellenar huecos
    t_s, _ = cv2.threshold(sat[region > 0].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vals = sat[region > 0]
    refined = region.copy()
    # Solo si la región es claramente bimodal (fruta saturada + sombra gris)
    if (vals > t_s).mean() > 0.4 and vals[vals > t_s].mean() - vals[vals <= t_s].mean() > 40:
        refined = cv2.bitwise_and(region, (sat > 0.6 * t_s).astype(np.uint8) * 255)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
    cs, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs:
        c_ref = max(cs, key=cv2.contourArea)
        if cv2.contourArea(c_ref) > 0.3 * cv2.contourArea(best):
            best = c_ref
    seg = _measure(best, img_bgr.shape)  # _measure rellena el contorno -> sin huecos
    pts = best.reshape(-1, 2)
    touches = ((pts[:, 0] <= 2) | (pts[:, 0] >= w - 3) | (pts[:, 1] <= 2) | (pts[:, 1] >= h - 3)).mean()
    seg.ok = seg.ok and touches < 0.15 and seg.solidity > 0.7
    return seg


def segment_all(img_bgr: np.ndarray, min_frac: float = 0.01) -> List[Segmentation]:
    """Segmenta TODAS las frutas de una imagen (para la carpeta mixed_quality)."""
    mask = foreground_mask(img_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = img_bgr.shape[0] * img_bgr.shape[1]
    return [_measure(c, img_bgr.shape) for c in contours if cv2.contourArea(c) / total >= min_frac]


def crop_square(img_bgr: np.ndarray, seg: Segmentation, size: int = config.IMG_SIZE,
                pad: float = 0.1, mask_bg: bool = True) -> np.ndarray:
    """Recorte cuadrado centrado en la fruta, redimensionado a size x size.

    Si `mask_bg` es True el fondo se pinta de negro, de modo que el modelo solo
    vea la fruta y no aprenda a distinguir clases por el fondo (fuga de datos
    típica cuando cada fuente tiene un fondo distinto).
    """
    x, y, w, h = seg.bbox
    img = img_bgr.copy()
    if mask_bg:
        img[seg.mask == 0] = 0
    side = int(max(w, h) * (1 + 2 * pad))
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = cx - side // 2, cy - side // 2
    H, W = img.shape[:2]
    canvas = np.zeros((side, side, 3), np.uint8)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, W), min(y0 + side, H)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def overlay(img_bgr: np.ndarray, seg: Segmentation, color=(0, 255, 0)) -> np.ndarray:
    """Dibuja el contorno y el círculo equivalente (para la app y el informe)."""
    out = img_bgr.copy()
    contours, _ = cv2.findContours(seg.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    x, y, w, h = seg.bbox
    cv2.circle(out, (x + w // 2, y + h // 2), int(seg.d_eq / 2), (255, 0, 0), 1)
    return out


def segment_path(path) -> Tuple[np.ndarray, Optional[Segmentation]]:
    img = load_image(path)
    return img, segment(img)
