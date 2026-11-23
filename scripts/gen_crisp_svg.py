"""Genera docs/figuras/crisp_dm.svg a partir del flujo CRISP-DM del proyecto.

SVG vectorial, sin dependencias (no usa mermaid-cli/Chromium). La fuente
canónica del diagrama sigue siendo el bloque mermaid de docs/crisp_dm.md;
este script produce el export para el informe.
"""
from __future__ import annotations

import html

W = 900
PHASE = {  # color por fase (relleno de encabezado, relleno de chip, borde)
    "N": ("#1d4ed8", "#eff6ff", "#bfdbfe"),
    "D": ("#0f766e", "#f0fdfa", "#99f6e4"),
    "P": ("#15803d", "#f0fdf4", "#bbf7d0"),
    "M": ("#b45309", "#fffbeb", "#fde68a"),
    "E": ("#7e22ce", "#faf5ff", "#e9d5ff"),
    "S": ("#be123c", "#fff1f2", "#fecdd3"),
}

out: list[str] = []


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def chip(cx, cy, w, lines, phase, h=52, bold0=True, sub=False):
    """Caja redondeada con texto centrado (primera línea en negrita)."""
    _, fill, border = PHASE[phase]
    x, y = cx - w / 2, cy - h / 2
    out.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="9" '
               f'fill="{fill}" stroke="{border}" stroke-width="1.5"/>')
    n = len(lines)
    lh = 15
    y0 = cy - (n - 1) * lh / 2 + 4
    for i, ln in enumerate(lines):
        weight = "700" if (i == 0 and bold0) else "400"
        size = 12 if (i == 0 and bold0) else 10.5
        col = "#0f172a" if (i == 0 and bold0) else "#475569"
        out.append(
            f'<text x="{cx:.0f}" y="{y0 + i * lh:.0f}" text-anchor="middle" '
            f'font-size="{size}" font-weight="{weight}" fill="{col}">{esc(ln)}</text>')


def diamond(cx, cy, w, h, lines, phase):
    _, fill, border = PHASE[phase]
    pts = f"{cx},{cy-h/2} {cx+w/2},{cy} {cx},{cy+h/2} {cx-w/2},{cy}"
    out.append(f'<polygon points="{pts}" fill="{fill}" stroke="{border}" stroke-width="1.5"/>')
    lh = 14
    y0 = cy - (len(lines) - 1) * lh / 2 + 4
    for i, ln in enumerate(lines):
        out.append(f'<text x="{cx:.0f}" y="{y0 + i*lh:.0f}" text-anchor="middle" '
                   f'font-size="11" font-weight="700" fill="#0f172a">{esc(ln)}</text>')


def phase_label(x, y, num, title, phase):
    hdr, _, _ = PHASE[phase]
    out.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="26" height="26" rx="6" fill="{hdr}"/>')
    out.append(f'<text x="{x+13:.0f}" y="{y+18:.0f}" text-anchor="middle" font-size="14" '
               f'font-weight="800" fill="#ffffff">{num}</text>')
    out.append(f'<text x="{x+34:.0f}" y="{y+18:.0f}" font-size="14.5" font-weight="800" '
               f'fill="{hdr}">{esc(title)}</text>')


def arrow(x1, y1, x2, y2, label="", dashed=False, color="#64748b", curve=None):
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    if curve is None:
        out.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                   f'stroke="{color}" stroke-width="2"{dash} marker-end="url(#ah)"/>')
    else:
        out.append(f'<path d="{curve}" fill="none" stroke="{color}" stroke-width="2"{dash} '
                   f'marker-end="url(#ah)"/>')
    if label:
        lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
        tw = 7 * len(label) + 10
        out.append(f'<rect x="{lx - tw/2:.0f}" y="{ly - 10:.0f}" width="{tw}" height="18" '
                   f'rx="4" fill="#ffffff" stroke="{color}" stroke-width="1"/>')
        out.append(f'<text x="{lx:.0f}" y="{ly + 3:.0f}" text-anchor="middle" font-size="10.5" '
                   f'font-weight="700" fill="{color}">{esc(label)}</text>')


# ---- layout ----
H = 1500
cx = 430  # eje central

# fases: (y_label, numero, titulo)
phase_label(40, 66, "1", "Comprensión del negocio", "N")
chip(cx, 128, 420, ["Clasificación manual lenta y subjetiva", "en plazas de mercado"], "N")
chip(cx, 196, 420, ["Objetivo: calidad (good/regular/bad)", "+ tamaño (pequeño/mediano/grande)"], "N")
chip(cx, 264, 420, ["Métrica de éxito: F1 macro ≥ 0.80", "en fotos propias de test"], "N")
arrow(cx, 154, cx, 170)
arrow(cx, 222, cx, 238)

phase_label(40, 312, "2", "Comprensión de los datos", "D")
chip(250, 372, 320, ["Kaggle: Fruit Quality", "(good / bad / mixed)"], "D")
chip(610, 372, 320, ["Fotos propias en Drive", "(good / regular)"], "D")
chip(cx, 452, 420, ["EDA: clases por fruta y fuente, fondo,", "desbalance · notebooks/01_eda.ipynb"], "D")
arrow(cx, 290, 250, 346)   # N3 -> D1
arrow(cx, 290, 610, 346)   # N3 -> D2
arrow(250, 398, cx - 40, 426)
arrow(610, 398, cx + 40, 426)

phase_label(40, 500, "3", "Preparación de los datos", "P")
chip(cx, 560, 440, ["mixed_quality → recortes individuales", "src/data/segment_mixed.py"], "P")
chip(cx, 626, 440, ["Índice + anotación manual", "src/data/dataset.py"], "P")
chip(cx, 692, 440, ["Segmentación HSV + Otsu, recorte", "128×128 fondo negro · preprocess.py"], "P")
chip(cx, 758, 440, ["Tamaño: d_eq = √(4A/π) normalizado,", "terciles por fruta"], "P")
chip(cx, 824, 440, ["Split estratificado 70/15/15", "por calidad y fuente (sin fuga)"], "P")
arrow(cx, 478, cx, 534)    # D3 -> P0
for y in (586, 652, 718, 784):
    arrow(cx, y, cx, y + 14)

phase_label(40, 872, "4", "Modelado", "M")
chip(250, 936, 320, ["62 características manuales", "color·textura·defectos·forma"], "M")
chip(250, 1004, 320, ["SVM · RF · LogReg · KNN · MLP", "GridSearchCV k=5, F1 macro"], "M")
chip(610, 970, 320, ["CNN pequeña desde cero", "3 conv+GAP · búsqueda de hiperpar."], "M", h=58)
arrow(cx, 850, 250, 910)   # P4 -> M1
arrow(cx, 850, 610, 941)   # P4 -> CNN
arrow(250, 962, 250, 978)  # M1 -> M2

phase_label(40, 1058, "5", "Evaluación", "E")
chip(cx, 1118, 440, ["Test: accuracy, P, R, F1 macro", "vs. línea base mayoritaria"], "E")
chip(cx, 1184, 440, ["Matrices de confusión, curvas,", "F1 Kaggle vs. fotos propias · tamaño"], "E")
diamond(cx, 1266, 230, 76, ["¿Cumple la métrica", "en fotos propias?"], "E")
arrow(250, 1030, cx - 70, 1092)   # M2 -> E1
arrow(610, 999, cx + 70, 1092)    # CNN -> E1
arrow(cx, 1144, cx, 1158)
arrow(cx, 1210, cx, 1228)

phase_label(40, 1360, "6", "Despliegue", "S")
chip(cx, 1420, 440, ["App Streamlit: subir foto o cámara", "src/app/streamlit_app.py"], "S", h=56)
arrow(cx, 1304, cx, 1392, label="Sí", color="#15803d")

# realimentación: No -> volver a Preparación (índice/anotación)
fb = f"M {cx+115} 1266 H 840 V 626 H {cx+225}"  # a la derecha, sube, entra a P1
out.append(f'<path d="{fb}" fill="none" stroke="#dc2626" stroke-width="2" '
           f'stroke-dasharray="6 5" marker-end="url(#ah)"/>')
out.append('<rect x="786" y="930" width="108" height="20" rx="4" fill="#ffffff" '
           'stroke="#dc2626" stroke-width="1"/>')
out.append('<text x="840" y="944" text-anchor="middle" font-size="10.5" font-weight="700" '
           'fill="#dc2626">No: más datos</text>')

svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'font-family="Segoe UI, Helvetica, Arial, sans-serif">',
    '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
    'markerHeight="7" orient="auto-start-reverse">'
    '<path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>',
    f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
    '<text x="40" y="38" font-size="19" font-weight="800" fill="#0f172a">'
    'CRISP-DM · Clasificación de calidad de fruta por visión</text>',
    *out,
    '</svg>',
]
import pathlib
# scripts/gen_crisp_svg.py -> raíz del repo -> docs/figuras/crisp_dm.svg
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "docs" / "figuras" / "crisp_dm.svg"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text("\n".join(svg), encoding="utf-8")
print("wrote", p, len("\n".join(svg)), "bytes")
