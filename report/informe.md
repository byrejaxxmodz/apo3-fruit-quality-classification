# Clasificación de calidad y tamaño de frutas con visión por computador: de un dataset público a fotos propias

Jaime Muñoz y Andres Felipe Lopez. Algoritmos y Programación III, Universidad Icesi, 2026-2.

Video de presentación: https://www.youtube.com/watch?v=ujYcXWM8miQ

El informe final está en [`report/ieee/informe.tex`](ieee/informe.tex), con la plantilla oficial de conferencia de IEEE. Para compilarlo, sube a Overleaf `informe.tex` y la carpeta `figuras/`.

## Resumen

En las plazas de mercado la fruta se separa por calidad a ojo, y el resultado depende de quién lo haga. Construimos un sistema que, a partir de la foto de una fruta sobre un fondo uniforme, predice su calidad (good, regular o bad) y una clase de tamaño. Usamos 13 696 imágenes de un dataset público y 850 archivos de fotos propias, que corresponden a 714 tomas distintas. Comparamos seis clasificadores sobre 62 características de color, textura y forma con una CNN pequeña entrenada desde cero, con una partición que deja todas las fotos de una misma fruta en el mismo conjunto.

## Resultados principales

F1 macro en prueba. Fotos propias: 111 imágenes de 62 frutas físicas.

| Modelo | CV | Kaggle | Fotos propias [IC 95 %] | Recall regular | Recall bad |
|--------|---:|-------:|------------------------:|---------------:|-----------:|
| KNN (k = 3) | 0.90 | 0.96 | 0.86 [0.75, 0.94] | 0.91 | 0.76 |
| SVM RBF (C = 10) | 0.92 | 0.97 | 0.86 [0.74, 0.95] | 0.87 | 0.72 |
| MLP (128, 64) | 0.89 | 0.96 | 0.85 [0.73, 0.93] | 0.78 | 0.83 |
| XGBoost | 0.92 | 0.97 | 0.82 [0.71, 0.91] | 0.57 | 0.93 |
| Random Forest | 0.86 | 0.94 | 0.69 [0.55, 0.79] | 0.30 | 0.83 |
| Regresión logística | 0.72 | 0.89 | 0.60 [0.42, 0.72] | 0.87 | 0.59 |
| CNN desde cero | 0.74 | 0.88 | 0.59 [0.46, 0.70] | 0.43 | 0.62 |
| Clase mayoritaria | – | 0.32 | 0.23 [0.18, 0.27] | 0.00 | 0.00 |

En nuestras fotos, KNN, SVM, MLP y XGBoost no se distinguen con la prueba de McNemar corregida por Holm. Las ráfagas de fotos casi idénticas y las copias rotadas de una misma foto inflaban el F1 en fotos propias; los resultados de arriba ya usan la partición corregida. Figuras: [intervalos por fuente](../docs/figuras/ic_por_fuente.svg) y [proyección LDA](../docs/figuras/lda_caracteristicas.svg).
