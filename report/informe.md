# [Título del proyecto]

**Integrantes:** Nombre 1, Nombre 2, Nombre 3 — Algoritmos y Programación III, Universidad Icesi, 2026-2

> Plantilla con la estructura obligatoria de la consigna (**máximo 7 páginas**). Al final se pasa a LaTeX/IEEE o a Word. Las figuras deben ser vectoriales (SVG o PDF): las que genera `evaluate.py` ya salen en SVG en `docs/figuras/`.

## Resumen
_(150–200 palabras: problema, datos, modelos, mejor resultado en números, conclusión principal.)_

## 1. Introducción
- Contexto: pérdidas poscosecha y clasificación manual en plazas de mercado de Colombia (citar cifras de FAO/DANE).
- Problema y por qué es relevante.
- Objetivo y aportes: dataset propio + comparación de ML clásico frente a CNN + app.

## 2. Fundamentos teóricos
_(Solo lo que el lector necesita: Otsu, características de color y textura, SVM/RF, CNN, F1 macro. Resumir `docs/formulacion_matematica.md`.)_

## 3. Metodología
- Diagrama propio de CRISP-DM (`docs/crisp_dm.md` exportado a SVG).
- **Formulación matemática del problema** (PI3).
- Datos: fuentes, tabla de conteos por clase, fruta y fuente; protocolo de anotación; kappa.
- Preparación: segmentación, tamaño, split.
- Modelado: rejillas de hiperparámetros, validación cruzada, arquitectura de la CNN.
- Alternativas descartadas.

## 4. Resultados

> **Resultados con Kaggle + fotos propias (21-nov-2026):** 14.301 imágenes: 13.696 de Kaggle (Pomegranate_Good recortada a 1.200) y 605 propias. Partición agrupada por casi duplicados: 10.148 en train, 2.106 en val y 2.047 en test (106 de ellas propias).

| Modelo | F1 macro CV | F1 macro test (global) | F1 test Kaggle | F1 test propias | Recall *bad* propias | Recall *regular* propias |
|--------|-----------:|----------------------:|---------------:|----------------:|---------------------:|-------------------------:|
| **SVM RBF** (C=10) | **0.91** | **0.97** | **0.98** | 0.81 | 0.60 | 0.93 |
| KNN (k=3) | 0.88 | 0.93 | 0.97 | **0.84** | **0.80** | 0.82 |
| Random Forest | 0.85 | 0.93 | 0.97 | 0.71 | 0.40 | 0.75 |
| MLP | 0.86 | 0.90 | 0.96 | 0.71 | 0.40 | 0.71 |
| CNN desde cero (ajustada) | 0.73 (val) | 0.81 | 0.89 | 0.48 | 0.00 | 0.93 |
| Regresión logística | 0.70 | 0.78 | 0.89 | 0.44 | 0.00 | 0.98 |
| Línea base | — | 0.22 | 0.33 | 0.23 | 0.00 | 0.00 |

Hallazgos principales:
- **Brecha de dominio:** todos los modelos rinden entre 0.13 y 0.45 puntos menos con nuestras fotos que con las de Kaggle. El resultado «global» de 0.97 está dominado por Kaggle (95 % del test): **la métrica que importa para el negocio es la de las fotos propias**.
- **Atajo (*shortcut learning*) visible:** la regresión logística tiene recall de *regular* = 0.98 pero de *good* = 0.40 en nuestras fotos. Aprendió «foto propia ⇒ regular», porque *regular* solo existe en nuestras fotos. Es la evidencia empírica del riesgo descrito en `docs/datos.md`.
- **Kaggle sí ayudó:** el F1 en fotos propias del SVM subió de 0.75 (solo fotos propias) a 0.81, y el del KNN de 0.78 a 0.84.
- ***bad* en fotos propias sigue débil:** solo hay 5 en test, así que un acierto más o menos mueve el recall 0.20. Necesitamos más fotos *bad* reales.

<details><summary>Resultados anteriores: solo fotos propias (605)</summary>

Partición agrupada: 415 en train, 96 en val y 94 en test.

| Modelo | F1 macro CV (train+val) | F1 macro train | Accuracy test | F1 macro test |
|--------|------------------------:|---------------:|--------------:|--------------:|
| KNN (k=3) | 0.81 | 0.95 | 0.85 | **0.78** |
| SVM RBF (C=100, γ=0.01) | **0.88** | 1.00 | **0.87** | 0.75 |
| Random Forest (500 árboles) | 0.80 | 1.00 | 0.86 | 0.74 |
| CNN desde cero (102k parámetros) | — | — | 0.84 | 0.72 |
| Regresión logística | 0.77 | 0.94 | 0.77 | 0.64 |
| MLP (128, 64) | 0.68 | 0.80 | 0.82 | 0.64 |
| Línea base (clase mayoritaria) | — | — | 0.63 | 0.26 |

Observaciones para el análisis:
- **Fuga de datos detectada y corregida.** Con una partición aleatoria, el SVM daba F1 = 0.97 en test, pero 53 de las 91 imágenes de test tenían un «gemelo» casi idéntico en train (fotos en ráfaga de la misma fruta). Con `StratifiedGroupKFold` el F1 bajó a 0.75. **Esa diferencia es en sí misma un resultado que vale la pena reportar.**
- **Sobreajuste:** el SVM y el RF llegan a F1 de 1.00 en train frente a 0.75 en test.
- **La clase *bad* es el cuello de botella:** solo hay 5 ejemplos en test y 1 en val. Por eso el F1 de val de la CNN (0.96) no es confiable, y el recall de *bad* del SVM es 0.40. Hacen falta más fotos *bad* reales o los datos de Kaggle.
- *good* y *regular* se distinguen bien: F1 de 0.91 y 0.85 con el SVM.

</details>

### 4.1 Estimación de tamaño

El tamaño se estima por el diámetro equivalente normalizado `d_norm` y se
clasifica por terciles por fruta. Como las etiquetas de tamaño se derivan de
`d_norm` y el grupo aún no anotó `size_manual`, **no hay una exactitud
supervisada que reportar** (sería circular). En cambio se mide la **fiabilidad
test–retest**: los grupos de casi duplicados son la misma fruta física desde
otro ángulo, así que deberían recibir el mismo tamaño (`evaluate_size.py`).

| Fuente | Grupos (≥2 fotos) | Acuerdo observado | Acuerdo por azar | κ ajustado | CV intra-grupo de d_norm |
|--------|------------------:|------------------:|-----------------:|-----------:|-------------------------:|
| Fotos propias | 85 | 0.69 | 0.36 | **0.52** | 0.021 |
| Kaggle | 1208 | 0.82 | 0.36 | **0.72** | 0.008 |
| Todas | 1293 | 0.82 | 0.36 | **0.72** | 0.009 |

- Cobertura de la segmentación (único caso en que el tamaño es válido): **87.9 %** (12 567/14 301).
- El diámetro en píxeles es **físicamente estable** (CV intra-grupo ≈ 0.01–0.02: la misma fruta mide casi igual entre tomas), pero la **clasificación por terciles es frágil en las fronteras**, sobre todo en las fotos propias (κ = 0.52 vs. 0.72 en Kaggle): un cambio pequeño de d_norm cruza el umbral del tercil.
- Sin objeto de referencia ni distancia fija, el tamaño en píxeles **solo es comparable dentro de la misma fruta y fuente**, no en centímetros reales (ver trabajo futuro).
- Figura: `docs/figuras/tamano_d_norm.svg` (distribución de d_norm por fuente). Datos: `tamano_distribucion.csv`, `tamano_fiabilidad.csv`.

### 4.2 Ajuste de hiperparámetros de la CNN

Se exploró una rejilla de 8 configuraciones (`src/training/search_cnn.py`),
entrenando cada una con el mismo presupuesto (35 épocas, parada temprana con
paciencia 8) y comparando por **F1 macro en validación**. Fijos: batch 32,
weight decay 1e-4, Adam + recocido coseno, entropía cruzada ponderada.

| lr | dropout | widths | params | F1 val | Mejor época |
|---|---|---|---:|---:|---:|
| **3e-4** | **0.5** | **(32,64,128)** | **101 923** | **0.734** | 25 |
| 1e-3 | 0.5 | (32,64,128) | 101 923 | 0.723 | 16 |
| 3e-4 | 0.3 | (32,64,128) | 101 923 | 0.704 | 29 |
| 1e-3 | 0.3 | (32,64,128) | 101 923 | 0.690 | 16 |
| 1e-3 | 0.5 | (16,32,64) | 28 051 | 0.633 | 2 |
| 1e-3 | 0.3 | (16,32,64) | 28 051 | 0.623 | 2 |
| 3e-4 | 0.5 | (16,32,64) | 28 051 | 0.623 | 2 |
| 3e-4 | 0.3 | (16,32,64) | 28 051 | 0.598 | 2 |

Hallazgos del ajuste:
- **La capacidad del extractor es lo que más pesa:** las cuatro redes anchas (≈102 k parámetros) superan a las angostas (28 k) por 0.07–0.14 de F1. Las angostas **subajustan**: su mejor época es la 2 y no mejoran después (subentrenan la textura/color que separa *good* de *regular*).
- **Dropout 0.5 > 0.3** en todos los pares comparables: con un dataset pequeño, más regularización ayuda.
- **Tasa de aprendizaje baja (3e-4) algo mejor** con dropout alto, a costa de más épocas.
- **El ajuste sube la validación pero no el test de fotos propias:** el F1 de validación pasó de 0.69 a 0.73, pero el F1 en test de nuestras fotos quedó en 0.48 (recall *bad* = 0.00). La validación está dominada por Kaggle y en propias hay solo ~5 ejemplos *bad* en test, así que la ganancia no se traslada a nuestro dominio. **Es otra evidencia de que el cuello de botella no son los hiperparámetros sino la falta de fotos *bad* propias**, no un problema de arquitectura.
- Datos: `experiments/results/cnn_busqueda.csv` y `cnn_config.json`.

Completar con:
- Tabla `experiments/results/comparacion_modelos.csv` (incluye la línea base).
- Tabla de F1 por fuente (`f1_por_fuente.csv`).
- Matrices de confusión y curvas de la CNN.

## 5. Análisis de resultados
- ¿Generalizan? Brecha train–val (sobreajuste).
- ¿Qué confunde el modelo? (Probablemente regular ↔ good.) Ejemplos de errores.
- ¿Funciona igual con nuestras fotos que con las de Kaggle?
- Comparación con la literatura (citar al menos 2 trabajos que hayan usado este u otro dataset similar).

## 6. Ética e impactos
- Tabla de dilemas éticos (PI1) y matriz de impactos (PI2): resumen de `docs/etica_impactos.md`.

## 7. Conclusiones y trabajo futuro
- Volver a conectar con la formulación inicial: ¿qué se resolvió y qué no?
- Mejoras: más fotos propias por clase, split agrupado por fruta física, objeto de referencia para medir el tamaño en cm, Raspberry Pi.

## Referencias (IEEE)
[1] R. Park, "Fruit Quality Classification," Kaggle, 2024. [Online]. Available: https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
[1b] V. Meshram and K. Patil, "FruitNet: Indian fruits image dataset with quality for machine learning applications," *Data in Brief*, vol. 40, p. 107686, 2022. _(Verificar volumen y página en ScienceDirect antes de entregar.)_
[2] N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Trans. Syst., Man, Cybern.*, vol. 9, no. 1, pp. 62–66, 1979.
[3] T. Ojala, M. Pietikäinen, and T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 24, no. 7, pp. 971–987, 2002.
[4] R. M. Haralick, K. Shanmugam, and I. Dinstein, "Textural features for image classification," *IEEE Trans. Syst., Man, Cybern.*, vol. SMC-3, no. 6, pp. 610–621, 1973.
[5] C. Cortes and V. Vapnik, "Support-vector networks," *Mach. Learn.*, vol. 20, pp. 273–297, 1995.
[6] L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, pp. 5–32, 2001.
[7] P. Chapman et al., "CRISP-DM 1.0: Step-by-step data mining guide," SPSS, 2000.
[8] ACM/IEEE-CS, "Software Engineering Code of Ethics and Professional Practice," 1999.
_(Agregar solo lo que de verdad lean y usen. La consigna pide revisar artículos de NeurIPS/ICML/ICLR.)_
