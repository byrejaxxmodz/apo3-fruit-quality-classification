# Datos: fuentes, definiciones y protocolo de anotación

## Fuentes

| Fuente | Ubicación local | Clases disponibles | Referencia |
|--------|-----------------|--------------------|-----------|
| Fruit Quality Classification (Kaggle) | `data/raw/kaggle/` | good, bad, mixed | R. Park, Kaggle, `ryandpark/fruit-quality-classification` |
| Fotos propias del grupo | `data/raw/propias/` | good, regular, bad | Carpeta de Drive `proyectoAPO` |
| Recortes de mixed_quality | `data/interim/mixed_crops/` | se etiquetan a mano | Derivado de Kaggle |

### Inventario de las fotos propias (descarga del 24-nov-2026)

880 archivos (425 HEIC del iPhone convertidos a JPG, 428 JPG/JPEG, 25 PNG y 2 sin extensión). La verificación de procedencia (`src/data/provenance.py`) excluye **30 imágenes generadas con IA** (ver hallazgo 6), así que quedan **850 archivos utilizables**. De ellos, 148 son copias rotadas o volteadas de otras fotos (ver hallazgo 8), así que corresponden a **714 tomas distintas**:

| Fruta | good | regular | bad | Total |
|-------|-----:|--------:|----:|------:|
| apple | 169 | 24 | 3 | 196 |
| banana | 46 | 66 | 40 | 152 |
| granadilla | 0 | 11 | 7 | 18 |
| guava | 31 | 62 | 103 | 196 |
| lime | 70 | 39 | 30 | 139 |
| orange | 86 | 36 | 27 | 149 |
| **Total** | **402** | **238** | **210** | **850** |

Tomas distintas: guava *bad* 24 (de 103 archivos), lime *bad* 7 (de 30) y lime *regular* 5 (de 39); en total 402 *good*, 204 *regular* y 108 *bad*.

La versión anterior (21-nov-2026) tenía 605 fotos, con solo 42 *bad* y 161 *regular*.

Estructura de la carpeta (`Orage_Bad` tiene un error de digitación; `config.FRUIT_ALIASES` la mapea a *orange*):

```
proyectoAPO/
├── Bad_Quality_Fruits/      Apple_Bad, Orage_Bad, banana_bad, guava_bad, lime_bad
│                            + grupoAPO_granadilla_mala_*.jpg (sueltas)
├── Good_Quality_Fruits/     Apple_good, Banana_good, Orange_good, guava_good, lime_good
└── Regular_Quality_Fruits/  apple_regular, banana_regular, guava_regular, lime_regular, orange_regular
                             + grupoAPO_granadilla_regular_*.jpg, grupoAPO_guayaba_regular_*.jpg (sueltas)
```

Las fotos sueltas no tienen subcarpeta de fruta: `parse_fruit` toma la fruta del nombre del archivo (`config.KNOWN_FRUITS`). La **granadilla** solo existe en las fotos propias y no tiene ejemplos *good*.

### Hallazgos de calidad de los datos

1. **Desbalance:** en las fotos propias *bad* pasó del 7 % al 25 %, pero sigue mal repartido: *apple bad* tiene solo 3 fotos reales y la granadilla no tiene *good*.
2. **Posibles etiquetas erróneas:** `Apple_Bad/1.jpg` y `Apple_Bad/10.jpg` parecen manzanas sin defectos visibles.
3. **Fondos no uniformes:** pared más piso de cemento, granito moteado, papel. La consigna pide un fondo simple y uniforme; esto dificulta la segmentación (ver «Segmentación» abajo).
4. **Varias frutas en una foto:** hay al menos un caso en `orange_regular`.
5. **La clase *regular* existe solo en nuestras fotos:** Kaggle trae únicamente *good*, *bad* y *mixed*. Hay riesgo de que el modelo asocie «regular» con el estilo de nuestras fotos. Ver el F1 por fuente en `experiments/results/f1_por_fuente.csv`.
6. **Imágenes generadas con IA en la carpeta compartida:** 27 de las 30 fotos de `Apple_Bad` (todas menos `1.jpg`, `9.jpg` y `10.jpg`) y las 3 de `Orage_Bad` (`n1`–`n3`) tienen metadatos que las declaran sintéticas: 24 con la marca C2PA de ChatGPT/OpenAI y 6 con SynthID (Google), todas con la etiqueta IPTC `trainedAlgorithmicMedia`. Se excluyen del entrenamiento (`config.EXCLUDE_AI_GENERATED`). La consigna pide fotos **reales**; hay que retirarlas del Drive o moverlas a una carpeta aparte claramente marcada como sintética. Ver la fila 10 de `docs/etica_impactos.md`.
7. **La conversión HEIC → JPG duplicaba fotos:** cada vez que se ejecutaba `convert` se creaba otra copia `_heic.jpg` de cada foto del iPhone. Se corrigió (ahora es idempotente) y se agregaron pruebas en `tests/test_convert.py`. Si alguien del grupo ejecutó `convert` más de una vez con la versión anterior, debe borrar los `*_heic.jpg` duplicados.
8. **Copias aumentadas entre las fotos propias:** 148 archivos con sufijo `_aug_N` (79 de `guava_bad`, 30 de `lime_bad` y 39 de `lime_regular`) son rotaciones o volteos de solo 33 fotos originales; de las de lima no está la original. Como la rotación baja la similitud, la detección de casi duplicados no las juntaba con su original: 17 de las 28 *bad* propias de test tenían una copia suya en train, y el F1 en fotos propias estaba inflado (SVM 0.88 con la fuga, 0.82 en las fotos de test sin copia en train). Desde el 25-nov-2026 `near_duplicate_groups` une cada copia al grupo de su original (`original_stem` en `src/data/preprocess.py`) y la fiabilidad del tamaño las excluye. Las copias se pueden seguir usando para entrenar; el aumento de datos de la CNN ya se hace al vuelo, así que no hace falta subir copias al Drive.
9. **Más copias sin agrupar (encontradas en el EDA del 26-nov-2026):** `banana_bad/banana_001_*` (17 archivos) y `banana_002_*` (11) son recortes, rotaciones y cambios de color de solo dos fotos. No llevan el sufijo `_aug_N` y la similitud de miniaturas no las agrupó, así que 4 de las 29 *bad* propias de test tienen copias en train. El SVM falla 3 de esas 4 y XGBoost ninguna, así que no inflan los resultados. Para corregirlo habría que unirlas en `near_duplicate_groups`, volver a partir y reentrenar. Ver `notebooks/01_eda.ipynb`, hallazgo 7.

### Dataset de Kaggle (descargado el 21-nov-2026)

*Fruit Quality Classification* (ryandpark), 3,3 GB, 19.526 imágenes, licencia CC0. Es una copia del dataset **FruitNet** publicado en Mendeley Data por Meshram y Patil (*Data in Brief*, 2022), que es el que hay que citar. Son imágenes de 256×256 de 6 frutas indias tomadas con celular, con fondos e iluminación variados.

| Carpeta | apple | banana | guava | lime | orange | pomegranate |
|---------|------:|-------:|------:|-----:|-------:|------------:|
| Bad Quality_Fruits | 1141 | 1087 | 1129 | 1085 | 1159 | 1187 |
| Good Quality_Fruits | 1149 | 1113 | 1152 | 1094 | 1216 | **5940** |
| Mixed Qualit_Fruits | 113 | 285 | 148 | 278 (*Lemon*) | 125 | 125 |

- `Pomegranate_Good` se submuestrea a 1.200 imágenes (`config.KAGGLE_MAX_PER_FOLDER`) para no desbalancear *good*.
- **Hallazgo ético:** las imágenes de Kaggle conservan EXIF con **coordenadas GPS**, modelo del teléfono y fecha. Por eso las imágenes crudas no se redistribuyen (fila 8 de `docs/etica_impactos.md`).
- Los grupos de casi duplicados también aplican aquí: FruitNet fotografía cada fruta desde 5 direcciones.

**Qué hacer con las fotos (en orden de prioridad):** tomar fotos *bad* reales (sobre todo de manzana, guayaba y lima), más *apple regular*, y repetir las tomas sobre un fondo liso y claro.

### Segmentación con fondos reales

Elegir «la región más grande» fallaba porque la pared o el piso eran más grandes que la fruta. `segment()` ahora elige la región con mayor puntaje A·S̄²·(1−b)², es decir, área por saturación media al cuadrado, penalizando la fracción del contorno que toca el borde de la imagen. Si la región elegida toca el borde, la vuelve a dividir con Otsu sobre la saturación y además recorta la sombra. Al revisar 36 imágenes, unas 31 quedan limpias. Las demás conservan un pedazo del fondo (granito o pared). Mejora futura: GrabCut iniciado con esta máscara, o fotos con un fondo uniforme.

## Definición operativa de las clases de calidad

Todos los anotadores deben usar los mismos criterios (la consigna exige etiquetas coherentes):

| Clase | Criterio visual |
|-------|-----------------|
| **good** | Color uniforme y propio de la madurez comercial. Sin golpes, manchas ni cortes. Forma regular. |
| **regular** | Apta para el consumo, pero con defectos leves: manchas pequeñas (< 10 % de la superficie), un golpe leve, madurez avanzada o color desigual. |
| **bad** | No apta para la venta: podredumbre, moho, golpes grandes, piel rota o manchas en más del 30 % de la superficie. |

## Tamaño

- **Automático:** diámetro equivalente `d_eq = sqrt(4·A/π)` del contorno segmentado, normalizado por el lado menor de la imagen. Se clasifica por terciles por fruta (ver `src/data/preprocess.py`).
- **Manual (recomendado para las fotos propias):** columna `size_manual` de `data/anotaciones_propias.csv`. Para que el tamaño en píxeles signifique algo, **las fotos deben tomarse a una distancia fija** (por ejemplo 30 cm) o con un objeto de referencia en la escena, como una moneda de 500 COP (23,5 mm).

## Protocolo de anotación

1. Ejecutar `python -m src.main index`. Esto crea `data/anotaciones_propias.csv` con una fila por foto propia.
2. Cada integrante revisa las fotos y completa `quality`, `size_manual`, `defectos` y `anotador`.
3. Para medir qué tan consistente es el etiquetado, **dos integrantes anotan por separado un mismo subconjunto de 20 fotos** y se calcula el kappa de Cohen: `κ = (p_o − p_e) / (1 − p_e)`. Un valor κ > 0.6 indica acuerdo sustancial.
4. Se vuelve a ejecutar `index`: las etiquetas manuales reemplazan a las que se dedujeron de las carpetas.
5. El CSV se sube al repositorio y a la carpeta compartida de toda la clase.

## Captura de fotos (checklist)

- [ ] Una sola fruta por foto, centrada
- [ ] Fondo liso y de color neutro (blanco o gris), distinto del color de la fruta
- [ ] Luz difusa, sin flash ni sombras duras
- [ ] Distancia fija a la cámara (anotarla)
- [ ] Varias caras de la misma fruta, cada una como una foto distinta (**ojo**: todas las fotos de una misma fruta deben quedar en el mismo split; ver «trabajo futuro»)
- [ ] Sin personas, rostros ni información identificable en la escena
