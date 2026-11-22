# Datos: fuentes, definiciones y protocolo de anotación

## Fuentes

| Fuente | Ubicación local | Clases disponibles | Referencia |
|--------|-----------------|--------------------|-----------|
| Fruit Quality Classification (Kaggle) | `data/raw/kaggle/` | good, bad, mixed | R. Park, Kaggle, `ryandpark/fruit-quality-classification` |
| Fotos propias del grupo | `data/raw/propias/` | good, regular, bad | Carpeta de Drive `proyectoAPO` |
| Recortes de mixed_quality | `data/interim/mixed_crops/` | se etiquetan a mano | Derivado de Kaggle |

### Inventario de las fotos propias (descarga del 21-nov-2026)

635 archivos (425 HEIC del iPhone, 183 JPG, 27 PNG). Después de convertir los formatos quedan **605 fotos utilizables**:

| Fruta | good | regular | bad | Total |
|-------|-----:|--------:|----:|------:|
| apple | 169 | 5 | 3 | 177 |
| banana | 46 | 66 | 12 | 124 |
| guava | 31 | 54 | 0 | 85 |
| lime | 70 | 0 | 0 | 70 |
| orange | 86 | 36 | 27 | 149 |
| **Total** | **402** | **161** | **42** | **605** |

Estructura de la carpeta (`Orage_Bad` tiene un error de digitación; `config.FRUIT_ALIASES` la mapea a *orange*; `guava_bad/` y `lime_bad/` están vacías):

```
proyectoAPO/
├── Bad_Quality_Fruits/      Apple_Bad, Orage_Bad, banana_bad, guava_bad (vacía), lime_bad (vacía)
├── Good_Quality_Fruits/     Apple_good, Banana_good, Orange_good, guava_good, lime_good
└── Regular_Quality_Fruits/  apple_regular, banana_regular, guava_regular, orange_regular
```

### ⚠️ Hallazgos de calidad de los datos (discutirlos en el informe)

1. **Desbalance fuerte:** *bad* es el 7 % del total y en guava y lime no existe. *apple bad* tiene solo 3 fotos y *apple regular* solo 5.
2. **Posibles etiquetas erróneas:** `Apple_Bad/1.jpg` y `Apple_Bad/10.jpg` parecen manzanas sin defectos visibles.
3. **Fondos no uniformes:** pared más piso de cemento, granito moteado, papel. La consigna pide un fondo simple y uniforme; esto dificulta la segmentación (ver «Segmentación» abajo).
4. **Varias frutas en una foto:** hay al menos un caso en `orange_regular`.
5. **La clase *regular* existe solo en nuestras fotos:** Kaggle trae únicamente *good*, *bad* y *mixed*. Hay riesgo de que el modelo asocie «regular» con el estilo de nuestras fotos. Ver el F1 por fuente en `experiments/results/f1_por_fuente.csv`.

### Dataset de Kaggle (descargado el 21-nov-2026)

*Fruit Quality Classification* (ryandpark), 3,3 GB, 19.526 imágenes, licencia CC0. Es una copia del dataset **FruitNet** publicado en Mendeley Data por Meshram y Patil (*Data in Brief*, 2022), que es el que hay que citar. Son imágenes de 256×256 de 6 frutas indias tomadas con celular, con fondos e iluminación variados.

| Carpeta | apple | banana | guava | lime | orange | pomegranate |
|---------|------:|-------:|------:|-----:|-------:|------------:|
| Bad Quality_Fruits | 1141 | 1087 | 1129 | 1085 | 1159 | 1187 |
| Good Quality_Fruits | 1149 | 1113 | 1152 | 1094 | 1216 | **5940** |
| Mixed Qualit_Fruits | 113 | 285 | 148 | 278 (*Lemon*) | 125 | 125 |

- `Pomegranate_Good` se submuestrea a 1.200 imágenes (`config.KAGGLE_MAX_PER_FOLDER`) para no desbalancear *good*.
- **Hallazgo ético:** las imágenes de Kaggle conservan EXIF con **coordenadas GPS**, modelo del teléfono y fecha. Es un ejemplo real de fuga de privacidad en un dataset público, y vale la pena mencionarlo en la tabla de ética.
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
