# Ética e impactos

Código de referencia: *Software Engineering Code of Ethics and Professional Practice* (ACM/IEEE-CS) y el código de ética de COPNIA (Ley 842 de 2003, Colombia).

## 1. Dilemas éticos identificados

| # | Situación o decisión del proyecto | Riesgo ético | Principio del código | Medida adoptada |
|---|-----------------------------------|--------------|----------------------|-----------------|
| 1 | Fotos propias tomadas con nuestros celulares | Los metadatos EXIF guardan la ubicación GPS de cada toma y el modelo del teléfono | ACM/IEEE 3.12 (privacidad); Ley 1581 de 2012 | `src/data/convert.py` quita el EXIF al pasar de HEIC a JPG; los recortes se guardan como PNG sin EXIF; las fotos crudas no se suben al repositorio |
| 2 | Compartir el dataset con toda la clase | Uso de los datos para otros fines, sin trazabilidad | ACM/IEEE 3.13 (datos confidenciales), 1.06 (honestidad) | Licencia y README del dataset; registro de quién aportó cada foto (`anotador`) |
| 3 | Usar el dataset de Kaggle | Plagio o violación de la licencia | ACM/IEEE 2.02, 3.03; la consigna considera fraude no citar | Citar la fuente en IEEE en el informe y en `docs/datos.md`; respetar su licencia |
| 4 | Sesgo en las clases (regular solo en fotos propias, bad solo en Kaggle) | Modelo que «acierta» por el fondo o la cámara: falsa sensación de exactitud | ACM/IEEE 3.10 (pruebas adecuadas), 1.06 (no exagerar capacidades) | Fondo enmascarado; métrica por fuente; completar las clases con fotos propias; reportar esta limitación con honestidad |
| 5 | Sesgo por variedad, región o iluminación | El sistema castiga productos locales que no se parecen a los de Kaggle | Equidad y no discriminación | Incluir variedades locales (banano criollo, naranja valencia); análisis de errores por fruta |
| 6 | Automatizar una decisión que afecta los ingresos de un productor | Un falso «bad» hace que se descarte o se pague menos por un producto bueno | ACM/IEEE 1.03, 1.04 (avisar riesgos) | El sistema **recomienda** y una persona decide; se muestra la probabilidad y la segmentación; umbral conservador para «bad» |
| 7 | Etiquetado subjetivo | Etiquetas inconsistentes que pasan a ser «verdad» del modelo | ACM/IEEE 6.07 (exactitud) | Definiciones operativas escritas. La herramienta de doble anotación con kappa de Cohen está lista (`src/evaluation/kappa.py`), pero no alcanzamos a medir el acuerdo; el informe lo reporta como limitación |
| 8 | Usar un dataset público (Kaggle/FruitNet) cuyas imágenes **conservan coordenadas GPS** en el EXIF | Redistribuir la ubicación de los autores o de los lugares de captura | ACM/IEEE 3.12 y 3.13 (privacidad y datos); Ley 1581 de 2012 | No se redistribuyen las imágenes crudas (están en `.gitignore`); los recortes procesados se guardan como PNG sin EXIF; se reporta el hallazgo |
| 9 | Presentar los resultados | Reportar solo la mejor corrida o el test usado para ajustar | ACM/IEEE 1.06, 3.10 | El test se usa una sola vez; se reportan la media y la desviación de la CV, la línea base y los intervalos de confianza |
| 10 | Completar la clase *bad* con imágenes generadas por IA (ChatGPT, Gemini) dentro de la carpeta de fotos «propias» | Presentar datos sintéticos como recolección real (honestidad académica); el modelo aprendería el estilo del generador y no defectos reales; métricas infladas | ACM/IEEE 1.06 (honestidad), 3.10 (pruebas adecuadas), 6.07 (exactitud); la consigna exige fotos reales | `src/data/provenance.py` busca marcas C2PA, IPTC `trainedAlgorithmicMedia` y SynthID, y el índice excluye esas imágenes: **30 detectadas y excluidas** en la descarga del 24-nov-2026. Límite: si se borran los metadatos la marca desaparece, así que se complementa con revisión visual |

## 2. Matriz de impactos

### 2.1 Análisis final, con los resultados (matriz de impactos del informe)

Cada riesgo se apoya en una cifra medida en el proyecto. Las cifras de errores son del conjunto de prueba de fotos propias: 111 fotos, de ellas 59 good y 29 bad.

| Dimensión | Beneficio | Riesgo y evidencia | Mitigación |
|-----------|-----------|--------------------|------------|
| **Social** | El mismo criterio en todos los puestos; la app muestra la máscara y las probabilidades | Sensación de vigilancia si la cámara queda fija en un puesto. Seguridad del comprador: el sistema deja pasar como *good* 5 de 29 *bad* propias (SVM y KNN), y una fruta *bad* puede tener moho o podredumbre | La cámara encuadra solo la fruta y el fondo se borra antes de clasificar. La app no guarda imágenes. El sistema recomienda y una persona decide |
| **Económica** | Cuesta lo que un celular o una cámara web, no una clasificadora industrial | Un falso *good* genera reclamos del comprador. Un falso *bad* hace vender barata fruta buena: el SVM no comete ninguno en 59 *good* propias, el KNN comete 2 | Elegir la clase de menor costo esperado, $\hat y=\arg\min_k \sum_j C_{kj}P(j\mid\mathbf{x})$, con los costos que fije el vendedor. Si dejar pasar una *bad* cuesta más que rebajar una buena, el umbral para predecir *bad* baja y su recall sube |
| **Ambiental** | La fruta *regular* se vende con descuento o para jugo en vez de botarse | Energía de entrenar y de clasificar en tiempo real | Modelos en la CPU de un portátil (75 ms por foto, unos 2 J si el portátil consume unos 30 W); en vivo solo 5 cuadros por segundo (`CUADROS_POR_SEGUNDO` en la app); la búsqueda de la CNN tardó 1.5 h, unos 0.05 kWh |
| **Global** | El método sirve para otras frutas y otros mercados | Las variedades y las normas de calidad cambian por país: el F1 del SVM baja de 0.97 en fruta de India (Kaggle) a 0.86 en la nuestra. Leyes de datos distintas (Ley 1581 de 2012, GDPR) | Fotografiar el mercado de destino y medir el F1 por fuente antes de usar el sistema allí. Recortes sin metadatos |

### 2.2 Análisis inicial, antes de tener resultados

| Dimensión | Impacto positivo | Riesgo o impacto negativo | Mitigación |
|-----------|------------------|---------------------------|------------|
| **Social** | Criterios de calidad uniformes; menos discusiones entre comprador y vendedor; herramienta accesible (celular) | Desplazamiento de trabajadores de clasificación; percepción de vigilancia si hay cámaras fijas en los puestos; desconfianza hacia la «caja negra» | Plantearlo como apoyo, no como reemplazo; la cámara apunta solo a la banda o la mesa y no graba personas; procesamiento local sin guardar imágenes; mostrar la segmentación para hacerlo explicable |
| **Económica** | Menos pérdidas por mezcla de calidades; precios diferenciados por calidad; costo bajo (webcam o Raspberry Pi de unos 300 000 COP frente a equipos industriales) | Errores que causan pérdidas (falsos «bad»); costo de mantenimiento y reentrenamiento; brecha para pequeños productores sin acceso a tecnología | Analizar el costo de cada tipo de error en la matriz de confusión; código abierto; app que funciona en cualquier celular |
| **Ambiental** | Menos desperdicio de alimentos: el producto *regular* se redirige (jugos, descuentos) en vez de botarse | Consumo energético del entrenamiento y la inferencia; residuos electrónicos del hardware | CNN pequeña (unos 100 mil parámetros) que corre en CPU o Raspberry Pi; entrenamiento corto con parada temprana; reportar el tiempo de entrenamiento y el consumo estimado |
| **Global** | Replicable en otros mercados y para otras frutas | Los criterios de calidad cambian según el país o la norma (NTC del ICONTEC, Codex Alimentarius, normas de la UE); modelos entrenados con frutas de otro país; regulación de datos (Ley 1581 de 2012 en Colombia, GDPR) | Clases configurables (`config.py`); reentrenar con datos locales; cumplir la ley de protección de datos en la captura |

## 3. Medidas éticas implementadas en el código

- `src/data/segmentation.py`: el fondo se enmascara, así que el modelo no ve el entorno ni personas.
- `src/evaluation/evaluate.py`: métricas por fuente y línea base, para no inflar los resultados.
- `src/app/streamlit_app.py`: muestra las probabilidades y la segmentación (explicabilidad) y avisa cuando la segmentación es dudosa.
- `docs/datos.md`: definiciones de las clases y protocolo de anotación.
- `src/data/convert.py`: al pasar de HEIC a JPG se eliminan los metadatos EXIF (GPS, modelo del teléfono, fecha).
- **Pendiente**: los JPG y PNG originales todavía conservan su EXIF; hay que quitarlo antes de subirlos a la carpeta compartida de toda la clase.
