# Ética e impactos (evidencias PI1 y PI2)

> Borrador para completar y ajustar con las decisiones reales del grupo. La rúbrica pide una **tabla** que relacione cada decisión con el código de ética y una **matriz de impactos** con propuestas de mitigación.
>
> Código de referencia: *Software Engineering Code of Ethics and Professional Practice* (ACM/IEEE-CS) y el código de ética de COPNIA (Ley 842 de 2003, Colombia). Deben citarse en IEEE.

## 1. Dilemas éticos identificados (PI1)

| # | Situación o decisión del proyecto | Riesgo ético | Principio del código | Medida adoptada |
|---|-----------------------------------|--------------|----------------------|-----------------|
| 1 | Tomar fotos en plazas de mercado y supermercados | Captar sin permiso a personas, a vendedores o la marca de un puesto | ACM/IEEE 1.03 (bien público), privacidad | Pedir permiso al vendedor; encuadrar solo la fruta; revisar y descartar fotos con rostros; quitar los metadatos EXIF (GPS) antes de compartir |
| 2 | Compartir el dataset con toda la clase | Uso de los datos para otros fines, sin trazabilidad | ACM/IEEE 3.13 (datos confidenciales), 1.06 (honestidad) | Licencia y README del dataset; registro de quién aportó cada foto (`anotador`) |
| 3 | Usar el dataset de Kaggle | Plagio o violación de la licencia | ACM/IEEE 2.02, 3.03; la consigna considera fraude no citar | Citar la fuente en IEEE en el informe y en `docs/datos.md`; respetar su licencia |
| 4 | Sesgo en las clases (regular solo en fotos propias, bad solo en Kaggle) | Modelo que «acierta» por el fondo o la cámara: falsa sensación de exactitud | ACM/IEEE 3.10 (pruebas adecuadas), 1.06 (no exagerar capacidades) | Fondo enmascarado; métrica por fuente; completar las clases con fotos propias; reportar esta limitación con honestidad |
| 5 | Sesgo por variedad, región o iluminación | El sistema castiga productos locales que no se parecen a los de Kaggle | Equidad y no discriminación | Incluir variedades locales (banano criollo, naranja valencia); análisis de errores por fruta |
| 6 | Automatizar una decisión que afecta los ingresos de un productor | Un falso «bad» hace que se descarte o se pague menos por un producto bueno | ACM/IEEE 1.03, 1.04 (avisar riesgos) | El sistema **recomienda** y una persona decide; se muestra la probabilidad y la segmentación; umbral conservador para «bad» |
| 7 | Etiquetado subjetivo | Etiquetas inconsistentes que pasan a ser «verdad» del modelo | ACM/IEEE 6.07 (exactitud) | Definiciones operativas escritas; doble anotación y kappa de Cohen |
| 8 | Usar un dataset público (Kaggle/FruitNet) cuyas imágenes **conservan coordenadas GPS** en el EXIF | Redistribuir la ubicación de los autores o de los lugares de captura | ACM/IEEE 3.12 y 3.13 (privacidad y datos); Ley 1581 de 2012 | No se redistribuyen las imágenes crudas (están en `.gitignore`); los recortes procesados se guardan como PNG sin EXIF; se reporta el hallazgo |
| 9 | Presentar los resultados | Reportar solo la mejor corrida o el test usado para ajustar | ACM/IEEE 1.06, 3.10 | El test se usa una sola vez; se reportan la media y la desviación de la CV y la línea base |

## 2. Matriz de impactos (PI2)

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
