# Clasificación automática de calidad y tamaño de frutas

Proyecto final de **Algoritmos y Programación III** (2026-2), Ingeniería de Sistemas, Universidad Icesi.

#### Estado del proyecto: Activo

## Integrantes

| Nombre | GitHub | Rol |
|--------|--------|-----|
| _Nombre 1_ | [@usuario](https://github.com/usuario) | Líder · modelado |
| _Nombre 2_ | [@usuario](https://github.com/usuario) | Datos · anotación |
| _Nombre 3_ | [@usuario](https://github.com/usuario) | Despliegue · informe |

**Profesor:** Milton Sarria

## Objetivo
Construir un sistema de visión por computador que, a partir de la foto de **una** fruta sobre un fondo uniforme, prediga su **clase de calidad** (*good*, *regular* o *bad*) y su **tamaño** (pequeño, mediano o grande), para apoyar la clasificación manual en plazas de mercado, que hoy es lenta y subjetiva.

## Métodos
- Metodología **CRISP-DM** ([diagrama propio](docs/crisp_dm.md))
- Segmentación clásica (HSV + Otsu + morfología) y estimación de tamaño por diámetro equivalente
- 62 características manuales (color, textura LBP/GLCM, defectos, forma)
- ML clásico con `GridSearchCV`: **SVM**, **Random Forest**, regresión logística, KNN, MLP (y XGBoost opcional)
- Deep learning: **CNN pequeña entrenada desde cero**
- App en **Streamlit** con subida de imagen o cámara

## Datos
- [Fruit Quality Classification](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification) (Kaggle)
- Fotos propias del grupo (carpeta de Drive `proyectoAPO`)
- Detalles, definiciones de las clases y protocolo de anotación: [docs/datos.md](docs/datos.md)

## Estructura del repositorio (por fase de CRISP-DM)

```
calidad-frutas/
├── data/
│   ├── raw/kaggle/            # [Datos] dataset de Kaggle (no se versiona)
│   ├── raw/propias/           # [Datos] fotos del grupo (no se versiona)
│   ├── interim/               # [Preparación] recortes de mixed_quality
│   ├── processed/             # [Preparación] index.csv, splits.csv, features.npz, crops/
│   └── anotaciones_propias.csv# [Preparación] etiquetado manual (SÍ se versiona)
├── notebooks/
│   └── 01_eda.ipynb           # [Datos] análisis exploratorio
├── src/
│   ├── config.py              # rutas, clases, semillas
│   ├── data/                  # [Datos/Preparación] descarga, índice, segmentación, splits
│   ├── features/              # [Preparación] características manuales
│   ├── models/                # [Modelado] modelos clásicos y CNN
│   ├── training/              # [Modelado] GridSearchCV y entrenamiento de la CNN
│   ├── evaluation/            # [Evaluación] métricas, matrices de confusión, errores
│   ├── app/                   # [Despliegue] app Streamlit
│   ├── predict.py             # [Despliegue] inferencia sobre una imagen
│   └── main.py                # orquestador del pipeline
├── experiments/
│   ├── checkpoints/           # modelos entrenados (no se versionan)
│   ├── logs/                  # historial de entrenamiento
│   └── results/               # CSV de métricas (sí se versionan)
├── docs/
│   ├── crisp_dm.md            # diagrama de flujo propio + evidencias
│   ├── datos.md               # fuentes, clases, anotación
│   ├── formulacion_matematica.md   # PI3
│   ├── etica_impactos.md      # PI1 y PI2
│   ├── instalacion.md
│   └── figuras/               # figuras SVG para el informe
├── report/informe.md          # borrador del informe (máx. 7 páginas)
├── tests/                     # pruebas con pytest
└── requirements.txt
```

## Cómo empezar

```bash
pip install -r requirements.txt
python -m src.main download     # Kaggle + Drive (ver docs/instalacion.md)
python -m src.main all          # index → preprocess → features → train → train_cnn → evaluate
streamlit run src/app/streamlit_app.py
```

## Entregables
- [ ] Informe final (máx. 7 páginas, IEEE): [report/informe.md](report/informe.md)
- [ ] Video de máximo 10 minutos: _enlace_
- [ ] App funcionando: `src/app/streamlit_app.py`

## Créditos y referencias
Dataset base: R. Park, *Fruit Quality Classification*, Kaggle. La plantilla de estructura sigue la sugerencia del curso ([miltonsarria/ETLInfrati](https://github.com/miltonsarria/ETLInfrati)). Todo código o dato de terceros se cita en el informe.
