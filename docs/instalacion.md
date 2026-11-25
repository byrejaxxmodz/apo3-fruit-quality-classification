# Instalación y ejecución

## 1. Entorno

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Funciona con Python 3.9 o superior. No hace falta GPU: si hay CUDA o Apple MPS, la CNN los usa sola.

Dos detalles del entorno que afectan la reproducibilidad:

- **scikit-learn está fijado en 1.7.2.** Con la 1.9, `StratifiedGroupKFold(shuffle=True)` reparte distinto y la partición train/val/test deja de coincidir con la de los resultados versionados.
- **Cada paso del pipeline corre en su propio proceso** cuando se piden varios (`python -m src.main all` o `python -m src.main train search_cnn ...`). scikit-learn, XGBoost y PyTorch traen cada uno su copia de OpenMP y, cargadas juntas, pueden bloquear el proceso para siempre (pasó en macOS al entrenar la CNN después de XGBoost). Por la misma razón, la inferencia de la CNN en `evaluate.py` y en la app usa un solo hilo.

## 2. Datos

**Opción automática:**

```bash
python -m src.main download
```

- **Kaggle:** crear un token en kaggle.com → Settings → API → *Create New Token* y guardarlo en `~/.kaggle/kaggle.json`.
- **Drive:** la carpeta `proyectoAPO` debe estar compartida como «Cualquier persona con el enlace puede ver».

**Opción manual:** descargar las carpetas y copiarlas así:

```
data/raw/kaggle/    <- contenido del zip de Kaggle
data/raw/propias/   <- Good_Quality_Fruits/, Regular_Quality_Fruits/, ...
```

## 3. Pipeline completo

```bash
python -m src.main mixed        # opcional: recortes de mixed_quality
python -m src.main all          # index -> preprocess -> features -> feature_space -> train
                                #   -> train_cnn -> evaluate -> intervals -> evaluate_size
```

Cada paso se puede correr por separado (`python -m src.main preprocess`, etc.).

**Agregar un modelo sin repetir los demás.** Por ejemplo, para XGBoost:

```bash
python -m src.training.train_classical --models xgboost
python -m src.evaluation.evaluate --models xgboost
python -m src.main intervals
```

`evaluate --models` agrega las filas del modelo a `comparacion_modelos.csv` y `f1_por_fuente.csv` sin borrar las de los otros. Antes verifica que la partición de test sea la misma con la que se calcularon los resultados guardados; si cambió (por ejemplo, porque se agregaron fotos al Drive), se detiene y hay que correr la evaluación completa.

**Análisis estadístico sin las imágenes.** `python -m src.main intervals` funciona aunque no estén descargados los datos: reconstruye el conjunto de test a partir de los `errores_*.csv` versionados. En ese caso el bootstrap es por imagen; con `data/processed/splits.csv` presente es por grupo de casi duplicados, que es más honesto.

## 4. App

```bash
streamlit run src/app/streamlit_app.py
```

## 5. Pruebas

```bash
pytest -q
```
