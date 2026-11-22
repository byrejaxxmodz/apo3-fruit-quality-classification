# Instalación y ejecución

## 1. Entorno

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Funciona con Python 3.9 o superior. No hace falta GPU: si hay CUDA o Apple MPS, la CNN los usa sola.

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
python -m src.main all          # index -> preprocess -> features -> train -> train_cnn -> evaluate
```

Cada paso se puede correr por separado (`python -m src.main preprocess`, etc.).

## 4. App

```bash
streamlit run src/app/streamlit_app.py
```

## 5. Pruebas

```bash
pytest -q
```
