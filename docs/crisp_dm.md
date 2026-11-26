# CRISP-DM aplicado al proyecto

> La consigna pide un **diagrama de flujo propio**, no el diagrama genérico de CRISP-DM. Este diagrama muestra lo que hace cada fase del proyecto y en qué script está. GitHub dibuja el mermaid de abajo automáticamente; el **export vectorial para el informe** está en [`docs/figuras/crisp_dm.svg`](figuras/crisp_dm.svg) (se regenera con `python scripts/gen_crisp_svg.py`).

![Diagrama CRISP-DM del proyecto](figuras/crisp_dm.svg)

```mermaid
flowchart TD
    subgraph N["1 · Comprensión del negocio"]
        N1["Clasificación manual lenta y subjetiva<br/>en plazas de mercado"] --> N2["Objetivo: calidad (good/regular/bad)<br/>+ tamaño (pequeño/mediano/grande)"]
        N2 --> N3["Métrica de éxito: F1 macro ≥ 0.80<br/>en fotos propias de test"]
    end
    subgraph D["2 · Comprensión de los datos"]
        D1["Kaggle: Fruit Quality Classification<br/>(good / bad / mixed)"]
        D2["Fotos propias en Drive<br/>(good / regular / bad) · sin IA"]
        D1 & D2 --> D3["EDA: clases por fruta y fuente,<br/>resolución, fondo, desbalance<br/>notebooks/01_eda.ipynb"]
    end
    subgraph P["3 · Preparación"]
        P0["mixed_quality → recortes individuales<br/>src/data/segment_mixed.py"]
        P1["Índice + anotación manual<br/>src/data/dataset.py"]
        P2["Segmentación HSV + Otsu<br/>recorte 128×128 con fondo negro<br/>src/data/preprocess.py"]
        P3["Tamaño: d_eq = √(4A/π) normalizado<br/>terciles por fruta"]
        P4["Split 70/15/15 estratificado por calidad<br/>y agrupado por casi duplicados"]
        P0 --> P1 --> P2 --> P3 --> P4
    end
    subgraph M["4 · Modelado"]
        M1["62 características manuales<br/>color · textura · defectos · forma"]
        M2["SVM · Random Forest · XGBoost · LogReg · KNN · MLP<br/>GridSearchCV k=5, F1 macro"]
        M3["CNN pequeña desde cero<br/>3 bloques conv + GAP, aumento de datos"]
        M1 --> M2
    end
    subgraph E["5 · Evaluación"]
        E1["Test: accuracy, P, R, F1 macro<br/>vs. línea base · IC 95 % y McNemar"]
        E2["Matrices de confusión, curvas,<br/>F1 Kaggle vs. fotos propias"]
        E3{"¿Cumple la métrica<br/>en fotos propias?"}
        E1 --> E2 --> E3
    end
    subgraph S["6 · Despliegue"]
        S1["App Streamlit: subir foto o cámara<br/>src/app/streamlit_app.py"]
    end
    N3 --> D1
    D3 --> P0
    P4 --> M1 & M3
    M2 & M3 --> E1
    E3 -- "No: más datos / ajustar" --> P1
    E3 -- "Sí" --> S1
```

## Evidencia por fase

| Fase | Pregunta que responde | Evidencia en el repo |
|------|-----------------------|----------------------|
| Negocio | ¿Por qué importa clasificar calidad? | `report/informe.md` §Introducción, `docs/etica_impactos.md` |
| Datos | ¿Qué datos hay y qué problemas tienen? | `notebooks/01_eda.ipynb`, `data/processed/index.csv` |
| Preparación | ¿Cómo quedan listos para modelar? | `src/data/*`, `docs/datos.md`, `data/anotaciones_propias.csv` |
| Modelado | ¿Qué modelos y con qué hiperparámetros? | `src/models/*`, `src/training/*`, `experiments/results/cv_*.csv` |
| Evaluación | ¿Qué tan bien funcionan y dónde fallan? | `experiments/results/comparacion_modelos.csv`, `docs/figuras/*.svg` |
| Despliegue | ¿Cómo lo usa una persona? | `src/app/streamlit_app.py`, video de demostración |
