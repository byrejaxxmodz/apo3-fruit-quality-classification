"""Modelos clásicos de machine learning y sus rejillas de hiperparámetros.

Todos van dentro de un Pipeline con StandardScaler:  z = (x - mu) / sigma,
indispensable para SVM, KNN, regresión logística y MLP, que dependen de
distancias o de productos punto entre características de escalas distintas.

`class_weight="balanced"` pondera cada clase k por  n / (K * n_k)  para
compensar el desbalance (p. ej. hay pocas fotos "regular").
"""
from __future__ import annotations

from typing import Dict, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config

# Modelos sin parámetro class_weight: reciben pesos por muestra al entrenar
SAMPLE_WEIGHTED = {"xgboost"}


def get_models() -> Dict[str, Tuple[Pipeline, dict]]:
    """Devuelve {nombre: (pipeline, rejilla)} para GridSearchCV."""
    s = config.SEED
    models = {
        # Línea base: frontera lineal, softmax sobre w^T x + b
        "logreg": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        # SVM con kernel RBF: K(x, x') = exp(-gamma ||x - x'||^2)
        "svm": (
            Pipeline([("scaler", StandardScaler()),
                      # probability se activa solo en el reentrenamiento final
                      # (train_classical.py): la calibración de Platt hace 5
                      # ajustes internos extra por cada combinación
                      ("clf", SVC(probability=False, class_weight="balanced", random_state=s))]),
            {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001],
             "clf__kernel": ["rbf"]},
        ),
        # Random Forest: bagging de árboles + submuestreo de características
        "random_forest": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", RandomForestClassifier(class_weight="balanced", random_state=s, n_jobs=-1))]),
            {"clf__n_estimators": [200, 500], "clf__max_depth": [None, 10, 20],
             "clf__min_samples_leaf": [1, 3]},
        ),
        # KNN: voto de los k vecinos más cercanos en distancia euclidiana
        "knn": (
            Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3, 5, 9, 15], "clf__weights": ["uniform", "distance"]},
        ),
        # MLP sobre características manuales (permitido por la consigna)
        "mlp": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", MLPClassifier(max_iter=1000, early_stopping=True, random_state=s))]),
            {"clf__hidden_layer_sizes": [(64,), (128, 64)], "clf__alpha": [1e-4, 1e-2]},
        ),
    }
    try:  # XGBoost es opcional
        from xgboost import XGBClassifier

        # Boosting (Unidad 2): cada árbol nuevo ajusta el gradiente de la pérdida
        # logarítmica del ensamble anterior, F_m = F_{m-1} + eta * h_m.
        # XGBClassifier no acepta class_weight: el desbalance se compensa con
        # pesos por muestra w_i = n / (K n_{y_i}) que pasa train_classical.py
        # (SAMPLE_WEIGHTED), equivalentes a class_weight="balanced".
        models["xgboost"] = (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", XGBClassifier(eval_metric="mlogloss", tree_method="hist",
                                            n_jobs=-1, random_state=s))]),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 6],
             "clf__learning_rate": [0.05, 0.1]},
        )
    except ImportError:
        pass
    return models
