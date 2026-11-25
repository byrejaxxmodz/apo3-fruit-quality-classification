"""Pruebas de las fórmulas estadísticas de src/evaluation/intervals.py."""
import math

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.evaluation.intervals import f1_macro_cm, holm, mcnemar_exact, wald_se, wilson


def test_wald_se():
    assert wald_se(50, 100) == pytest.approx(0.05)
    assert math.isnan(wald_se(0, 0))


def test_wilson_small_sample():
    # 5 aciertos de 5: Wald daría [1, 1]; Wilson deja un límite inferior realista
    lo, hi = wilson(5, 5)
    assert hi == pytest.approx(1.0)
    assert lo == pytest.approx(0.5655, abs=1e-3)
    lo, hi = wilson(3, 5)   # 3 de 5 bad propias (versión con 605 fotos): IC muy ancho
    assert lo < 0.6 < hi and hi - lo > 0.5


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 2 ** 10)
    assert mcnemar_exact(7, 3) == mcnemar_exact(3, 7)
    assert mcnemar_exact(5, 5) == 1.0


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_f1_macro_cm_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(3, size=300)
    p = np.where(rng.random(300) < 0.7, y, rng.integers(3, size=300))
    cm = np.zeros((3, 3))
    np.add.at(cm, (y, p), 1)
    for labels in ([0, 1, 2], [0, 2]):  # [0, 2]: fuente sin la clase 1 (como Kaggle sin «regular»)
        assert f1_macro_cm(cm, labels) == pytest.approx(
            f1_score(y, p, labels=labels, average="macro", zero_division=0))
