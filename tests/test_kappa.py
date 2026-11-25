import numpy as np
from sklearn.metrics import cohen_kappa_score

from src.evaluation.kappa import cohen_kappa, fleiss_kappa, interpretar


def test_cohen_igual_a_sklearn():
    rng = np.random.default_rng(1)
    a = rng.choice(["good", "regular", "bad"], 60)
    b = np.where(rng.random(60) < 0.7, a, rng.choice(["good", "regular", "bad"], 60))
    assert np.isclose(cohen_kappa(a, b), cohen_kappa_score(a, b))


def test_fleiss_acuerdo_total_y_dos_anotadores():
    t = np.array([["good"] * 3, ["bad"] * 3, ["regular"] * 3, ["good"] * 3])
    assert np.isclose(fleiss_kappa(t), 1.0)
    # Con todos de acuerdo en solo una clase, p_e = 1 y kappa no está definido
    assert np.isnan(fleiss_kappa(np.array([["good"] * 3] * 4)))


def test_fleiss_ejemplo_conocido():
    # Fleiss (1971): 4 fotos, 3 anotadores, valores calculados a mano
    t = np.array([["good", "good", "bad"],
                  ["good", "good", "good"],
                  ["bad", "bad", "regular"],
                  ["regular", "regular", "regular"]])
    cuentas = np.array([[2, 0, 1], [3, 0, 0], [0, 1, 2], [0, 3, 0]])
    p_i = ((cuentas**2).sum(1) - 3) / 6
    p_k = cuentas.sum(0) / 12
    esperado = (p_i.mean() - (p_k**2).sum()) / (1 - (p_k**2).sum())
    assert np.isclose(fleiss_kappa(t), esperado)


def test_interpretacion():
    assert interpretar(0.72) == "sustancial"
    assert interpretar(0.5) == "moderado"
    assert interpretar(-0.1) == "peor que el azar"
