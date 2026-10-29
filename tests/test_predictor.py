"""Tests de la logique de prédiction (sans serveur web)."""

from tests.conftest import PERSON


def test_predictor_output(predictor):
    out = predictor.predict([PERSON])[0]
    assert 0 <= out["probabilite_50k"] <= 1
    assert out["prediction"] in {">50K", "<=50K"}
    assert out["seuil"] == 0.4


def test_predictor_accepts_missing_values(predictor):
    person = {**PERSON, "workclass": None, "native-country": "Atlantis"}
    assert len(predictor.predict([person, PERSON])) == 2
