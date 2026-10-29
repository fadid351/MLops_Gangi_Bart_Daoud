"""Tests des routes de l'API FastAPI (ignorés si FastAPI / httpx ne sont pas installés)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from src.api import main  # noqa: E402
from tests.conftest import PERSON  # noqa: E402


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_predict_without_model_returns_503(client):
    main.state["predictor"] = None
    assert client.post("/predict", json=PERSON).status_code == 503


def test_predict(client, predictor):
    main.state["predictor"] = predictor
    response = client.post("/predict", json=PERSON)
    assert response.status_code == 200
    assert response.json()["prediction"] in {">50K", "<=50K"}


def test_predict_batch(client, predictor):
    main.state["predictor"] = predictor
    response = client.post("/predict/batch", json=[PERSON, PERSON])
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_invalid_input_returns_422(client, predictor):
    main.state["predictor"] = predictor
    assert client.post("/predict", json={**PERSON, "age": -5}).status_code == 422
