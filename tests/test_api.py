import pytest
from fastapi.testclient import TestClient

from src.app.api import app, load_model_and_metrics


@pytest.fixture
def client():
    model, metrics = load_model_and_metrics(app)
    app.state.model = model
    app.state.metrics = metrics
    test_client = TestClient(app)
    yield test_client


@pytest.fixture
def valid_payload():
    return {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }


def test_predict_valid(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert isinstance(prediction, (int, float))
    assert 0 <= prediction <= 10


def test_predict_invalid_missing_feature(client, valid_payload):
    payload = valid_payload.copy()
    del payload["citric_acid"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "metrics" in data
    assert isinstance(data["metrics"], dict)


def test_model_none(client, valid_payload, monkeypatch):
    monkeypatch.setattr(app.state, "model", None)
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 503
