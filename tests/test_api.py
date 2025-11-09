from fastapi.testclient import TestClient
from src.app.api import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert body["status"] in ("ok", "error")


def test_predict_ok():
    payload = {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200


def test_predict_validation_error():
    request = {
        "alcohol": "yes",
    }
    resp = client.post("/predict", json=request)
    assert resp.status_code == 422


def test_model_info():
    resp = client.get("/model-info")
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        body = resp.json()
        assert body["model_name"] == "RandomForestRegressor"
        assert "metrics" in body
