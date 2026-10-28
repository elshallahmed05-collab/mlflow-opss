
import src.api as api
from fastapi.testclient import TestClient


def test_home_endpoint():
    with TestClient(api.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Churn Prediction API"


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: None)

    with TestClient(api.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
