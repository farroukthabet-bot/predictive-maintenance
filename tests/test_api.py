import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint_returns_ok():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_rejects_wrong_shape():
    """Sending the wrong number of cycles/features should return an error, not crash."""
    bad_data = {"readings": [[1.0, 2.0, 3.0]]}  # way too small
    response = client.post("/predict", json=bad_data)
    assert response.status_code == 200  # our API returns a 200 with an error message, not a 500
    assert "error" in response.json()

def test_predict_returns_valid_rul_for_correct_shape():
    """A correctly-shaped request should return a plausible RUL prediction."""
    fake_reading = [[0.0] * 24 for _ in range(30)]  # 30 cycles, 24 features, all zeros
    response = client.post("/predict", json={"readings": fake_reading})
    assert response.status_code == 200
    result = response.json()
    assert "predicted_RUL" in result
    assert 0 <= result["predicted_RUL"] <= 130  # sanity range, capped at 125 +buffer