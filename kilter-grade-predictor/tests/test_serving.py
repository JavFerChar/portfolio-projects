"""Tests for the prediction API (schemas, endpoints)."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.serving.schemas import HealthResponse, Hold, PredictRequest, PredictResponse

from .conftest import DB_PATH, MODEL_PATH, SAMPLE_HOLDS

SAMPLE_PAYLOAD = {"holds": SAMPLE_HOLDS, "angle": 40}


# ---------------------------------------------------------------------------
# Schema validation tests (always run)
# ---------------------------------------------------------------------------
class TestSchemas:
    def test_hold_valid(self):
        h = Hold(x=56, y=16, role="start")
        assert h.x == 56
        assert h.role == "start"

    @pytest.mark.parametrize("role", ["start", "middle", "finish", "foot"])
    def test_hold_valid_roles(self, role: str):
        Hold(x=0, y=0, role=role)

    def test_hold_invalid_role(self):
        with pytest.raises(ValidationError, match="role must be one of"):
            Hold(x=0, y=0, role="crimp")

    def test_hold_x_out_of_bounds(self):
        with pytest.raises(ValidationError):
            Hold(x=200, y=0, role="start")

    def test_hold_y_out_of_bounds(self):
        with pytest.raises(ValidationError):
            Hold(x=0, y=200, role="start")

    def test_hold_negative_coords(self):
        with pytest.raises(ValidationError):
            Hold(x=-1, y=0, role="start")

    def test_predict_request_valid(self):
        req = PredictRequest(**SAMPLE_PAYLOAD)
        assert len(req.holds) == 5
        assert req.angle == 40

    def test_predict_request_too_few_holds(self):
        with pytest.raises(ValidationError):
            PredictRequest(holds=[{"x": 0, "y": 0, "role": "start"}], angle=40)

    def test_predict_request_angle_too_high(self):
        with pytest.raises(ValidationError):
            PredictRequest(holds=SAMPLE_HOLDS, angle=80)

    def test_predict_request_angle_negative(self):
        with pytest.raises(ValidationError):
            PredictRequest(holds=SAMPLE_HOLDS, angle=-5)

    def test_predict_response(self):
        resp = PredictResponse(predicted_grade=20.5, v_grade="V5")
        assert resp.predicted_grade == 20.5

    def test_health_response(self):
        resp = HealthResponse(status="ok", model_loaded=True)
        assert resp.model_loaded is True


# ---------------------------------------------------------------------------
# API unit tests (mocked model, always run)
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_client():
    """TestClient with mocked model and hold_scores (no lifespan, no model file needed)."""
    from src.serving.app import create_app

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([20.0])

    mock_hold_scores = pd.DataFrame(
        {
            "placement_id": [1, 2],
            "hold_usability": [0.5, 0.8],
            "hold_angle_sensitivity": [0.1, 0.2],
        }
    )

    test_app = create_app(use_lifespan=False)
    test_app.state.model = mock_model
    test_app.state.hold_scores = mock_hold_scores

    with TestClient(test_app, raise_server_exceptions=False) as client:
        yield client


class TestHealthEndpoint:
    def test_health_returns_200(self, mock_client: TestClient):
        resp = mock_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    def test_predict_valid_request(self, mock_client: TestClient):
        resp = mock_client.post("/predict", json=SAMPLE_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_grade" in data
        assert "v_grade" in data

    def test_predict_invalid_role_422(self, mock_client: TestClient):
        bad = {
            "holds": [
                {"x": 0, "y": 0, "role": "crimp"},
                {"x": 10, "y": 10, "role": "middle"},
            ],
            "angle": 40,
        }
        resp = mock_client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_one_hold_422(self, mock_client: TestClient):
        bad = {"holds": [{"x": 0, "y": 0, "role": "start"}], "angle": 40}
        resp = mock_client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_angle_too_high_422(self, mock_client: TestClient):
        resp = mock_client.post("/predict", json={**SAMPLE_PAYLOAD, "angle": 80})
        assert resp.status_code == 422

    def test_predict_missing_holds_422(self, mock_client: TestClient):
        resp = mock_client.post("/predict", json={"angle": 40})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Integration tests (skip if model/db not present)
# ---------------------------------------------------------------------------
@pytest.fixture()
def live_client():
    """TestClient with real model (requires model + db files)."""
    from src.serving.app import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.mark.skipif(
    not MODEL_PATH.exists() or not DB_PATH.exists(),
    reason="model or database not available",
)
class TestIntegration:
    def test_predict_returns_valid_grade(self, live_client: TestClient):
        resp = live_client.post("/predict", json=SAMPLE_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert 10 <= data["predicted_grade"] <= 33

    def test_predict_v_grade_format(self, live_client: TestClient):
        resp = live_client.post("/predict", json=SAMPLE_PAYLOAD)
        data = resp.json()
        assert "V" in data["v_grade"]
