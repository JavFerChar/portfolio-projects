"""Tests for model training and prediction modules."""

import joblib
import numpy as np
import pytest

from src.models.predict import ALL_FEATURE_COLS, load_model, predict_grade
from src.models.train import load_feature_matrix, split_data, train_baseline

from .conftest import DB_PATH, MODEL_PATH, SAMPLE_HOLDS

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists() or not DB_PATH.exists(),
    reason="model or database not available",
)


@pytest.fixture(scope="module")
def model():
    return load_model(MODEL_PATH)


@pytest.fixture(scope="module")
def data():
    features = load_feature_matrix()
    return split_data(features)


class TestModel:
    def test_prediction_shape(self, model, data):
        X_train, X_val, X_test, y_train, y_val, y_test = data
        preds = model.predict(X_test[:10])
        assert preds.shape == (10,)

    def test_predictions_deterministic(self, model, data):
        _, _, X_test, _, _, _ = data
        sample = X_test[:50]
        preds1 = model.predict(sample)
        preds2 = model.predict(sample)
        np.testing.assert_array_equal(preds1, preds2)

    def test_mae_below_baseline(self, model, data):
        X_train, X_val, X_test, y_train, y_val, y_test = data
        baseline = train_baseline(y_train, y_test)
        preds = model.predict(X_test)
        model_mae = float(np.mean(np.abs(y_test - preds)))
        assert (
            model_mae < baseline["mae"]
        ), f"Model MAE ({model_mae:.3f}) should be below baseline ({baseline['mae']:.3f})"

    def test_serialization_roundtrip(self, model, data, tmp_path):
        _, _, X_test, _, _, _ = data
        sample = X_test[:10]
        preds_before = model.predict(sample)

        path = tmp_path / "model.joblib"
        joblib.dump(model, path)
        loaded = joblib.load(path)
        preds_after = loaded.predict(sample)

        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_predictions_in_grade_range(self, model, data):
        _, _, X_test, _, _, _ = data
        preds = model.predict(X_test)
        assert preds.min() >= 5, f"Min prediction {preds.min():.1f} too low"
        assert preds.max() <= 35, f"Max prediction {preds.max():.1f} too high"

    def test_feature_count_matches(self, model):
        assert model.n_features_in_ == len(ALL_FEATURE_COLS)


class TestPredictGrade:
    def test_predict_returns_dict(self, model):
        result = predict_grade(SAMPLE_HOLDS, angle=40, model=model, db_path=DB_PATH)
        assert "predicted_grade" in result
        assert "v_grade" in result
        assert isinstance(result["predicted_grade"], float)
        assert isinstance(result["v_grade"], str)
        assert 10 <= result["predicted_grade"] <= 33
