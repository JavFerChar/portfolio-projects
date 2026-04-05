"""Inference module: load trained model and predict grade from holds + angle."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.data.ingest import load_difficulty_grades, load_placements
from src.features.hold_usability import (
    HOLD_USABILITY_FEATURE_COLS,
    compute_hold_usability,
)
from src.features.spatial import SPATIAL_FEATURE_COLS, _extract_one

ALL_FEATURE_COLS = SPATIAL_FEATURE_COLS + HOLD_USABILITY_FEATURE_COLS

DEFAULT_MODEL_PATH = Path("models/xgboost_tuned.joblib")
DEFAULT_DB_PATH = Path("data/raw/kilter.db")


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> XGBRegressor:
    """Load a trained XGBoost model from disk."""
    return joblib.load(model_path)


def grade_to_vgrade(grade: float, db_path: Path = DEFAULT_DB_PATH) -> str:
    """Map a continuous grade to the nearest V-grade label."""
    grades_map = load_difficulty_grades(db_path)
    labels = grades_map.set_index("difficulty")["boulder_name"].to_dict()
    nearest = min(labels.keys(), key=lambda k: abs(k - grade))
    return labels[nearest]


def predict_grade(
    holds: list[dict],
    angle: int,
    model: XGBRegressor | None = None,
    hold_scores: pd.DataFrame | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict:
    """Predict grade for a single route.

    Args:
        holds: List of dicts with keys 'x', 'y', 'role'.
        angle: Board angle in degrees.
        model: Pre-loaded model (loaded from disk if None).
        hold_scores: Pre-computed hold usability scores (computed if None).
        db_path: Path to kilter.db for computing hold scores.

    Returns:
        Dict with predicted_grade, v_grade.
    """
    if model is None:
        model = load_model()

    # Spatial features
    spatial = _extract_one(holds, angle)

    # Hold usability features — need hold scores
    if hold_scores is None:
        from src.data.ingest import load_climbs

        climbs = load_climbs(db_path, min_ascents=5)
        placements = load_placements(db_path)
        hold_scores = compute_hold_usability(climbs, placements)

    scores_lookup = hold_scores.set_index("placement_id")
    valid_pids = set(scores_lookup.index)

    # Match holds to placement_ids by (x, y) coordinates
    placements = load_placements(db_path)
    coord_to_pid = {
        (int(r["x"]), int(r["y"])): int(r["placement_id"]) for _, r in placements.iterrows()
    }

    pids = [coord_to_pid.get((h["x"], h["y"])) for h in holds]
    pids = [p for p in pids if p is not None and p in valid_pids]

    if pids:
        u_scores = np.array([float(scores_lookup.loc[pid, "hold_usability"]) for pid in pids])
        a_scores = np.array(
            [float(scores_lookup.loc[pid, "hold_angle_sensitivity"]) for pid in pids]
        )
        hard_threshold = scores_lookup["hold_usability"].quantile(0.75)

        usability_feats = {
            "avg_hold_usability": float(np.mean(u_scores)),
            "min_hold_usability": float(np.min(u_scores)),
            "max_hold_usability": float(np.max(u_scores)),
            "hold_usability_range": float(np.max(u_scores) - np.min(u_scores)),
            "avg_angle_sensitivity": float(np.mean(a_scores)),
            "pct_hard_holds": float(np.mean(u_scores > hard_threshold)),
        }
    else:
        usability_feats = {col: 0.0 for col in HOLD_USABILITY_FEATURE_COLS}

    # Combine into feature vector
    feature_vector = {**spatial, **usability_feats}
    X = np.array([[feature_vector[col] for col in ALL_FEATURE_COLS]])

    predicted_grade = float(model.predict(X)[0])
    v_grade = grade_to_vgrade(predicted_grade)

    return {
        "predicted_grade": round(predicted_grade, 2),
        "v_grade": v_grade,
    }
