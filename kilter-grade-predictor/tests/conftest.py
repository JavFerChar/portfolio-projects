"""Shared test fixtures and constants."""

from pathlib import Path

MODEL_PATH = Path("models/xgboost_tuned.joblib")
DB_PATH = Path("data/raw/kilter.db")

SAMPLE_HOLDS = [
    {"x": 56, "y": 16, "role": "start"},
    {"x": 72, "y": 48, "role": "middle"},
    {"x": 88, "y": 80, "role": "middle"},
    {"x": 72, "y": 120, "role": "middle"},
    {"x": 80, "y": 152, "role": "finish"},
]
