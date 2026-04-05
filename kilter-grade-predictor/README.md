# Kilter Board Grade Predictor

Predict Kilter Board climbing route difficulty from hold positions and board angle.

## Overview

Given a set of holds on a [Kilter Board](https://settercloset.com/pages/the-kilter-board) and a wall angle, this project predicts the route's difficulty grade using spatial feature engineering and gradient-boosted trees. The dataset comes from the Kilter Board community database via [BoardLib](https://github.com/lemeryfertitta/BoardLib) — roughly 67K route-angle combinations with consensus grades ranging from V0 to V14.

The core challenge is that the database contains no hold-type labels (crimp, jug, sloper, etc.). We infer hold quality entirely from usage patterns: which grades and angles each hold appears in.

## Results

| Model | Val MAE | Test MAE | Test R² |
|---|---|---|---|
| Baseline (mean) | 3.838 | — | — |
| XGBoost default | 0.998 | — | — |
| **XGBoost tuned** | **0.278** | **0.282** | **0.971** |

93% improvement over baseline. A MAE of 0.282 means predictions land within roughly one-quarter of a grade step on the Kilter difficulty scale.

## Feature Engineering

The model uses 29 features across two categories.

### Spatial features (23)

Extracted from hold coordinates and route geometry: hold count, start/finish positions, spatial spread, convex hull area, move distances (avg, max), path length, lateral ratio, direction changes, and angle interaction terms.

### Hold usability (6)

No hold-type labels exist in the database, so we infer hold "quality" from the grades of routes each hold appears in. A good hold (jug-like) appears in easy routes even at steep angles; a bad hold (crimp-like) only appears in hard routes. For each hold, the weighted mean grade of routes it appears in. Aggregated per route as avg, min, max, range, angle sensitivity, and percentage of hard holds.

**This is the #1 feature by SHAP importance** — `avg_hold_usability` alone contributes more to predictions than any spatial feature, validating the approach.

## Project Structure

```
src/
  data/ingest.py              # SQLite → pandas, frames parsing, role mapping
  features/spatial.py         # 23 spatial and displacement features
  features/hold_usability.py  # Hold quality inference from usage patterns
  models/train.py             # XGBoost + Optuna + MLflow pipeline
  models/predict.py           # Single-route inference
notebooks/
  01_eda.ipynb                # Grade/angle distributions, hold position heatmaps
  02_features.ipynb           # Feature correlations, hold usability board heatmaps
tests/                        # 41 tests (data, features, model)
```

## Quick Start

```bash
# Install dependencies
make install

# Download Kilter Board database
uv run boardlib database kilter data/raw/kilter.db

# Run tests
make test

# Train the model
make train

# View MLflow experiment tracking
uv run mlflow ui

# Refresh data (incremental sync) and retrain
uv run boardlib database kilter data/raw/kilter.db
rm data/processed/features.parquet  # clear feature cache
make train
```

## Data Freshness

The Kilter Board community adds ~6,500 new routes per month. The model trains on a snapshot of the database at download time. Since the board hardware (holds, positions) doesn't change and hold usability scores stabilise after ~20 routes per hold, the model generalises well without frequent retraining. To refresh, re-run `boardlib database kilter` (incremental sync) followed by `make train`.

## Project Status

| Phase | Status |
|---|---|
| Foundation (repo, CI, linting) | Done |
| Data ingestion + EDA | Done |
| Feature engineering | Done |
| Modeling (XGBoost + MLflow) | Done |
| Board image visualizations | Done |
| Serving (FastAPI + Streamlit) | Planned |
| Production polish (Docker, CI/CD) | Planned |

See [ROADMAP.md](ROADMAP.md) for detailed next steps.

## Stack

Python 3.12, uv, BoardLib, XGBoost, Optuna, MLflow, SHAP, pandas, scikit-learn, pytest, GitHub Actions

## License

MIT
