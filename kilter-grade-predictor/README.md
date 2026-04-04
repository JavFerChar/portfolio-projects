# Kilter Board Grade Predictor

Predict Kilter Board climbing route difficulty from hold positions and board angle.

## Overview

Given a set of holds on a [Kilter Board](https://settercloset.com/pages/the-kilter-board) and a wall angle, this project predicts the route's difficulty grade using spatial feature engineering and gradient-boosted trees.

## Quick Start

```bash
# Install dependencies
make install

# Download Kilter Board database
uv run boardlib download kilter data/raw/kilter.db

# Run tests
make test

# Train the model
make train

# Start API + dashboard
make docker-up
```

## Project Status

> Work in progress — see [ROADMAP.md](ROADMAP.md) for planned features.

| Phase | Status |
|---|---|
| Foundation (repo, CI) | In progress |
| Data ingestion + EDA | Planned |
| Feature engineering | Planned |
| Modeling (XGBoost + MLflow) | Planned |
| Serving (FastAPI + Streamlit) | Planned |
| Production polish | Planned |

## Stack

Python 3.12, uv, BoardLib, FastAPI, Streamlit, Docker, MLflow, XGBoost, Optuna, SHAP

## License

MIT
