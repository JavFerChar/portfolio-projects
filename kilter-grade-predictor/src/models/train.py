"""Training pipeline for Kilter Board grade prediction with MLflow tracking."""

from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.data.ingest import load_climbs, load_placements
from src.features.hold_usability import (
    HOLD_USABILITY_FEATURE_COLS,
    aggregate_angle_conditioned_features,
    aggregate_hold_usability_features,
    aggregate_role_typical_grade_features,
    compute_hold_usability,
    compute_hold_usability_by_angle,
    compute_role_typical_grade,
)
from src.features.spatial import SPATIAL_FEATURE_COLS, extract_spatial_features

ALL_FEATURE_COLS = SPATIAL_FEATURE_COLS + HOLD_USABILITY_FEATURE_COLS

DB_PATH = Path("data/raw/kilter.db")
MODEL_DIR = Path("models")
RANDOM_STATE = 42


def load_feature_matrix() -> pd.DataFrame:
    """Load or compute the full feature matrix."""
    cache_path = Path("data/processed/features.parquet")
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    climbs = load_climbs(DB_PATH, min_ascents=5)
    placements = load_placements(DB_PATH)

    spatial = extract_spatial_features(climbs, placements)
    hold_scores = compute_hold_usability(climbs, placements)
    usability = aggregate_hold_usability_features(climbs, hold_scores)
    hold_scores_by_angle = compute_hold_usability_by_angle(climbs, placements)
    angle_features = aggregate_angle_conditioned_features(climbs, hold_scores_by_angle)
    role_scores = compute_role_typical_grade(climbs, placements)
    role_features = aggregate_role_typical_grade_features(climbs, role_scores)

    features = (
        spatial.merge(usability, on="climb_uuid")
        .merge(angle_features, on="climb_uuid")
        .merge(role_features, on="climb_uuid")
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(cache_path, index=False)
    return features


def split_data(
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split (70/15/15).

    Stratifies by grade bucket to handle imbalanced grade distribution.
    """
    X = features[ALL_FEATURE_COLS].values
    y = features["grade"].values

    # Bin grades for stratification
    grade_bins = pd.cut(y, bins=[0, 14, 18, 22, 26, 40], labels=False)

    X_train, X_temp, y_train, y_temp, strat_train, strat_temp = train_test_split(
        X, y, grade_bins, test_size=0.3, random_state=RANDOM_STATE, stratify=grade_bins
    )
    # Split temp into val and test (50/50 of the 30% = 15% each)
    strat_temp_series = pd.Series(strat_temp).reset_index(drop=True)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=strat_temp_series
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_baseline(y_train: np.ndarray, y_val: np.ndarray) -> dict:
    """Baseline: predict mean grade for every route."""
    mean_pred = np.full_like(y_val, y_train.mean())
    mae = float(np.mean(np.abs(y_val - mean_pred)))
    rmse = float(np.sqrt(np.mean((y_val - mean_pred) ** 2)))
    return {"mae": mae, "rmse": rmse, "mean_grade": float(y_train.mean())}


def train_xgboost_default(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[XGBRegressor, dict]:
    """Train XGBoost with default hyperparameters."""
    model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = float(np.mean(np.abs(y_val - preds)))
    rmse = float(np.sqrt(np.mean((y_val - preds) ** 2)))
    r2 = float(1 - np.sum((y_val - preds) ** 2) / np.sum((y_val - y_val.mean()) ** 2))
    return model, {"mae": mae, "rmse": rmse, "r2": r2}


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
) -> tuple[XGBRegressor, dict, optuna.Study]:
    """Tune XGBoost with Optuna (Bayesian optimization)."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return float(np.mean(np.abs(y_val - preds)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Retrain best model
    best_model = XGBRegressor(**study.best_params, random_state=RANDOM_STATE, n_jobs=-1)
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_val)
    mae = float(np.mean(np.abs(y_val - preds)))
    rmse = float(np.sqrt(np.mean((y_val - preds) ** 2)))
    r2 = float(1 - np.sum((y_val - preds) ** 2) / np.sum((y_val - y_val.mean()) ** 2))

    return best_model, {"mae": mae, "rmse": rmse, "r2": r2, **study.best_params}, study


def evaluate_on_test(
    model: XGBRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Final evaluation on held-out test set."""
    preds = model.predict(X_test)
    mae = float(np.mean(np.abs(y_test - preds)))
    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
    r2 = float(1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    return {"test_mae": mae, "test_rmse": rmse, "test_r2": r2}


def generate_shap_plots(
    model: XGBRegressor,
    X_val: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate and save SHAP explainability plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_val[:1000])  # sample for speed
    shap_values.feature_names = ALL_FEATURE_COLS

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.summary_plot(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot (mean absolute SHAP)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()


def run() -> None:
    """Run the full training pipeline."""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("kilter-grade-prediction")

    print("Loading feature matrix...")
    features = load_feature_matrix()
    print(f"  {len(features):,} routes × {len(ALL_FEATURE_COLS)} features")

    print("Splitting data (70/15/15, stratified)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features)
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # --- Baseline ---
    print("\n--- Baseline (mean prediction) ---")
    with mlflow.start_run(run_name="baseline-mean"):
        baseline = train_baseline(y_train, y_val)
        mlflow.log_metrics(baseline)
        print(f"  MAE: {baseline['mae']:.3f} | RMSE: {baseline['rmse']:.3f}")

    # --- XGBoost default ---
    print("\n--- XGBoost (default params) ---")
    with mlflow.start_run(run_name="xgboost-default"):
        model_default, metrics_default = train_xgboost_default(X_train, y_train, X_val, y_val)
        mlflow.log_metrics(metrics_default)
        print(f"  MAE: {metrics_default['mae']:.3f} | RMSE: {metrics_default['rmse']:.3f}")
        print(f"  R²: {metrics_default['r2']:.3f}")

    # --- XGBoost tuned ---
    print("\n--- XGBoost (Optuna tuning, 30 trials) ---")
    with mlflow.start_run(run_name="xgboost-tuned"):
        model_tuned, metrics_tuned, study = tune_xgboost(
            X_train, y_train, X_val, y_val, n_trials=30
        )
        mlflow.log_metrics({k: v for k, v in metrics_tuned.items() if isinstance(v, int | float)})
        mlflow.log_params(study.best_params)
        print(f"  MAE: {metrics_tuned['mae']:.3f} | RMSE: {metrics_tuned['rmse']:.3f}")
        print(f"  R²: {metrics_tuned['r2']:.3f}")
        print(f"  Best params: {study.best_params}")

        # Test set evaluation
        test_metrics = evaluate_on_test(model_tuned, X_test, y_test)
        mlflow.log_metrics(test_metrics)
        print(f"\n  Test MAE: {test_metrics['test_mae']:.3f}")
        print(f"  Test RMSE: {test_metrics['test_rmse']:.3f}")
        print(f"  Test R²: {test_metrics['test_r2']:.3f}")

        # SHAP
        print("\nGenerating SHAP plots...")
        generate_shap_plots(model_tuned, X_val, Path("models/shap"))
        mlflow.log_artifacts("models/shap", artifact_path="shap")

        # Save model
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "xgboost_tuned.joblib"
        joblib.dump(model_tuned, model_path)
        mlflow.log_artifact(str(model_path))
        print(f"  Model saved to {model_path}")

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"  Baseline MAE:     {baseline['mae']:.3f}")
    print(f"  XGBoost default:  {metrics_default['mae']:.3f}")
    print(f"  XGBoost tuned:    {metrics_tuned['mae']:.3f}")
    print(f"  Test MAE:         {test_metrics['test_mae']:.3f}")
    improvement = (baseline["mae"] - test_metrics["test_mae"]) / baseline["mae"] * 100
    print(f"  Improvement over baseline: {improvement:.1f}%")


if __name__ == "__main__":
    run()
