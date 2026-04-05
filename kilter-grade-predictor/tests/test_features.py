"""Tests for feature engineering modules."""

from pathlib import Path

import numpy as np
import pytest

from src.data.ingest import load_climbs, load_placements
from src.features.hold_usability import (
    HOLD_USABILITY_FEATURE_COLS,
    aggregate_hold_usability_features,
    compute_hold_usability,
)
from src.features.spatial import SPATIAL_FEATURE_COLS, extract_spatial_features

DB_PATH = Path("data/raw/kilter.db")

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="kilter.db not downloaded")


@pytest.fixture(scope="module")
def climbs():
    return load_climbs(DB_PATH, min_ascents=5).head(500)


@pytest.fixture(scope="module")
def placements():
    return load_placements(DB_PATH)


@pytest.fixture(scope="module")
def spatial_features(climbs, placements):
    return extract_spatial_features(climbs, placements)


@pytest.fixture(scope="module")
def hold_scores(placements):
    # Use full dataset for hold scores (need enough routes per hold)
    full_climbs = load_climbs(DB_PATH, min_ascents=5)
    return compute_hold_usability(full_climbs, placements)


@pytest.fixture(scope="module")
def usability_features(climbs, hold_scores):
    return aggregate_hold_usability_features(climbs, hold_scores)


class TestSpatialFeatures:
    def test_output_shape(self, spatial_features, climbs):
        assert len(spatial_features) == len(climbs)
        assert set(SPATIAL_FEATURE_COLS).issubset(set(spatial_features.columns))

    def test_no_nans(self, spatial_features):
        assert spatial_features[SPATIAL_FEATURE_COLS].isna().sum().sum() == 0

    def test_hold_count_min(self, spatial_features):
        assert (spatial_features["hold_count"] >= 2).all()

    def test_angle_range(self, spatial_features):
        assert spatial_features["angle"].min() >= 0
        assert spatial_features["angle"].max() <= 70

    def test_convex_hull_non_negative(self, spatial_features):
        assert (spatial_features["convex_hull_area"] >= 0).all()

    def test_distances_non_negative(self, spatial_features):
        for col in ["avg_dx", "max_dx", "avg_dy", "max_dy", "avg_move_dist", "max_move_dist"]:
            assert (spatial_features[col] >= 0).all(), f"{col} has negative values"

    def test_lateral_ratio_non_negative(self, spatial_features):
        assert (spatial_features["lateral_ratio"] >= 0).all()

    def test_values_finite(self, spatial_features):
        assert np.isfinite(spatial_features[SPATIAL_FEATURE_COLS].values).all()

    def test_has_grade_column(self, spatial_features):
        assert "grade" in spatial_features.columns


class TestHoldUsability:
    def test_hold_scores_not_empty(self, hold_scores):
        assert len(hold_scores) > 100

    def test_hold_scores_have_coords(self, hold_scores):
        assert "x" in hold_scores.columns
        assert "y" in hold_scores.columns

    def test_usability_centered(self, hold_scores):
        # Usability is mean_grade - global_mean, so mean should be near 0
        assert abs(hold_scores["hold_usability"].mean()) < 2.0

    def test_usability_has_spread(self, hold_scores):
        assert hold_scores["hold_usability"].std() > 0.5

    def test_route_features_shape(self, usability_features, climbs):
        assert len(usability_features) == len(climbs)
        assert set(HOLD_USABILITY_FEATURE_COLS).issubset(set(usability_features.columns))

    def test_route_features_no_nans(self, usability_features):
        assert usability_features[HOLD_USABILITY_FEATURE_COLS].isna().sum().sum() == 0

    def test_route_features_finite(self, usability_features):
        assert np.isfinite(usability_features[HOLD_USABILITY_FEATURE_COLS].values).all()

    def test_pct_hard_holds_range(self, usability_features):
        assert (usability_features["pct_hard_holds"] >= 0).all()
        assert (usability_features["pct_hard_holds"] <= 1).all()
