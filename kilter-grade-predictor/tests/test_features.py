"""Tests for feature engineering modules."""

from pathlib import Path

import numpy as np
import pytest

from src.data.ingest import load_climbs, load_placements
from src.features.hold_usability import (
    BASE_USABILITY_COLS,
    ROLE_TYPICAL_GRADE_COLS,
    aggregate_angle_conditioned_features,
    aggregate_hold_usability_features,
    aggregate_role_typical_grade_features,
    compute_hold_usability,
    compute_hold_usability_by_angle,
    compute_role_typical_grade,
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
def hold_scores_by_angle(placements):
    full_climbs = load_climbs(DB_PATH, min_ascents=5)
    return compute_hold_usability_by_angle(full_climbs, placements)


@pytest.fixture(scope="module")
def usability_features(climbs, hold_scores):
    return aggregate_hold_usability_features(climbs, hold_scores)


@pytest.fixture(scope="module")
def angle_features(climbs, hold_scores_by_angle):
    return aggregate_angle_conditioned_features(climbs, hold_scores_by_angle)


@pytest.fixture(scope="module")
def role_scores(placements):
    full_climbs = load_climbs(DB_PATH, min_ascents=5)
    return compute_role_typical_grade(full_climbs, placements)


@pytest.fixture(scope="module")
def role_features(climbs, role_scores):
    return aggregate_role_typical_grade_features(climbs, role_scores)


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
        assert set(BASE_USABILITY_COLS).issubset(set(usability_features.columns))

    def test_route_features_no_nans(self, usability_features):
        assert usability_features[BASE_USABILITY_COLS].isna().sum().sum() == 0

    def test_route_features_finite(self, usability_features):
        assert np.isfinite(usability_features[BASE_USABILITY_COLS].values).all()

    def test_pct_hard_holds_range(self, usability_features):
        assert (usability_features["pct_hard_holds"] >= 0).all()
        assert (usability_features["pct_hard_holds"] <= 1).all()

    def test_angle_conditioned_shape(self, angle_features, climbs):
        assert len(angle_features) == len(climbs)
        for col in [
            "avg_hold_usability_at_angle",
            "min_hold_usability_at_angle",
            "max_hold_usability_at_angle",
        ]:
            assert col in angle_features.columns

    def test_angle_conditioned_no_nans(self, angle_features):
        cols = [
            "avg_hold_usability_at_angle",
            "min_hold_usability_at_angle",
            "max_hold_usability_at_angle",
        ]
        assert angle_features[cols].isna().sum().sum() == 0

    def test_angle_conditioned_finite(self, angle_features):
        cols = [
            "avg_hold_usability_at_angle",
            "min_hold_usability_at_angle",
            "max_hold_usability_at_angle",
        ]
        assert np.isfinite(angle_features[cols].values).all()

    def test_hold_scores_by_angle_has_bins(self, hold_scores_by_angle):
        for col in ["hold_usability_low", "hold_usability_mid", "hold_usability_steep"]:
            assert col in hold_scores_by_angle.columns

    def test_role_scores_has_columns(self, role_scores):
        for label in ["low", "mid", "steep"]:
            assert f"start_typical_grade_{label}" in role_scores.columns
            assert f"finish_typical_grade_{label}" in role_scores.columns

    def test_role_scores_not_empty(self, role_scores):
        assert len(role_scores) > 50

    def test_role_features_shape(self, role_features, climbs):
        assert len(role_features) == len(climbs)
        for col in ROLE_TYPICAL_GRADE_COLS:
            assert col in role_features.columns

    def test_role_features_grade_range(self, role_features):
        """Typical grades should be within the valid grade range when present."""
        for col in ROLE_TYPICAL_GRADE_COLS:
            valid = role_features[col].dropna()
            if len(valid) > 0:
                assert valid.min() >= 0, f"{col} has negative grades"
                assert valid.max() <= 40, f"{col} has grades > 40"
