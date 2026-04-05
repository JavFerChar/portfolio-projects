"""Tests for data ingestion module."""

from pathlib import Path

import pytest

from src.data.ingest import (
    BOARD_X_MAX,
    BOARD_X_MIN,
    BOARD_Y_MAX,
    BOARD_Y_MIN,
    ROLE_MAP,
    load_climbs,
    load_difficulty_grades,
    load_placements,
)

DB_PATH = Path("data/raw/kilter.db")

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="kilter.db not downloaded")


@pytest.fixture(scope="module")
def climbs():
    return load_climbs(DB_PATH, min_ascents=5)


@pytest.fixture(scope="module")
def placements():
    return load_placements(DB_PATH)


@pytest.fixture(scope="module")
def grades():
    return load_difficulty_grades(DB_PATH)


class TestLoadClimbs:
    def test_expected_columns(self, climbs):
        expected = {
            "climb_uuid",
            "name",
            "angle",
            "grade",
            "ascensionist_count",
            "quality_average",
            "frames",
            "holds",
            "hold_count",
        }
        assert set(climbs.columns) == expected

    def test_not_empty(self, climbs):
        assert len(climbs) > 1000

    def test_grade_range(self, climbs):
        assert climbs["grade"].min() >= 10
        assert climbs["grade"].max() <= 33

    def test_no_null_grades(self, climbs):
        assert climbs["grade"].notna().all()

    def test_no_null_angles(self, climbs):
        assert climbs["angle"].notna().all()

    def test_angle_range(self, climbs):
        assert climbs["angle"].min() >= 0
        assert climbs["angle"].max() <= 70

    def test_min_ascents_applied(self, climbs):
        assert (climbs["ascensionist_count"] >= 5).all()

    def test_min_hold_count(self, climbs):
        assert (climbs["hold_count"] >= 2).all()

    def test_holds_structure(self, climbs):
        sample = climbs["holds"].iloc[0]
        assert isinstance(sample, list)
        assert len(sample) >= 2
        hold = sample[0]
        assert "x" in hold
        assert "y" in hold
        assert "role" in hold
        assert hold["role"] in ROLE_MAP.values()


class TestLoadPlacements:
    def test_expected_columns(self, placements):
        assert set(placements.columns) == {"placement_id", "hole_id", "x", "y"}

    def test_not_empty(self, placements):
        assert len(placements) > 100

    def test_coords_are_numeric(self, placements):
        assert placements["x"].dtype in ("int64", "float64")
        assert placements["y"].dtype in ("int64", "float64")

    def test_coords_within_board_bounds(self, placements):
        assert placements["x"].min() >= BOARD_X_MIN
        assert placements["x"].max() <= BOARD_X_MAX
        assert placements["y"].min() >= BOARD_Y_MIN
        assert placements["y"].max() <= BOARD_Y_MAX


class TestLoadDifficultyGrades:
    def test_expected_columns(self, grades):
        assert set(grades.columns) == {"difficulty", "boulder_name"}

    def test_not_empty(self, grades):
        assert len(grades) > 10

    def test_difficulty_range(self, grades):
        assert grades["difficulty"].min() >= 10
        assert grades["difficulty"].max() <= 33
