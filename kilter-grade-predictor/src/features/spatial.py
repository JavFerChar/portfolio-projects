"""Spatial and displacement feature extraction for Kilter Board routes."""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from src.data.ingest import _FRAMES_PATTERN, ROLE_MAP


def _holds_from_frames(frames: str, placements_lookup: pd.DataFrame) -> list[dict]:
    """Parse a frames string into a list of hold dicts with x, y, role."""
    holds = []
    for p_str, r_str in _FRAMES_PATTERN.findall(frames):
        pid = int(p_str)
        if pid in placements_lookup.index:
            row = placements_lookup.loc[pid]
            holds.append(
                {
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                    "role": ROLE_MAP.get(int(r_str), "unknown"),
                }
            )
    return holds


def _extract_one(holds: list[dict], angle: int) -> dict:
    """Extract spatial and displacement features for a single route."""
    xs = np.array([h["x"] for h in holds], dtype=float)
    ys = np.array([h["y"] for h in holds], dtype=float)
    roles = [h["role"] for h in holds]

    # Sort by y (bottom-to-top) to approximate climbing order
    order = np.argsort(ys)
    xs_sorted = xs[order]
    ys_sorted = ys[order]

    n = len(holds)

    # --- Basic spatial ---
    start_mask = [r == "start" for r in roles]
    finish_mask = [r == "finish" for r in roles]

    start_x = float(np.mean(xs[start_mask])) if any(start_mask) else float(xs_sorted[0])
    start_y = float(np.mean(ys[start_mask])) if any(start_mask) else float(ys_sorted[0])
    finish_x = float(np.mean(xs[finish_mask])) if any(finish_mask) else float(xs_sorted[-1])
    finish_y = float(np.mean(ys[finish_mask])) if any(finish_mask) else float(ys_sorted[-1])

    x_range = float(xs.max() - xs.min())
    y_range = float(ys.max() - ys.min())

    # Convex hull area (need ≥3 non-collinear points)
    hull_area = 0.0
    if n >= 3:
        points = np.column_stack([xs, ys])
        try:
            hull_area = float(ConvexHull(points).volume)  # 2D: volume = area
        except Exception:
            hull_area = 0.0

    hold_density = n / hull_area if hull_area > 0 else 0.0

    # --- Displacement features (consecutive moves in climbing order) ---
    dx = np.diff(xs_sorted)
    dy = np.diff(ys_sorted)
    dists = np.sqrt(dx**2 + dy**2)

    avg_dx = float(np.mean(np.abs(dx))) if len(dx) > 0 else 0.0
    max_dx = float(np.max(np.abs(dx))) if len(dx) > 0 else 0.0
    avg_dy = float(np.mean(np.abs(dy))) if len(dy) > 0 else 0.0
    max_dy = float(np.max(np.abs(dy))) if len(dy) > 0 else 0.0
    avg_move_dist = float(np.mean(dists)) if len(dists) > 0 else 0.0
    max_move_dist = float(np.max(dists)) if len(dists) > 0 else 0.0
    total_path_length = float(np.sum(dists))

    sum_abs_dx = float(np.sum(np.abs(dx)))
    sum_abs_dy = float(np.sum(np.abs(dy)))
    lateral_ratio = sum_abs_dx / sum_abs_dy if sum_abs_dy > 0 else 0.0

    # Direction changes
    direction_changes_x = int(np.sum(np.diff(np.sign(dx)) != 0)) if len(dx) > 1 else 0
    direction_changes_y = int(np.sum(np.diff(np.sign(dy)) != 0)) if len(dy) > 1 else 0

    # --- Angle interaction features ---
    angle_f = float(angle)

    return {
        "hold_count": n,
        "start_x": start_x,
        "start_y": start_y,
        "finish_x": finish_x,
        "finish_y": finish_y,
        "x_range": x_range,
        "y_range": y_range,
        "convex_hull_area": hull_area,
        "hold_density": hold_density,
        "angle": angle_f,
        "avg_dx": avg_dx,
        "max_dx": max_dx,
        "avg_dy": avg_dy,
        "max_dy": max_dy,
        "avg_move_dist": avg_move_dist,
        "max_move_dist": max_move_dist,
        "total_path_length": total_path_length,
        "lateral_ratio": lateral_ratio,
        "direction_changes_x": direction_changes_x,
        "direction_changes_y": direction_changes_y,
        "angle_x_avg_dx": angle_f * avg_dx,
        "angle_x_max_move": angle_f * max_move_dist,
        "angle_x_hold_count": angle_f * n,
    }


SPATIAL_FEATURE_COLS = [
    "hold_count",
    "start_x",
    "start_y",
    "finish_x",
    "finish_y",
    "x_range",
    "y_range",
    "convex_hull_area",
    "hold_density",
    "angle",
    "avg_dx",
    "max_dx",
    "avg_dy",
    "max_dy",
    "avg_move_dist",
    "max_move_dist",
    "total_path_length",
    "lateral_ratio",
    "direction_changes_x",
    "direction_changes_y",
    "angle_x_avg_dx",
    "angle_x_max_move",
    "angle_x_hold_count",
]


def extract_spatial_features(
    climbs_df: pd.DataFrame,
    placements_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract spatial and displacement features for all routes.

    Args:
        climbs_df: DataFrame from load_climbs() with 'frames', 'angle', 'grade' columns.
        placements_df: DataFrame from load_placements() with placement_id, x, y.

    Returns:
        DataFrame with one row per route: all spatial features + grade as target.
    """
    lookup = placements_df.set_index("placement_id")

    records = []
    for _, row in climbs_df.iterrows():
        holds = _holds_from_frames(row["frames"], lookup)
        if len(holds) < 2:
            continue
        feats = _extract_one(holds, row["angle"])
        feats["grade"] = row["grade"]
        feats["climb_uuid"] = row["climb_uuid"]
        records.append(feats)

    return pd.DataFrame(records)
