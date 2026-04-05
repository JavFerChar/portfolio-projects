"""Derived per-hold usability scores from route usage patterns.

Novel feature: no hold-type labels exist in the database, so we infer hold
"quality" from the grades and angles of routes each hold appears in.
A good hold (jug-like) appears in easy routes even at steep angles.
A bad hold (crimp-like) only appears in hard routes.
"""

import numpy as np
import pandas as pd

from src.data.ingest import _FRAMES_PATTERN

MIN_HOLD_USAGE = 20  # minimum routes a hold must appear in for stable estimates


def compute_hold_usability(
    climbs_df: pd.DataFrame,
    placements_df: pd.DataFrame,
    min_usage: int = MIN_HOLD_USAGE,
) -> pd.DataFrame:
    """Compute per-hold usability scores from route usage patterns.

    For each hold, computes:
    - mean_grade: weighted mean grade of routes containing this hold
    - usability: mean_grade - global_mean_grade (positive = hard, negative = easy)
    - angle_sensitivity: correlation between grade and angle for routes with this hold

    Args:
        climbs_df: DataFrame from load_climbs() with 'frames', 'grade', 'angle',
            'ascensionist_count' columns.
        placements_df: DataFrame from load_placements() with placement_id, x, y.

    Returns:
        DataFrame indexed by placement_id with usability metrics.
    """
    placement_ids = set(placements_df["placement_id"])
    global_mean_grade = climbs_df["grade"].mean()

    # Build hold → list of (grade, angle, ascent_count) from all routes
    hold_records: dict[int, list[tuple[float, int, int]]] = {}

    for _, row in climbs_df.iterrows():
        grade = row["grade"]
        angle = row["angle"]
        ascents = row["ascensionist_count"]
        for p_str, _ in _FRAMES_PATTERN.findall(row["frames"]):
            pid = int(p_str)
            if pid in placement_ids:
                if pid not in hold_records:
                    hold_records[pid] = []
                hold_records[pid].append((grade, angle, ascents))

    # Compute per-hold metrics
    records = []
    for pid, entries in hold_records.items():
        if len(entries) < min_usage:
            continue
        grades = np.array([e[0] for e in entries])
        angles = np.array([e[1] for e in entries])
        ascents = np.array([e[2] for e in entries])

        # Weighted mean grade (weighted by ascent count for consensus stability)
        weights = ascents / ascents.sum()
        wmean_grade = float(np.average(grades, weights=weights))

        # Angle sensitivity: how much does grade increase with angle for this hold?
        angle_sensitivity = 0.0
        if np.std(angles) > 0 and np.std(grades) > 0:
            angle_sensitivity = float(np.corrcoef(grades, angles)[0, 1])

        records.append(
            {
                "placement_id": pid,
                "hold_mean_grade": wmean_grade,
                "hold_usability": wmean_grade - global_mean_grade,
                "hold_angle_sensitivity": angle_sensitivity,
                "hold_usage_count": len(entries),
            }
        )

    hold_scores = pd.DataFrame(records)

    # Merge with placement coordinates
    hold_scores = hold_scores.merge(placements_df, on="placement_id", how="left")

    return hold_scores


def aggregate_hold_usability_features(
    climbs_df: pd.DataFrame,
    hold_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-hold usability scores into route-level features.

    For each route, summarises the usability scores of its holds into:
    avg, min, max, range, pct_hard, and avg angle sensitivity.

    Args:
        climbs_df: DataFrame with 'frames' and 'climb_uuid' columns.
        hold_scores: DataFrame from compute_hold_usability(), indexed by placement_id.

    Returns:
        DataFrame with climb_uuid and 6 hold-usability features.
    """
    scores_lookup = hold_scores.set_index("placement_id")["hold_usability"]
    sensitivity_lookup = hold_scores.set_index("placement_id")["hold_angle_sensitivity"]
    valid_pids = set(scores_lookup.index)

    # Threshold for "hard" holds: top quartile of usability scores
    hard_threshold = scores_lookup.quantile(0.75)

    records = []
    for _, row in climbs_df.iterrows():
        pids = [int(p) for p, _ in _FRAMES_PATTERN.findall(row["frames"]) if int(p) in valid_pids]
        if not pids:
            records.append(
                {
                    "climb_uuid": row["climb_uuid"],
                    "avg_hold_usability": np.nan,
                    "min_hold_usability": np.nan,
                    "max_hold_usability": np.nan,
                    "hold_usability_range": np.nan,
                    "avg_angle_sensitivity": np.nan,
                    "pct_hard_holds": np.nan,
                }
            )
            continue

        u_scores = np.array([scores_lookup[pid] for pid in pids])
        a_scores = np.array([sensitivity_lookup[pid] for pid in pids])

        records.append(
            {
                "climb_uuid": row["climb_uuid"],
                "avg_hold_usability": float(np.mean(u_scores)),
                "min_hold_usability": float(np.min(u_scores)),
                "max_hold_usability": float(np.max(u_scores)),
                "hold_usability_range": float(np.max(u_scores) - np.min(u_scores)),
                "avg_angle_sensitivity": float(np.mean(a_scores)),
                "pct_hard_holds": float(np.mean(u_scores > hard_threshold)),
            }
        )

    return pd.DataFrame(records)


HOLD_USABILITY_FEATURE_COLS = [
    "avg_hold_usability",
    "min_hold_usability",
    "max_hold_usability",
    "hold_usability_range",
    "avg_angle_sensitivity",
    "pct_hard_holds",
]
