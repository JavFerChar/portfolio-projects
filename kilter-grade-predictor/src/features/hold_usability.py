"""Derived per-hold usability scores from route usage patterns.

Novel feature: no hold-type labels exist in the database, so we infer hold
"quality" from the grades and angles of routes each hold appears in.
A good hold (jug-like) appears in easy routes even at steep angles.
A bad hold (crimp-like) only appears in hard routes.

Only hand-role appearances (start, middle, finish) contribute to usability
scores. Foot-only appearances are excluded since they don't reflect grip
difficulty.
"""

import numpy as np
import pandas as pd

from src.data.ingest import _FRAMES_PATTERN

MIN_HOLD_USAGE = 20  # minimum routes a hold must appear in for stable estimates

# Roles that involve gripping with hands
HAND_ROLES = {12, 13, 14}  # start, middle, finish

# Angle bins for conditioned usability: (lower_bound_inclusive, upper_bound_exclusive)
ANGLE_BINS = [(0, 20), (20, 35), (35, 71)]
ANGLE_BIN_LABELS = ["low", "mid", "steep"]


def _angle_to_bin_label(angle: int) -> str:
    """Map an angle to its bin label."""
    for (lo, hi), label in zip(ANGLE_BINS, ANGLE_BIN_LABELS):
        if lo <= angle < hi:
            return label
    return ANGLE_BIN_LABELS[-1]  # fallback to steep for angles >= 71


def _collect_hand_records(
    climbs_df: pd.DataFrame,
    placement_ids: set[int],
) -> dict[int, list[tuple[float, int, int]]]:
    """Build hold → list of (grade, angle, ascent_count) from hand-role appearances."""
    hand_records: dict[int, list[tuple[float, int, int]]] = {}

    for _, row in climbs_df.iterrows():
        grade = row["grade"]
        angle = row["angle"]
        ascents = row["ascensionist_count"]
        for p_str, r_str in _FRAMES_PATTERN.findall(row["frames"]):
            pid = int(p_str)
            role_id = int(r_str)
            if pid in placement_ids and role_id in HAND_ROLES:
                hand_records.setdefault(pid, []).append((grade, angle, ascents))

    return hand_records


def compute_hold_usability(
    climbs_df: pd.DataFrame,
    placements_df: pd.DataFrame,
    min_usage: int = MIN_HOLD_USAGE,
) -> pd.DataFrame:
    """Compute per-hold usability scores from hand-role route usage patterns.

    For each hold, computes:
    - mean_grade: weighted mean grade of routes containing this hold (hand roles only)
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

    hand_records = _collect_hand_records(climbs_df, placement_ids)

    # Compute per-hold metrics
    records = []
    for pid, entries in hand_records.items():
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


def compute_hold_usability_by_angle(
    climbs_df: pd.DataFrame,
    placements_df: pd.DataFrame,
    min_usage: int = MIN_HOLD_USAGE,
) -> pd.DataFrame:
    """Compute per-hold usability conditioned on angle bin.

    Partitions hand-role appearances into angle bins (low/mid/steep) and computes
    usability relative to each bin's mean grade. Falls back to global usability
    when a hold has insufficient data in a bin.

    Returns:
        DataFrame with columns: placement_id, hold_usability_low,
        hold_usability_mid, hold_usability_steep.
    """
    placement_ids = set(placements_df["placement_id"])
    hand_records = _collect_hand_records(climbs_df, placement_ids)

    # Compute global usability as fallback
    global_scores = compute_hold_usability(climbs_df, placements_df, min_usage)
    global_lookup = global_scores.set_index("placement_id")["hold_usability"]

    # Compute per-bin mean grades
    bin_mean_grades = {}
    for label in ANGLE_BIN_LABELS:
        bin_mean_grades[label] = climbs_df["grade"].mean()  # init with global
    for (lo, hi), label in zip(ANGLE_BINS, ANGLE_BIN_LABELS):
        mask = (climbs_df["angle"] >= lo) & (climbs_df["angle"] < hi)
        if mask.any():
            bin_mean_grades[label] = float(climbs_df.loc[mask, "grade"].mean())

    # Partition each hold's entries by angle bin
    records = []
    for pid, entries in hand_records.items():
        if len(entries) < min_usage:
            continue

        binned: dict[str, list[tuple[float, int]]] = {lbl: [] for lbl in ANGLE_BIN_LABELS}
        for grade, angle, ascents in entries:
            label = _angle_to_bin_label(angle)
            binned[label].append((grade, ascents))

        row = {"placement_id": pid}
        global_fallback = float(global_lookup.get(pid, 0.0))

        for label in ANGLE_BIN_LABELS:
            col = f"hold_usability_{label}"
            bin_entries = binned[label]
            if len(bin_entries) >= min_usage:
                grades = np.array([e[0] for e in bin_entries])
                ascents = np.array([e[1] for e in bin_entries])
                weights = ascents / ascents.sum()
                wmean = float(np.average(grades, weights=weights))
                row[col] = wmean - bin_mean_grades[label]
            else:
                row[col] = global_fallback

        records.append(row)

    return pd.DataFrame(records)


def aggregate_hold_usability_features(
    climbs_df: pd.DataFrame,
    hold_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-hold usability scores into route-level features.

    For each route, summarises the usability scores of its holds into:
    avg, min, max, range, pct_hard, and avg angle sensitivity.
    Only hand-role appearances contribute.

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
        # Only consider hand-role holds
        pids = [
            int(p)
            for p, r in _FRAMES_PATTERN.findall(row["frames"])
            if int(p) in valid_pids and int(r) in HAND_ROLES
        ]
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


def aggregate_angle_conditioned_features(
    climbs_df: pd.DataFrame,
    hold_scores_by_angle: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate angle-conditioned hold scores into route-level features.

    For each route, selects the usability column matching the route's angle bin,
    then computes avg, min, max of the matched usability values for hand-role holds.

    Args:
        climbs_df: DataFrame with 'frames', 'angle', and 'climb_uuid' columns.
        hold_scores_by_angle: DataFrame from compute_hold_usability_by_angle().

    Returns:
        DataFrame with climb_uuid and 3 angle-conditioned features.
    """
    angle_lookup = hold_scores_by_angle.set_index("placement_id")
    valid_pids = set(angle_lookup.index)

    records = []
    for _, row in climbs_df.iterrows():
        pids = [
            int(p)
            for p, r in _FRAMES_PATTERN.findall(row["frames"])
            if int(p) in valid_pids and int(r) in HAND_ROLES
        ]
        bin_label = _angle_to_bin_label(row["angle"])
        col = f"hold_usability_{bin_label}"

        if not pids:
            records.append(
                {
                    "climb_uuid": row["climb_uuid"],
                    "avg_hold_usability_at_angle": np.nan,
                    "min_hold_usability_at_angle": np.nan,
                    "max_hold_usability_at_angle": np.nan,
                }
            )
            continue

        scores = np.array([float(angle_lookup.loc[pid, col]) for pid in pids])
        records.append(
            {
                "climb_uuid": row["climb_uuid"],
                "avg_hold_usability_at_angle": float(np.mean(scores)),
                "min_hold_usability_at_angle": float(np.min(scores)),
                "max_hold_usability_at_angle": float(np.max(scores)),
            }
        )

    return pd.DataFrame(records)


ROLE_START = 12
ROLE_FINISH = 14


def compute_role_typical_grade(
    climbs_df: pd.DataFrame,
    placements_df: pd.DataFrame,
    min_usage: int = 5,
) -> pd.DataFrame:
    """Compute typical grade for each hold when used as start or finish, per angle bin.

    For each hold and role (start/finish), computes the weighted mean grade of
    routes using that hold in that role, partitioned by angle bin. Falls back to
    the all-angle typical grade when a bin has insufficient data.

    Args:
        climbs_df: DataFrame with 'frames', 'grade', 'angle', 'ascensionist_count'.
        placements_df: DataFrame with placement_id, x, y.
        min_usage: Minimum appearances in a role+bin for a stable estimate.

    Returns:
        DataFrame with columns: placement_id, start_typical_grade_low/mid/steep,
        finish_typical_grade_low/mid/steep.
    """
    placement_ids = set(placements_df["placement_id"])

    # Collect per-hold, per-role records: {(pid, role): [(grade, angle, ascents), ...]}
    role_records: dict[tuple[int, int], list[tuple[float, int, int]]] = {}

    for _, row in climbs_df.iterrows():
        grade = row["grade"]
        angle = row["angle"]
        ascents = row["ascensionist_count"]
        for p_str, r_str in _FRAMES_PATTERN.findall(row["frames"]):
            pid = int(p_str)
            role_id = int(r_str)
            if pid in placement_ids and role_id in (ROLE_START, ROLE_FINISH):
                role_records.setdefault((pid, role_id), []).append((grade, angle, ascents))

    # Compute all-angle typical grade per (pid, role) as fallback
    global_typical: dict[tuple[int, int], float] = {}
    for (pid, role_id), entries in role_records.items():
        if len(entries) >= min_usage:
            grades = np.array([e[0] for e in entries])
            ascents = np.array([e[2] for e in entries])
            weights = ascents / ascents.sum()
            global_typical[(pid, role_id)] = float(np.average(grades, weights=weights))

    # Collect unique pids that have at least one role with enough data
    valid_pids = {pid for (pid, _) in global_typical}

    records = []
    for pid in valid_pids:
        row = {"placement_id": pid}

        for role_id, role_name in [(ROLE_START, "start"), (ROLE_FINISH, "finish")]:
            entries = role_records.get((pid, role_id), [])
            fallback = global_typical.get((pid, role_id), np.nan)

            # Partition by angle bin
            binned: dict[str, list[tuple[float, int]]] = {lbl: [] for lbl in ANGLE_BIN_LABELS}
            for grade, angle, ascents in entries:
                label = _angle_to_bin_label(angle)
                binned[label].append((grade, ascents))

            for label in ANGLE_BIN_LABELS:
                col = f"{role_name}_typical_grade_{label}"
                bin_entries = binned[label]
                if len(bin_entries) >= min_usage:
                    grades = np.array([e[0] for e in bin_entries])
                    ascents = np.array([e[1] for e in bin_entries])
                    weights = ascents / ascents.sum()
                    row[col] = float(np.average(grades, weights=weights))
                else:
                    row[col] = fallback

        records.append(row)

    return pd.DataFrame(records)


def aggregate_role_typical_grade_features(
    climbs_df: pd.DataFrame,
    role_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate role-specific typical grades into route-level features.

    For each route, looks up the start/finish holds' typical grade at the
    route's angle bin. When multiple start or finish holds exist, averages them.

    Returns:
        DataFrame with climb_uuid, start_hold_typical_grade, finish_hold_typical_grade.
    """
    role_lookup = role_scores.set_index("placement_id")
    valid_pids = set(role_lookup.index)

    records = []
    for _, row in climbs_df.iterrows():
        bin_label = _angle_to_bin_label(row["angle"])

        start_pids = []
        finish_pids = []
        for p_str, r_str in _FRAMES_PATTERN.findall(row["frames"]):
            pid = int(p_str)
            role_id = int(r_str)
            if pid not in valid_pids:
                continue
            if role_id == ROLE_START:
                start_pids.append(pid)
            elif role_id == ROLE_FINISH:
                finish_pids.append(pid)

        start_col = f"start_typical_grade_{bin_label}"
        finish_col = f"finish_typical_grade_{bin_label}"

        start_val = np.nan
        if start_pids:
            vals = [
                float(role_lookup.loc[pid, start_col])
                for pid in start_pids
                if not np.isnan(role_lookup.loc[pid, start_col])
            ]
            if vals:
                start_val = float(np.mean(vals))

        finish_val = np.nan
        if finish_pids:
            vals = [
                float(role_lookup.loc[pid, finish_col])
                for pid in finish_pids
                if not np.isnan(role_lookup.loc[pid, finish_col])
            ]
            if vals:
                finish_val = float(np.mean(vals))

        records.append(
            {
                "climb_uuid": row["climb_uuid"],
                "start_hold_typical_grade": start_val,
                "finish_hold_typical_grade": finish_val,
            }
        )

    return pd.DataFrame(records)


BASE_USABILITY_COLS = [
    "avg_hold_usability",
    "min_hold_usability",
    "max_hold_usability",
    "hold_usability_range",
    "avg_angle_sensitivity",
    "pct_hard_holds",
]

ANGLE_CONDITIONED_COLS = [
    "avg_hold_usability_at_angle",
    "min_hold_usability_at_angle",
    "max_hold_usability_at_angle",
]

ROLE_TYPICAL_GRADE_COLS = [
    "start_hold_typical_grade",
    "finish_hold_typical_grade",
]

HOLD_USABILITY_FEATURE_COLS = (
    BASE_USABILITY_COLS + ANGLE_CONDITIONED_COLS + ROLE_TYPICAL_GRADE_COLS
)
