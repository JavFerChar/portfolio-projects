"""Ingest Kilter Board SQLite database into clean pandas DataFrames."""

import re
import sqlite3
from pathlib import Path

import pandas as pd

# Kilter Board Original layout & product IDs
KILTER_LAYOUT_ID = 1
KILTER_PRODUCT_ID = 1

# Placement role IDs for Kilter Board Original (product_id=1)
ROLE_START = 12
ROLE_MIDDLE = 13
ROLE_FINISH = 14
ROLE_FOOT = 15

ROLE_MAP = {ROLE_START: "start", ROLE_MIDDLE: "middle", ROLE_FINISH: "finish", ROLE_FOOT: "foot"}

# Regex to parse frames string: "p1100r15p1103r15..." -> [(1100, 15), (1103, 15), ...]
_FRAMES_PATTERN = re.compile(r"p(\d+)r(\d+)")


def _parse_frames(frames: str) -> list[tuple[int, int]]:
    """Parse a frames string into list of (placement_id, role_id) tuples."""
    return [(int(p), int(r)) for p, r in _FRAMES_PATTERN.findall(frames)]


def load_placements(db_path: str | Path) -> pd.DataFrame:
    """Load hold placements with x/y coordinates for Kilter Board Original.

    Returns DataFrame with columns: placement_id, hole_id, x, y
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT p.id AS placement_id, p.hole_id, h.x, h.y
        FROM placements p
        JOIN holes h ON p.hole_id = h.id
        WHERE p.layout_id = ?
        """,
        conn,
        params=(KILTER_LAYOUT_ID,),
    )
    conn.close()
    return df


def load_climbs(
    db_path: str | Path,
    min_ascents: int = 5,
) -> pd.DataFrame:
    """Load climbs with grade and hold information.

    Joins climbs with climb_stats to get consensus grades per angle.
    Filters to Kilter Board Original layout and listed, non-draft routes.

    Args:
        db_path: Path to the kilter.db SQLite file.
        min_ascents: Minimum ascensionist count for a route-angle combo.

    Returns:
        DataFrame with columns: climb_uuid, name, angle, grade, ascensionist_count,
        quality_average, frames (raw), hold_placements (parsed list of dicts).
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT
            c.uuid AS climb_uuid,
            c.name,
            cs.angle,
            cs.display_difficulty AS grade,
            cs.ascensionist_count,
            cs.quality_average,
            c.frames
        FROM climbs c
        JOIN climb_stats cs ON c.uuid = cs.climb_uuid
        WHERE c.layout_id = ?
          AND c.is_listed = 1
          AND c.is_draft = 0
          AND cs.ascensionist_count >= ?
          AND cs.display_difficulty IS NOT NULL
        ORDER BY cs.ascensionist_count DESC
        """,
        conn,
        params=(KILTER_LAYOUT_ID, min_ascents),
    )
    conn.close()

    # Parse frames into structured hold placements
    placements_lookup = load_placements(db_path).set_index("placement_id")

    def _resolve_holds(frames: str) -> list[dict]:
        holds = []
        for placement_id, role_id in _parse_frames(frames):
            if placement_id in placements_lookup.index:
                row = placements_lookup.loc[placement_id]
                holds.append(
                    {
                        "placement_id": placement_id,
                        "x": int(row["x"]),
                        "y": int(row["y"]),
                        "role": ROLE_MAP.get(role_id, "unknown"),
                    }
                )
        return holds

    df["holds"] = df["frames"].apply(_resolve_holds)
    df["hold_count"] = df["holds"].apply(len)

    # Drop routes with fewer than 2 holds (need at least start + finish)
    df = df[df["hold_count"] >= 2].reset_index(drop=True)

    return df


def load_difficulty_grades(db_path: str | Path) -> pd.DataFrame:
    """Load the difficulty-to-grade mapping table.

    Returns DataFrame with columns: difficulty (int), boulder_name (str).
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT difficulty, boulder_name FROM difficulty_grades WHERE is_listed = 1",
        conn,
    )
    conn.close()
    return df
