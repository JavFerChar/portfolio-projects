"""Streamlit dashboard for Kilter Board grade prediction."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work when run via `streamlit run`
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import shap  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.ingest import load_climbs, load_placements  # noqa: E402
from src.features.hold_usability import compute_hold_usability  # noqa: E402
from src.features.spatial import _extract_one  # noqa: E402
from src.models.predict import (  # noqa: E402
    ALL_FEATURE_COLS,
    DEFAULT_DB_PATH,
    DEFAULT_MODEL_PATH,
    load_model,
    predict_grade,
)

# Kilter Board color scheme
COLOR_START = "#00FF00"
COLOR_MIDDLE = "#00BFFF"
COLOR_FINISH = "#FF00FF"
COLOR_FOOT = "#FFA500"
COLOR_UNSELECTED = "rgba(180, 180, 180, 0.4)"

ROLE_COLORS = {
    "start": COLOR_START,
    "middle": COLOR_MIDDLE,
    "finish": COLOR_FINISH,
    "foot": COLOR_FOOT,
}

BOARD_IMAGE_PATH = Path("assets/board_image.png")


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    return load_model(DEFAULT_MODEL_PATH)


@st.cache_resource
def get_hold_scores():
    climbs = load_climbs(DEFAULT_DB_PATH, min_ascents=5)
    placements = load_placements(DEFAULT_DB_PATH)
    return compute_hold_usability(climbs, placements)


@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def get_placements():
    return load_placements(DEFAULT_DB_PATH)


@st.cache_data
def get_board_image():
    return Image.open(BOARD_IMAGE_PATH)


# ---------------------------------------------------------------------------
# Board figure
# ---------------------------------------------------------------------------
def build_board_figure(
    placements,
    board_img,
    selected_indices: set[int],
    role_map: dict[int, str],
    hold_numbers: dict[int, int] | None = None,
    show_heatmap: bool = False,
    hold_scores=None,
):
    """Build Plotly figure with board image background and hold positions."""
    fig = go.Figure()

    # Background image scaled to board coordinates
    fig.add_layout_image(
        source=board_img,
        xref="x",
        yref="y",
        x=0,
        y=156,
        sizex=144,
        sizey=156,
        layer="below",
        sizing="stretch",
    )

    if show_heatmap and hold_scores is not None:
        # Heatmap layer — unselected holds colored by usability
        merged = placements.merge(
            hold_scores[["placement_id", "hold_usability"]],
            on="placement_id",
            how="left",
        )
        merged["hold_usability"] = merged["hold_usability"].fillna(0)

        unsel_mask = [i not in selected_indices for i in range(len(placements))]
        unsel_merged = merged[unsel_mask]
        fig.add_trace(
            go.Scatter(
                x=unsel_merged["x"],
                y=unsel_merged["y"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=unsel_merged["hold_usability"],
                    colorscale="RdBu_r",
                    showscale=True,
                    colorbar=dict(title="Usability"),
                    opacity=0.8,
                ),
                text=[
                    f"({row.x}, {row.y}) usability: {row.hold_usability:.2f}"
                    for _, row in unsel_merged.iterrows()
                ],
                hoverinfo="text",
                customdata=unsel_merged.index.tolist(),
            )
        )
    else:
        # Unselected holds
        unsel_mask = [i not in selected_indices for i in range(len(placements))]
        unsel = placements[unsel_mask]
        fig.add_trace(
            go.Scatter(
                x=unsel["x"],
                y=unsel["y"],
                mode="markers",
                marker=dict(size=8, color=COLOR_UNSELECTED),
                text=[f"({row.x}, {row.y})" for _, row in unsel.iterrows()],
                hoverinfo="text",
                customdata=unsel.index.tolist(),
            )
        )

    # Selected holds always on top, colored by role, with number labels
    num = hold_numbers or {}
    if selected_indices:
        for role, color in ROLE_COLORS.items():
            role_idxs = [i for i in selected_indices if role_map.get(i) == role]
            if not role_idxs:
                continue
            sel = placements.loc[role_idxs]
            labels = [str(num.get(i, "")) for i in role_idxs]
            fig.add_trace(
                go.Scatter(
                    x=sel["x"],
                    y=sel["y"],
                    mode="markers+text",
                    marker=dict(size=20, color=color, line=dict(width=2, color="white")),
                    text=labels,
                    textposition="middle center",
                    textfont=dict(size=10, color="white", family="Arial Black"),
                    hovertext=[
                        f"#{num.get(i, '?')} ({row.x}, {row.y}) [{role}]"
                        for i, (_, row) in zip(role_idxs, sel.iterrows())
                    ],
                    hoverinfo="text",
                    name=role.capitalize(),
                    customdata=sel.index.tolist(),
                )
            )

    fig.update_xaxes(range=[0, 144], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[0, 156], showgrid=False, zeroline=False, visible=False)
    fig.update_layout(
        width=540,
        height=585,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def _auto_assign_roles(indices: set[int], placements, role_map: dict[int, str]):
    """Auto-assign roles: first hold = start, rest = middle. User sets finish/foot manually."""
    if not indices:
        return
    has_start = any(v == "start" for v in role_map.values())
    for idx in indices:
        if idx not in role_map:
            if not has_start:
                role_map[idx] = "start"
                has_start = True
            else:
                role_map[idx] = "middle"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Kilter Grade Predictor", layout="wide")
    st.title("Kilter Board Grade Predictor")

    # Load data
    model = get_model()
    hold_scores = get_hold_scores()
    placements = get_placements()
    board_img = get_board_image()

    # Session state for selected holds
    if "selected_indices" not in st.session_state:
        st.session_state.selected_indices = set()
    if "role_map" not in st.session_state:
        st.session_state.role_map = {}

    # Sidebar controls
    st.sidebar.header("Controls")
    angle = st.sidebar.slider("Angle (°)", 0, 70, 40, step=5)
    show_heatmap = st.sidebar.checkbox("Show hold usability heatmap")

    if st.sidebar.button("Clear selection"):
        st.session_state.selected_indices = set()
        st.session_state.role_map = {}

    # Compute stable numbering for selected holds (sorted by y, bottom-to-top)
    sorted_selected = sorted(
        st.session_state.selected_indices,
        key=lambda i: placements.loc[i, "y"],
    )
    hold_numbers = {idx: n + 1 for n, idx in enumerate(sorted_selected)}

    # Layout
    col_board, col_controls = st.columns([2, 1])

    with col_board:
        st.subheader("Select holds on the board")
        if show_heatmap:
            st.caption("Heatmap mode — blue = easy holds, red = hard holds")

        fig = build_board_figure(
            placements,
            board_img,
            st.session_state.selected_indices,
            st.session_state.role_map,
            hold_numbers=hold_numbers,
            show_heatmap=show_heatmap,
            hold_scores=hold_scores,
        )

        event = st.plotly_chart(
            fig,
            key="board",
            on_select="rerun",
        )

        # Process selection events — accumulate into session state
        if event and event.selection and event.selection.point_indices:
            clicked_indices = set()
            for pt in event.selection.points:
                cd = (
                    pt.get("customdata")
                    if isinstance(pt, dict)
                    else getattr(pt, "customdata", None)
                )
                if cd is not None:
                    clicked_indices.add(int(cd))

            if clicked_indices:
                # Toggle: if already selected, deselect; otherwise add
                existing = st.session_state.selected_indices
                for idx in clicked_indices:
                    if idx in existing:
                        existing.discard(idx)
                        st.session_state.role_map.pop(idx, None)
                    else:
                        existing.add(idx)

                # Auto-assign roles for newly added holds
                _auto_assign_roles(existing, placements, st.session_state.role_map)
                st.rerun()

    with col_controls:
        selected = st.session_state.selected_indices
        role_map = st.session_state.role_map

        st.subheader("Selected holds")

        if not selected:
            st.info("Use box select or lasso on the board to pick holds.")
        else:
            st.write(f"**{len(selected)} holds selected**")

            # Role assignment per hold with numbers and remove buttons
            role_icons = {
                "start": "🟢",
                "middle": "🔵",
                "finish": "🟣",
                "foot": "🟠",
            }
            roles_changed = False
            to_remove = None
            for idx in sorted_selected:
                row = placements.loc[idx]
                n = hold_numbers[idx]
                current_role = role_map.get(idx, "middle")
                icon = role_icons.get(current_role, "⚪")

                c_select, c_remove = st.columns([4, 1])
                with c_select:
                    new_role = st.selectbox(
                        f"#{n} {icon} ({int(row.x)}, {int(row.y)})",
                        ["start", "middle", "finish", "foot"],
                        index=["start", "middle", "finish", "foot"].index(current_role),
                        key=f"role_{idx}",
                    )
                    if new_role != current_role:
                        st.session_state.role_map[idx] = new_role
                        roles_changed = True
                with c_remove:
                    st.write("")  # spacing to align with selectbox
                    if st.button("✕", key=f"rm_{idx}"):
                        to_remove = idx

            if to_remove is not None:
                st.session_state.selected_indices.discard(to_remove)
                st.session_state.role_map.pop(to_remove, None)
                st.rerun()
            if roles_changed:
                st.rerun()

            # Predict
            st.divider()
            if st.button("🔮 Predict Grade", type="primary", use_container_width=True):
                holds = [
                    {
                        "x": int(placements.loc[i, "x"]),
                        "y": int(placements.loc[i, "y"]),
                        "role": role_map.get(i, "middle"),
                    }
                    for i in sorted_selected
                ]

                if len(holds) < 2:
                    st.error("Select at least 2 holds.")
                else:
                    result = predict_grade(holds, angle, model=model, hold_scores=hold_scores)

                    st.metric(
                        "Predicted Grade",
                        result["v_grade"],
                        f"({result['predicted_grade']:.1f})",
                    )

                    # SHAP explanation
                    with st.expander("SHAP Explanation", expanded=True):
                        _show_shap(holds, angle, model, hold_scores)


def _show_shap(holds, angle, model, hold_scores):
    """Compute and display SHAP waterfall for a single prediction."""
    import matplotlib.pyplot as plt

    from src.features.hold_usability import HOLD_USABILITY_FEATURE_COLS

    explainer = get_shap_explainer(model)

    # Build feature vector (same logic as predict_grade)
    spatial = _extract_one(holds, angle)

    scores_lookup = hold_scores.set_index("placement_id")
    valid_pids = set(scores_lookup.index)

    placements = get_placements()
    coord_to_pid = {
        (int(r["x"]), int(r["y"])): int(r["placement_id"]) for _, r in placements.iterrows()
    }

    pids = [coord_to_pid.get((h["x"], h["y"])) for h in holds]
    pids = [p for p in pids if p is not None and p in valid_pids]

    if pids:
        u_scores = np.array([float(scores_lookup.loc[pid, "hold_usability"]) for pid in pids])
        a_scores = np.array(
            [float(scores_lookup.loc[pid, "hold_angle_sensitivity"]) for pid in pids]
        )
        hard_threshold = scores_lookup["hold_usability"].quantile(0.75)
        usability_feats = {
            "avg_hold_usability": float(np.mean(u_scores)),
            "min_hold_usability": float(np.min(u_scores)),
            "max_hold_usability": float(np.max(u_scores)),
            "hold_usability_range": float(np.max(u_scores) - np.min(u_scores)),
            "avg_angle_sensitivity": float(np.mean(a_scores)),
            "pct_hard_holds": float(np.mean(u_scores > hard_threshold)),
        }
    else:
        usability_feats = {col: 0.0 for col in HOLD_USABILITY_FEATURE_COLS}

    feature_vector = {**spatial, **usability_feats}
    X = np.array([[feature_vector[col] for col in ALL_FEATURE_COLS]])

    shap_values = explainer(X)
    shap_values.feature_names = ALL_FEATURE_COLS

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


if __name__ == "__main__":
    main()
