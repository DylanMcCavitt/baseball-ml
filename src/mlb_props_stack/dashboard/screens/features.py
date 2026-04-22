"""Feature inspection screen renderer."""

from __future__ import annotations

from html import escape

import pandas as pd

from ..lib.data import DashboardSettings, feature_drift_rows, get_feature_importance, leakage_checks
from ..lib.plots import feature_bars_fig


def _drift_html(rows: list[dict]) -> str:
    if not rows:
        return "<div class='strike-empty'>Not enough feature history was found to compute PSI.</div>"
    parts = ["<div class='strike-mini-bars'>"]
    for row in rows:
        status = str(row["status"])
        psi_label = "n/a" if row["psi"] is None else f"{float(row['psi']):.2f}"
        parts.append(
            "<div class='strike-statusbar'>"
            f"<span>{escape(str(row['name']))}</span>"
            f"<span class='strike-pill {'warn' if status == 'warn' else 'ok'}'>{escape(psi_label)}</span>"
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _leakage_html(rows: list[dict]) -> str:
    parts = ["<ul class='strike-note-list'>"]
    for row in rows:
        parts.append(
            "<li>"
            f"<span class='strike-dot {'ok' if bool(row['ok']) else 'warn'}'></span>"
            f"<span>{escape(str(row['label']))}</span>"
            "</li>"
        )
    parts.append("</ul>")
    return "".join(parts)


def render_features_screen(
    *,
    st: object,
    output_root: object,
    board: pd.DataFrame,
    settings: DashboardSettings,
) -> None:
    """Render the feature inspection screen."""
    importance = get_feature_importance(output_root)
    drift = feature_drift_rows(output_root)
    checks = leakage_checks(board, settings=settings)
    st.markdown(
        "<div class='strike-screen-head'>"
        "<div>"
        "<div class='strike-screen-title'>FEATURE INSPECTION "
        "<span class='strike-dim'>/ importance · drift · leakage</span></div>"
        "<div class='strike-crumb'>current model · starter-strikeout baseline</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    columns = st.columns([2, 1], gap="medium")
    with columns[0]:
        st.markdown(
            "<div class='strike-panel-head'><h4>GLOBAL IMPORTANCE</h4>"
            "<span class='strike-dim'>mean |importance| · sign from coefficient direction</span></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            feature_bars_fig(importance.to_dict("records")),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with columns[1]:
        st.markdown(
            "<div class='strike-panel-head'><h4>DRIFT (PSI L7 vs TRAIN)</h4></div>",
            unsafe_allow_html=True,
        )
        st.markdown(_drift_html(drift), unsafe_allow_html=True)
        st.markdown(
            "<div class='strike-panel-head'><h4>LEAKAGE CHECKS</h4></div>",
            unsafe_allow_html=True,
        )
        st.markdown(_leakage_html(checks), unsafe_allow_html=True)
