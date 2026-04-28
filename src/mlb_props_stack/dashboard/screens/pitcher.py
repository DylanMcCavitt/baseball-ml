"""Pitcher detail screen renderer."""

from __future__ import annotations

from datetime import date
from html import escape

import pandas as pd

from mlb_props_stack.pricing import fair_american_odds

from ..lib.data import (
    DashboardSettings,
    format_pct,
    get_feature_importance,
    get_pmf,
    get_recent_form,
)
from ..lib.navigation import rerun_streamlit, set_dashboard_query_params
from ..lib.plots import pmf_fig, probability_comparison_fig


def _recent_form_html(recent_form: pd.DataFrame) -> str:
    rows: list[str] = []
    for _, row in recent_form.iterrows():
        side_label = escape(str(row["res"])) if row["res"] else "—"
        rows.append(
            "<tr>"
            f"<td>{escape(str(row['date']))}</td>"
            f"<td>{escape(str(row['opp']))}</td>"
            f"<td class='strike-num'>{'—' if pd.isna(row['ip']) else row['ip']}</td>"
            f"<td class='strike-num'><strong>{int(row['k'])}</strong></td>"
            f"<td class='strike-num'>{'—' if pd.isna(row['bb']) else row['bb']}</td>"
            f"<td class='strike-num'>{'—' if pd.isna(row['er']) else row['er']}</td>"
            f"<td class='strike-num'>{'—' if pd.isna(row['pit']) else int(row['pit'])}</td>"
            f"<td class='strike-num'>{'—' if pd.isna(row['line']) else row['line']}</td>"
            f"<td>{side_label}</td>"
            "</tr>"
        )
    return (
        "<div class='strike-grid-wrap'>"
        "<table class='strike-grid'>"
        "<thead><tr>"
        "<th style='text-align:left'>DATE</th>"
        "<th style='text-align:left'>OPP</th>"
        "<th>IP</th><th>K</th><th>BB</th><th>ER</th><th>PIT</th><th>LINE</th><th>RES</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _feature_contrib_html(importance: pd.DataFrame) -> str:
    if importance.empty:
        return "<div class='strike-empty'>No feature-importance artifact found for this model run.</div>"
    rows: list[str] = []
    top_rows = importance.sort_values("importance", ascending=False).head(8)
    scale = float(top_rows["importance"].max()) or 1.0
    for _, row in top_rows.iterrows():
        width = float(row["importance"]) / scale * 100.0
        color = "var(--pos)" if str(row["direction"]).startswith("+") else "var(--neg)"
        rows.append(
            "<div class='strike-mini-row'>"
            f"<span>{escape(str(row['name']))}</span>"
            "<div class='strike-mini-track'>"
            f"<span style='width:{width:.1f}%;background:{color}'></span>"
            "</div>"
            f"<span class='{'strike-pos' if str(row['direction']).startswith('+') else 'strike-neg'}'>"
            f"{escape(str(row['direction']))}{float(row['importance']):.2f}</span>"
            "</div>"
        )
    return "<div class='strike-mini-bars'>" + "".join(rows) + "</div>"


def _guardrail_html(row: pd.Series, settings: DashboardSettings) -> str:
    if str(row.get("source") or "") == "edge_candidates":
        items = [
            (
                "model validation",
                bool(row.get("wager_approved")),
                str(row.get("approval_reason") or "validation-derived approval unavailable"),
            ),
            (
                "confidence bucket",
                bool(row.get("wager_approved")),
                str(row.get("model_confidence_bucket") or "n/a"),
            ),
            (
                "correlation group",
                int(row.get("correlation_group_rank") or 1) <= 1,
                (
                    f"rank {int(row.get('correlation_group_rank') or 1)} of "
                    f"{int(row.get('correlation_group_size') or 1)}"
                ),
            ),
            (
                "research readiness",
                str(row.get("research_readiness_status") or "") != "research_only",
                str(row.get("research_readiness_status") or "research_only"),
            ),
        ]
        parts: list[str] = []
        for label, ok, value in items:
            parts.append(
                "<div class='strike-guard {}'>".format("" if ok else "bad")
                + "<div class='strike-guard-top'>"
                + "<span class='strike-dot {}'></span>".format("ok" if ok else "bad")
                + f"<span class='strike-dim'>{escape(label)}</span>"
                + "</div>"
                + f"<div>{escape(value)}</div>"
                + "</div>"
            )
        return "<div class='strike-guard-grid'>" + "".join(parts) + "</div>"

    model_age_days = row.get("model_age_days")
    model_age_text = (
        f"trained {float(model_age_days):.1f}d ago"
        if pd.notna(model_age_days)
        else "training timestamp unavailable"
    )
    items = [
        (
            "edge >= threshold",
            bool(row["cleared_edge_gate"]),
            f"{float(row['edge']):.1%} >= {settings.edge_min:.1%}",
        ),
        (
            "vig cleared",
            bool(row["cleared_vig_gate"]),
            f"{float(row['raw_hold'] or 0.0):.1%} hold <= {settings.max_hold:.1%}",
        ),
        (
            "stake cap",
            bool(row["cleared_stake_gate"]),
            f"{float(row['kelly_units']):.2f}u <= {settings.max_stake_units:.2f}u",
        ),
        (
            "correlation",
            bool(row["cleared_correlation_gate"]),
            "no co-linear same-slate play",
        ),
        (
            "model staleness",
            bool(row["cleared_model_age_gate"]),
            model_age_text,
        ),
        (
            "late-scratch watch",
            bool(row["cleared_status_gate"]),
            str(row["pitcher_status"]),
        ),
    ]
    parts: list[str] = []
    for label, ok, value in items:
        parts.append(
            "<div class='strike-guard {}'>".format("" if ok else "bad")
            + "<div class='strike-guard-top'>"
            + "<span class='strike-dot {}'></span>".format("ok" if ok else "bad")
            + f"<span class='strike-dim'>{escape(label)}</span>"
            + "</div>"
            + f"<div>{escape(value)}</div>"
            + "</div>"
        )
    return "<div class='strike-guard-grid'>" + "".join(parts) + "</div>"


def render_pitcher_screen(
    *,
    st: object,
    output_root: object,
    row: pd.Series | None,
    settings: DashboardSettings,
) -> None:
    """Render the pitcher detail screen."""
    if row is None:
        st.markdown(
            "<div class='strike-empty'>Select a board row to open pitcher detail.</div>",
            unsafe_allow_html=True,
        )
        return

    board_date = str(row["official_date"])
    if st.button(
        "← board",
        key="pitcher_back_to_board",
        type="secondary",
    ):
        set_dashboard_query_params(
            st,
            screen="board",
            board_date=date.fromisoformat(board_date),
        )
        rerun_streamlit(st)

    st.markdown(
        "<div class='strike-screen-head'>"
        "<div>"
        f"<div class='strike-screen-title'>{escape(str(row['pitcher']).upper())} "
        f"<span class='strike-dim'>/ {escape(str(row['team']))} vs {escape(str(row['opp']))} / {escape(board_date)}</span></div>"
        "</div>"
        "<div class='strike-toolbar'>"
        f"<span class='strike-chip on'>{escape(str(row['side']).upper())} {float(row['line']):.1f}</span>"
        f"<span class='strike-chip'>{int(row['american']):+d}</span>"
        f"<span class='strike-chip'>{format_pct(float(row['edge']), digits=1)} edge</span>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    pmf_rows, ladder_row = get_pmf(
        output_root,
        official_date=str(row["official_date"]),
        pitcher_mlb_id=int(row["pitcher_mlb_id"]) if pd.notna(row["pitcher_mlb_id"]) else None,
        line=float(row["line"]),
        model_run_id=str(row["model_run_id"]) if row.get("model_run_id") else None,
    )
    recent_form = get_recent_form(
        output_root,
        pitcher_mlb_id=int(row["pitcher_mlb_id"]) if pd.notna(row["pitcher_mlb_id"]) else None,
    )
    importance = get_feature_importance(
        output_root,
        run_id=str(row["model_run_id"]) if row.get("model_run_id") else None,
    )

    top_columns = st.columns([2, 1], gap="medium")
    with top_columns[0]:
        st.markdown(
            "<div class='strike-panel-head'><h4>STRIKEOUT PMF</h4>"
            f"<span class='strike-dim'>μ = {float(ladder_row.get('model_mean') if ladder_row else 0.0):.2f}</span></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            pmf_fig(pmf_rows=pmf_rows, line=float(row["line"])),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.markdown(
            "<div class='strike-legend'>"
            "<span><i class='strike-legend-box' style='background:var(--pos)'></i>bars right of line</span>"
            "<span><i class='strike-legend-box' style='background:var(--dim)'></i>bars left of line</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    with top_columns[1]:
        st.markdown(
            "<div class='strike-panel-head'><h4>MODEL vs MARKET</h4></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            probability_comparison_fig(
                model_probability=float(row["p_model"]),
                market_probability=float(row["p_market"]),
                edge_probability=float(row["edge"]),
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
        fair_odds = fair_american_odds(float(row["p_model"])) if 0.0 < float(row["p_model"]) < 1.0 else None
        st.markdown(
            "<div class='strike-kpi-strip'>"
            f"<div class='strike-kpi'><div class='strike-kpi-label'>american</div><div class='strike-kpi-value sm'>{int(row['american']):+d}</div></div>"
            f"<div class='strike-kpi'><div class='strike-kpi-label'>fair</div><div class='strike-kpi-value sm'>{'n/a' if fair_odds is None else f'{fair_odds:+d}'}</div></div>"
            f"<div class='strike-kpi'><div class='strike-kpi-label'>kelly ×{settings.kelly_fraction:.2f}</div><div class='strike-kpi-value sm'>{float(row['kelly_units']):.2f}u</div></div>"
            f"<div class='strike-kpi'><div class='strike-kpi-label'>decision ts</div><div class='strike-kpi-value sm'>{escape(str(row['captured_at']) if row['captured_at'] is not None else 'n/a')}</div></div>"
            f"<div class='strike-kpi'><div class='strike-kpi-label'>projected K</div><div class='strike-kpi-value sm'>{float(row.get('model_projection') or 0.0):.2f}</div></div>"
            f"<div class='strike-kpi'><div class='strike-kpi-label'>over / under</div><div class='strike-kpi-value sm'>{format_pct(float(row.get('model_over_probability') or 0.0), digits=1)} / {format_pct(float(row.get('model_under_probability') or 0.0), digits=1)}</div></div>"
            f"<div class='strike-kpi'><div class='strike-kpi-label'>cal bucket</div><div class='strike-kpi-value sm'>{escape(str(row.get('model_confidence_bucket') or 'n/a'))}</div></div>"
            f"<div class='strike-kpi'><div class='strike-kpi-label'>approval</div><div class='strike-kpi-value sm'>{escape(str(row.get('wager_gate_status') or 'n/a'))}</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )

    middle_columns = st.columns([2, 1], gap="medium")
    with middle_columns[0]:
        st.markdown(
            "<div class='strike-panel-head'><h4>RECENT FORM (L5)</h4></div>",
            unsafe_allow_html=True,
        )
        if recent_form.empty:
            st.markdown(
                "<div class='strike-empty'>No recent starter outcomes were found for this pitcher.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(_recent_form_html(recent_form), unsafe_allow_html=True)
    with middle_columns[1]:
        st.markdown(
            "<div class='strike-panel-head'><h4>FEATURE CONTRIBS</h4></div>",
            unsafe_allow_html=True,
        )
        st.markdown(_feature_contrib_html(importance), unsafe_allow_html=True)

    st.markdown(
        "<div class='strike-panel-head'><h4>GUARDRAILS</h4></div>",
        unsafe_allow_html=True,
    )
    st.markdown(_guardrail_html(row, settings), unsafe_allow_html=True)
