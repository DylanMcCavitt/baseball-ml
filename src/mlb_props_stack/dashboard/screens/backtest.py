"""Backtest screen renderer."""

from __future__ import annotations

from html import escape

import pandas as pd

from ..lib.data import (
    backtest_date_range,
    backtest_kpis,
    backtest_scatter,
    format_pct,
    get_backtest,
    get_calibration,
)
from ..lib.plots import calibration_fig, pnl_fig, roi_fig, scatter_fig
from ..lib.theme import kpi_strip_html


def render_backtest_screen(
    *,
    st: object,
    output_root: object,
    model_summary: dict | None,
) -> None:
    """Render the backtest screen."""
    start_date, end_date = backtest_date_range(output_root)
    st.markdown(
        "<div class='strike-screen-head'>"
        "<div>"
        "<div class='strike-screen-title'>BACKTEST "
        "<span class='strike-dim'>/ walk-forward · timestamp-valid</span></div>"
        f"<div class='strike-crumb'>train: expanding · eval: 1d hold-out · {escape(start_date or 'n/a')} → {escape(end_date or 'n/a')}</div>"
        "</div>"
        "<div class='strike-toolbar'>"
        "<span class='strike-chip on'>season-to-date</span>"
        "<span class='strike-chip'>L30d</span>"
        "<span class='strike-chip'>L7d</span>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    kpis = backtest_kpis(output_root, model_summary=model_summary)
    st.markdown(
        kpi_strip_html(
            [
                {"label": "bets", "value": str(kpis["bets"])},
                {
                    "label": "cum pnl",
                    "value": f"{kpis['cum_pnl']:.2f}u",
                    "tone": "pos" if kpis["cum_pnl"] >= 0.0 else "neg",
                },
                {"label": "roi", "value": format_pct(kpis["roi"], digits=1)},
                {"label": "clv", "value": format_pct(kpis["clv"], digits=2)},
                {"label": "brier", "value": "n/a" if kpis["brier"] is None else f"{float(kpis['brier']):.3f}"},
                {
                    "label": "log-loss",
                    "value": "n/a" if kpis["log_loss"] is None else f"{float(kpis['log_loss']):.3f}",
                },
                {
                    "label": "max dd",
                    "value": "n/a" if kpis["max_drawdown"] is None else f"{float(kpis['max_drawdown']):.2f}u",
                    "tone": "neg" if kpis["max_drawdown"] is not None and float(kpis["max_drawdown"]) < 0 else None,
                },
            ]
        ),
        unsafe_allow_html=True,
    )

    series = get_backtest(output_root, start=start_date, end=end_date)
    calibration = get_calibration(output_root)
    scatter_rows = backtest_scatter(output_root)

    if series.empty:
        st.markdown(
            "<div class='strike-empty'>No date-level backtest reporting artifacts were found yet. "
            "Build a walk-forward backtest that writes `roi_summary.jsonl`, `clv_summary.jsonl`, and `bet_reporting.jsonl`."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        scrub_index = st.slider(
            "Scrubber",
            min_value=0,
            max_value=len(series) - 1,
            value=len(series) - 1,
            key="backtest_scrub_index",
        )
        scrub_row = series.iloc[int(scrub_index)]
        chart_columns = st.columns([2, 1], gap="medium")
        with chart_columns[0]:
            st.markdown(
                "<div class='strike-panel-head'><h4>WALK-FORWARD CUM PNL · CLV</h4>"
                "<span class='strike-dim'>drag the scrubber to inspect one date</span></div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                pnl_fig(series.to_dict("records")),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown(
                "<div class='strike-statusbar'>"
                f"<span>date {escape(str(scrub_row['date']))}</span>"
                "<span class='strike-sep'>│</span>"
                f"<span>cum <span class='strike-pos'>{float(scrub_row['cum_pnl_units']):.2f}u</span></span>"
                "<span class='strike-sep'>│</span>"
                f"<span>clv <span class='strike-blue'>{float(scrub_row['clv_units']):.3f}</span></span>"
                "</div>",
                unsafe_allow_html=True,
            )
        with chart_columns[1]:
            st.markdown(
                "<div class='strike-panel-head'><h4>CALIBRATION</h4>"
                "<span class='strike-dim'>reliability</span></div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                calibration_fig(calibration.to_dict("records")),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        bottom_columns = st.columns([1, 1], gap="medium")
        with bottom_columns[0]:
            st.markdown(
                "<div class='strike-panel-head'><h4>MODEL vs MARKET</h4>"
                "<span class='strike-dim'>p_model ↕ p_market</span></div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                scatter_fig(scatter_rows),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with bottom_columns[1]:
            st.markdown(
                "<div class='strike-panel-head'><h4>ROLLING 30D ROI</h4></div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                roi_fig(series[["date", "rolling_30d_roi"]].rename(columns={"rolling_30d_roi": "roi"}).to_dict("records")),
                use_container_width=True,
                config={"displayModeBar": False},
            )
