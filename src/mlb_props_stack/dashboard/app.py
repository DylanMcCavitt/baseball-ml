"""Streamlit dashboard for current slate and recent paper-tracking results."""

from __future__ import annotations

from datetime import date
import os
from pathlib import Path
from typing import Any

from mlb_props_stack.paper_tracking import (
    list_available_daily_candidate_dates,
    load_latest_daily_candidates,
    load_latest_paper_results,
    summarize_paper_results_by_date,
)


def build_dashboard_banner() -> str:
    """Return the current dashboard purpose string."""
    return "Streamlit dashboard for current slate candidates and recent paper results."


def _current_slate_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    actionable_rows = [row for row in rows if bool(row.get("bet_placed"))]
    total_stake = round(
        sum(float(row["stake_fraction"]) for row in actionable_rows),
        6,
    )
    mean_edge_pct = None
    if actionable_rows:
        mean_edge_pct = round(
            sum(float(row["edge_pct"]) for row in actionable_rows) / len(actionable_rows),
            6,
        )
    return {
        "scored_candidates": len(rows),
        "actionable_candidates": len(actionable_rows),
        "recommended_stake_units": total_stake,
        "mean_actionable_edge_pct": mean_edge_pct,
    }


def _paper_performance_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    settled_rows = [
        row for row in rows if row.get("settlement_status") in {"win", "loss", "push"}
    ]
    total_stake = round(
        sum(float(row["stake_fraction"]) for row in settled_rows),
        6,
    )
    total_profit = round(
        sum(float(row["profit_units"]) for row in settled_rows),
        6,
    )
    roi = round(total_profit / total_stake, 6) if total_stake > 0.0 else None
    return {
        "settled_bets": len(settled_rows),
        "pending_bets": len(rows) - len(settled_rows),
        "total_profit_units": total_profit,
        "roi": roi,
    }


def _current_slate_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table_rows: list[dict[str, Any]] = []
    for row in rows:
        table_rows.append(
            {
                "rank": row["slate_rank"],
                "actionable": bool(row["bet_placed"]),
                "pitcher": row["player_name"],
                "sportsbook": row["sportsbook_title"],
                "line": row["line"],
                "side": row["selected_side"],
                "odds": row["selected_odds"],
                "edge_pct": row["edge_pct"],
                "expected_value_pct": row["expected_value_pct"],
                "stake_fraction": row["stake_fraction"],
                "captured_at": row["captured_at"],
            }
        )
    return table_rows


def _recent_paper_results_table(rows: list[dict[str, Any]], *, max_rows: int = 25) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            str(row["official_date"]),
            int(row["actionable_rank"]),
            str(row["paper_result_id"]),
        ),
        reverse=True,
    )
    table_rows: list[dict[str, Any]] = []
    for row in sorted_rows[:max_rows]:
        table_rows.append(
            {
                "official_date": row["official_date"],
                "pitcher": row["player_name"],
                "sportsbook": row["sportsbook_title"],
                "line": row["line"],
                "side": row["selected_side"],
                "edge_pct": row["edge_pct"],
                "paper_result": row["paper_result"],
                "profit_units": row["profit_units"],
                "clv_outcome": row["clv_outcome"],
            }
        )
    return table_rows


def render_dashboard_page(
    *,
    output_dir: Path | str = "data",
    target_date: date | None = None,
) -> None:
    """Render the Streamlit dashboard page for the latest slate and paper results."""
    try:
        import streamlit as st
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Streamlit is not installed. Run `uv sync --extra dev` before launching the dashboard."
        ) from error

    output_root = Path(output_dir)
    available_dates = list_available_daily_candidate_dates(output_dir=output_root)
    st.set_page_config(page_title="MLB Props Stack", layout="wide")
    st.title("MLB Props Stack")
    st.caption(build_dashboard_banner())
    if not available_dates:
        st.info(
            "No `daily_candidates` artifacts were found. Run "
            "`uv run python -m mlb_props_stack build-daily-candidates` first."
        )
        return

    selected_date = (
        target_date.isoformat()
        if target_date is not None and target_date.isoformat() in available_dates
        else available_dates[-1]
    )
    selected_date = st.sidebar.selectbox(
        "Slate date",
        available_dates,
        index=available_dates.index(selected_date),
    )

    daily_rows = load_latest_daily_candidates(
        output_dir=output_root,
        target_date=date.fromisoformat(selected_date),
    )
    paper_rows = load_latest_paper_results(output_dir=output_root)
    slate_metrics = _current_slate_metrics(daily_rows)
    paper_metrics = _paper_performance_metrics(paper_rows)

    slate_columns = st.columns(4)
    slate_columns[0].metric("Scored candidates", slate_metrics["scored_candidates"])
    slate_columns[1].metric(
        "Actionable candidates",
        slate_metrics["actionable_candidates"],
    )
    slate_columns[2].metric(
        "Recommended stake units",
        slate_metrics["recommended_stake_units"],
    )
    slate_columns[3].metric(
        "Mean actionable edge",
        (
            f"{float(slate_metrics['mean_actionable_edge_pct']):.2%}"
            if slate_metrics["mean_actionable_edge_pct"] is not None
            else "n/a"
        ),
    )

    performance_columns = st.columns(4)
    performance_columns[0].metric("Settled paper bets", paper_metrics["settled_bets"])
    performance_columns[1].metric("Pending paper bets", paper_metrics["pending_bets"])
    performance_columns[2].metric(
        "Total profit units",
        paper_metrics["total_profit_units"],
    )
    performance_columns[3].metric(
        "ROI",
        (
            f"{float(paper_metrics['roi']):.2%}"
            if paper_metrics["roi"] is not None
            else "n/a"
        ),
    )

    st.subheader(f"Current Slate: {selected_date}")
    st.dataframe(_current_slate_table(daily_rows), use_container_width=True)

    st.subheader("Recent Paper Performance")
    st.dataframe(
        summarize_paper_results_by_date(paper_rows, max_dates=7),
        use_container_width=True,
    )

    st.subheader("Recent Paper Bets")
    st.dataframe(
        _recent_paper_results_table(paper_rows),
        use_container_width=True,
    )


def main() -> None:
    output_dir = os.environ.get("MLB_PROPS_STACK_DATA_DIR", "data")
    render_dashboard_page(output_dir=output_dir)


if __name__ == "__main__":
    main()
