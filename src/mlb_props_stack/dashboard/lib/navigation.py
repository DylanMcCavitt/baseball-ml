"""Streamlit query-param navigation helpers for the dashboard."""

from __future__ import annotations

from datetime import date
from typing import Any


def set_dashboard_query_params(
    streamlit_module: Any,
    *,
    screen: str,
    board_date: date | None,
    pitcher_id: str | None = None,
) -> None:
    """Replace dashboard query params with the requested screen context."""
    target: dict[str, str] = {"screen": screen}
    if board_date is not None:
        target["board_date"] = board_date.isoformat()
    if pitcher_id:
        target["pitcher_id"] = pitcher_id

    query_params = getattr(streamlit_module, "query_params", None)
    if query_params is None:
        return
    if hasattr(query_params, "clear"):
        query_params.clear()
    for key, value in target.items():
        query_params[key] = value


def rerun_streamlit(streamlit_module: Any) -> None:
    """Trigger a Streamlit rerun when the runtime exposes the hook."""
    rerun = getattr(streamlit_module, "rerun", None)
    if callable(rerun):
        rerun()
