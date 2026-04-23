"""Strike Ops dashboard entrypoint."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from html import escape
import os
from pathlib import Path
import sys
import tomllib
from typing import Any


if __package__ in {None, ""}:  # pragma: no cover - exercised by file-entry runtime check
    repo_root = Path(__file__).resolve().parents[3]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def build_dashboard_banner() -> str:
    """Return the current dashboard purpose string."""
    return (
        "Strike Ops Streamlit dashboard for current slate candidates, pitcher detail, "
        "backtests, registry, and feature inspection."
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _user_config_path() -> Path:
    return _repo_root() / "user_config.toml"


def _query_params_dict(streamlit_module: Any) -> dict[str, str]:
    raw = getattr(streamlit_module, "query_params", {})
    if hasattr(raw, "to_dict"):
        return {str(key): str(value) for key, value in raw.to_dict().items()}
    return {str(key): str(value) for key, value in dict(raw).items()}


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _load_user_config() -> tuple[dict[str, Any], dict[str, str]]:
    path = _user_config_path()
    if not path.exists():
        return {}, {}
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    dashboard = {
        str(key): value for key, value in (payload.get("dashboard") or {}).items()
    }
    registry = {
        str(key): str(value)
        for key, value in (payload.get("registry_stages") or {}).items()
    }
    return dashboard, registry


def _write_user_config(
    *,
    settings: dict[str, Any],
    stage_overrides: dict[str, str],
) -> None:
    lines = ["[dashboard]"]
    for key, value in settings.items():
        if value is None:
            continue
        lines.append(f"{key} = {_toml_scalar(value)}")
    if stage_overrides:
        lines.append("")
        lines.append("[registry_stages]")
        for key, value in stage_overrides.items():
            lines.append(f'"{key}" = {_toml_scalar(value)}')
    _user_config_path().write_text("\n".join(lines) + "\n", encoding="utf-8")


NAV_TABS: tuple[tuple[str, str, str], ...] = (
    ("board", "BOARD", "1"),
    ("pitcher", "PITCHER", "2"),
    ("backtest", "BACKTEST", "3"),
    ("registry", "MLFLOW", "4"),
    ("features", "FEATURES", "5"),
    ("config", "CONFIG", "6"),
)


def _set_dashboard_query_params(
    streamlit_module: Any,
    *,
    screen: str,
    board_date: date | None,
    pitcher_id: str | None = None,
) -> None:
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


def _rerun_streamlit(streamlit_module: Any) -> None:
    rerun = getattr(streamlit_module, "rerun", None)
    if callable(rerun):
        rerun()


def _render_nav_controls(
    *,
    streamlit_module: Any,
    active_screen: str,
    board_date: date | None,
    selected_pitcher_id: str | None,
) -> None:
    labels = {
        screen: f"{label} \u2318{key_hint}"
        for screen, label, key_hint in NAV_TABS
    }

    columns = streamlit_module.columns(len(NAV_TABS))
    for column, (screen, _, _) in zip(columns, NAV_TABS, strict=True):
        if column.button(
            labels[screen],
            key=f"strike_ops_screen_nav_{screen}",
            type="primary" if screen == active_screen else "secondary",
            use_container_width=True,
        ):
            _set_dashboard_query_params(
                streamlit_module,
                screen=screen,
                board_date=board_date,
                pitcher_id=selected_pitcher_id if screen == "pitcher" else None,
            )
            _rerun_streamlit(streamlit_module)


def _ticker_html(context: dict[str, str]) -> str:
    return (
        "<div class='strike-ticker'>"
        "<span class='strike-brand'>◆ STRIKE · OPS</span>"
        "<span class='strike-sep'>│</span>"
        f"<span>SLATE {escape(context['slate_date'])}</span>"
        "<span class='strike-sep'>│</span>"
        f"<span>{escape(context['game_count'])}</span>"
        "<span class='strike-sep'>│</span>"
        f"<span>{escape(context['first_pitch'])}</span>"
        "<span class='strike-sep'>│</span>"
        f"<span class='strike-pill-live'>● {escape(context['live_label'])}</span>"
        "<span class='strike-sep'>│</span>"
        f"<span>{escape(context['model'])}</span>"
        "<span class='strike-sep'>│</span>"
        f"<span>{escape(context['bankroll'])}</span>"
        f"<span style='margin-left:auto'>{escape(context['clock'])}</span>"
        "</div>"
    )


def _header_html(
    *,
    active_screen: str,
) -> str:
    current = next(
        (label for screen, label, _ in NAV_TABS if screen == active_screen),
        active_screen.upper(),
    )
    return (
        "<div class='strike-header'>"
        "<div class='strike-logo'>"
        "<span class='strike-logo-mark'>S</span>"
        "<span>STRIKE<span class='strike-dim'>·</span>OPS</span>"
        "<span style='margin-left:auto;color:var(--dim);font-size:10px'>v0.3.1</span>"
        "</div>"
        "<div class='strike-header-right'>"
        f"<span>{escape(current)}</span><span class='strike-sep'>│</span>"
        "<span>PY 3.11</span><span class='strike-sep'>│</span>"
        "<span>MLflow 2.x</span><span class='strike-sep'>│</span>"
        "<span>Streamlit</span><span class='strike-sep'>│</span>"
        "<span class='strike-connected'>CONNECTED</span>"
        "</div>"
        "</div>"
    )


def _statusbar_html(
    *,
    devig_method: str,
    edge_min: float,
    kelly_fraction: float,
    today_profit_units: float | None,
    row_count: int,
) -> str:
    pnl_label = "n/a" if today_profit_units is None else f"{today_profit_units:+.2f}u"
    pnl_class = "strike-pos" if today_profit_units is not None and today_profit_units >= 0 else "strike-neg"
    return (
        "<div class='strike-statusbar'>"
        "<span>READY</span><span class='strike-sep'>│</span>"
        f"<span>rows: {row_count}</span><span class='strike-sep'>│</span>"
        f"<span>devig: {escape(devig_method)}</span><span class='strike-sep'>│</span>"
        f"<span>edge≥{edge_min:.1%}</span><span class='strike-sep'>│</span>"
        f"<span>kelly×{kelly_fraction:.2f}</span><span class='strike-sep'>│</span>"
        f"<span>pnl today <span class='{pnl_class}'>{escape(pnl_label)}</span></span><span class='strike-sep'>│</span>"
        "<span style='margin-left:auto'>press 1-6 in the mock · use the nav links here</span>"
        "</div>"
    )


def render_dashboard_page(
    *,
    output_dir: Path | str = "data",
    target_date: date | None = None,
) -> None:
    """Render the Streamlit dashboard page."""
    try:
        import pandas as pd
        import streamlit as st
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Dashboard dependencies are not installed. Run `uv sync --extra dev` before launching the dashboard."
        ) from error

    from mlb_props_stack.dashboard.lib.data import (
        DashboardSettings,
        latest_pitcher_row,
        list_available_board_dates,
        load_board_dataframe,
        load_latest_model_summary,
        load_paper_results_dataframe,
        paper_performance_metrics,
        ticker_context,
    )
    from mlb_props_stack.dashboard.lib.mlflow_io import transition_stage
    from mlb_props_stack.dashboard.lib.theme import build_theme_css
    from mlb_props_stack.dashboard.screens import (
        render_backtest_screen,
        render_board_screen,
        render_config_screen,
        render_features_screen,
        render_pitcher_screen,
        render_registry_screen,
    )
    from mlb_props_stack.tracking import TrackingConfig

    st.set_page_config(page_title="Strike Ops", layout="wide")
    st.markdown(build_theme_css(), unsafe_allow_html=True)
    tracking = TrackingConfig()

    persisted_settings, persisted_stage_overrides = _load_user_config()
    default_settings = DashboardSettings(**persisted_settings)
    if "_strike_ops_settings" not in st.session_state:
        st.session_state["_strike_ops_settings"] = asdict(default_settings)
    if "_strike_ops_stage_overrides" not in st.session_state:
        st.session_state["_strike_ops_stage_overrides"] = dict(persisted_stage_overrides)

    settings = DashboardSettings(**st.session_state["_strike_ops_settings"])
    stage_overrides = dict(st.session_state["_strike_ops_stage_overrides"])
    output_root = Path(output_dir)

    available_board_dates = list_available_board_dates(output_root=output_root)
    params = _query_params_dict(st)
    active_screen = params.get("screen", "board").lower()
    if active_screen not in {"board", "pitcher", "backtest", "registry", "features", "config"}:
        active_screen = "board"

    preferred_date = (
        target_date.isoformat()
        if target_date is not None
        else params.get("board_date")
    )
    if available_board_dates:
        initial_date = (
            preferred_date if preferred_date in available_board_dates else available_board_dates[-1]
        )
        selected_date_str = st.selectbox(
            "Slate date",
            options=available_board_dates,
            index=available_board_dates.index(initial_date),
            key="strike_ops_board_date",
        )
        board_date = date.fromisoformat(selected_date_str)
    else:
        board_date = None

    board = pd.DataFrame()
    board_source = None
    if board_date is not None:
        board, board_source = load_board_dataframe(
            output_root,
            target_date=board_date,
            settings=settings,
        )

    selected_pitcher_id = params.get("pitcher_id") or st.session_state.get("selected_pitcher_id")
    if selected_pitcher_id:
        st.session_state["selected_pitcher_id"] = selected_pitcher_id
    selected_row = latest_pitcher_row(board, pitcher_id=selected_pitcher_id)

    if active_screen == "pitcher" and selected_row is None:
        active_screen = "board"

    model_summary = load_latest_model_summary(output_root, run_id=settings.active_run_id)
    paper_results = load_paper_results_dataframe(output_root, target_date=board_date)
    paper_metrics = paper_performance_metrics(paper_results)
    ticker = ticker_context(board, settings=settings)

    st.markdown(_ticker_html(ticker), unsafe_allow_html=True)
    st.markdown(
        _header_html(
            active_screen=active_screen,
        ),
        unsafe_allow_html=True,
    )
    _render_nav_controls(
        streamlit_module=st,
        active_screen=active_screen,
        board_date=board_date,
        selected_pitcher_id=str(selected_row["pitcher_id"]) if selected_row is not None else None,
    )

    if active_screen == "board":
        render_board_screen(
            st=st,
            board=board,
            board_date=board_date,
            board_source=board_source,
            selected_pitcher_id=str(selected_row["pitcher_id"]) if selected_row is not None else None,
            settings=settings,
        )
    elif active_screen == "pitcher":
        render_pitcher_screen(
            st=st,
            output_root=output_root,
            row=selected_row,
            settings=settings,
        )
    elif active_screen == "backtest":
        render_backtest_screen(
            st=st,
            output_root=output_root,
            model_summary=model_summary,
        )
    elif active_screen == "registry":
        def _stage_change(run_row: dict[str, Any], stage: str) -> str:
            overrides = dict(st.session_state["_strike_ops_stage_overrides"])
            overrides[str(run_row["run_id"])] = stage
            st.session_state["_strike_ops_stage_overrides"] = overrides
            message = (
                f"Updated local dashboard stage for {run_row['name']} to {stage}."
            )
            if run_row.get("model_name") and run_row.get("version"):
                try:
                    transition_stage(
                        tracking_uri=tracking.tracking_uri,
                        model_name=str(run_row["model_name"]),
                        version=str(run_row["version"]),
                        stage=stage,
                    )
                    message = (
                        f"Updated MLflow registry stage for {run_row['name']} to {stage}."
                    )
                except Exception as error:  # pragma: no cover - depends on local MLflow state
                    message = (
                        f"{message} MLflow registry transition was unavailable: {error}"
                    )
            return message

        render_registry_screen(
            st=st,
            output_root=output_root,
            settings=settings,
            stage_overrides=stage_overrides,
            on_stage_change=_stage_change,
        )
    elif active_screen == "features":
        render_features_screen(
            st=st,
            output_root=output_root,
            board=board,
            settings=settings,
        )
    elif active_screen == "config":
        updated_settings, save_clicked = render_config_screen(
            st=st,
            settings=settings,
        )
        st.session_state["_strike_ops_settings"] = asdict(updated_settings)
        settings = updated_settings
        if save_clicked:
            _write_user_config(
                settings=st.session_state["_strike_ops_settings"],
                stage_overrides=st.session_state["_strike_ops_stage_overrides"],
            )
            st.success("Persisted dashboard settings to user_config.toml.")

    st.markdown(
        _statusbar_html(
            devig_method=settings.devig_method,
            edge_min=settings.edge_min,
            kelly_fraction=settings.kelly_fraction,
            today_profit_units=paper_metrics.get("total_profit_units"),
            row_count=len(board),
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    render_dashboard_page(
        output_dir=Path(os.environ.get("MLB_PROPS_STACK_DATA_DIR", "data"))
    )


if __name__ == "__main__":
    main()
