"""Board screen renderer."""

from __future__ import annotations

from datetime import date
from html import escape

import pandas as pd

from ..lib.data import (
    DashboardSettings,
    current_slate_metrics,
    format_pct,
    format_timestamp,
    group_board_by_pitcher,
    line_snapshot_display_id,
)
from ..lib.navigation import rerun_streamlit, set_dashboard_query_params
from ..lib.theme import kpi_strip_html


def _short_cell_id(value: object, *, length: int = 8) -> str:
    if value is None or pd.isna(value) or str(value) == "":
        return "n/a"
    return str(value)[:length]


def _timestamp_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    return format_timestamp(value)


def _line_group_cell(row: pd.Series) -> str:
    line_row_count = int(row.get("line_row_count") or 1)
    hidden_line_row_count = int(row.get("hidden_line_row_count") or 0)
    sportsbook_count = int(row.get("sportsbook_count") or 0)
    summary = escape(str(row.get("line_group_summary") or ""))
    if line_row_count <= 1:
        return "<span class='strike-dim'>line row</span>"
    return (
        f"<strong>1 of {line_row_count}</strong><br>"
        f"<span class='strike-dim'>+{hidden_line_row_count} hidden · "
        f"{sportsbook_count} books</span><br>"
        f"<span class='strike-dim'>{summary}</span>"
    )


def _board_table_html(
    board: pd.DataFrame,
    *,
    selected_pitcher_id: str | None,
) -> str:
    rows: list[str] = []
    for _, row in board.iterrows():
        row_class = []
        if not bool(row["cleared"]):
            row_class.append("muted")
        if selected_pitcher_id and str(row["pitcher_id"]) == selected_pitcher_id:
            row_class.append("sel")
        side_class = "over" if str(row["side"]) == "over" else "under"
        confidence_width = max(0.0, min(100.0, float(row["conf"]) * 100.0))
        odds = row["american"]
        odds_label = "n/a" if pd.isna(odds) else f"{int(odds):+d}"
        edge = row["edge"]
        edge_label = "n/a" if pd.isna(edge) else f"{edge:+.1%}"
        sportsbook = escape(str(row.get("sportsbook") or "n/a"))
        source_event = escape(_short_cell_id(row.get("source_event_id")))
        line_snapshot = escape(line_snapshot_display_id(row.get("line_snapshot_id")))
        updated_at = escape(_timestamp_cell(row.get("market_last_update")))
        rows.append(
            "<tr class='{}'>".format(" ".join(row_class))
            + f"<td><span class='strike-dot {'ok' if bool(row['cleared']) else 'bad'}'></span></td>"
            + f"<td><strong>{escape(str(row['pitcher']))}</strong></td>"
            + "<td>{} vs {} <span class='strike-pill'>{}HP</span></td>".format(
                escape(str(row["team"])),
                escape(str(row["opp"])),
                escape(str(row["hand"])),
            )
            + "<td><strong>{}</strong><br><span class='strike-dim'>event {}</span><br>"
            "<span class='strike-dim'>snap {}</span><br><span class='strike-dim'>{}</span></td>".format(
                sportsbook,
                source_event,
                line_snapshot,
                updated_at,
            )
            + f"<td>{_line_group_cell(row)}</td>"
            + f"<td class='strike-num'>{float(row['line']):.1f}</td>"
            + "<td><span class='strike-side {}'>{}</span></td>".format(
                side_class,
                escape(str(row["side"]).upper()),
            )
            + f"<td class='strike-num'>{format_pct(float(row['p_model']), digits=1)}</td>"
            + f"<td class='strike-num strike-dim'>{format_pct(float(row['p_market']), digits=1)}</td>"
            + f"<td class='strike-num'>{escape(odds_label)}</td>"
            + "<td class='strike-num {}'>{}</td>".format(
                "strike-pos" if float(edge) >= 0.0 else "strike-neg",
                escape(edge_label),
            )
            + f"<td class='strike-num'>{float(row['kelly_units']):.2f}u</td>"
            + (
                "<td><div class='strike-progress'><span style='width:{:.1f}%;background:{}'></span></div></td>".format(
                    confidence_width,
                    "var(--accent)" if bool(row["cleared"]) else "var(--dim)",
                )
            )
            + f"<td>{escape(str(row['note']))}</td>"
            + "</tr>"
        )
    if not rows:
        return ""
    return (
        "<div class='strike-grid-wrap'>"
        "<table class='strike-grid'>"
        "<thead><tr>"
        "<th></th>"
        "<th style='text-align:left'>PITCHER</th>"
        "<th style='text-align:left'>MATCH</th>"
        "<th style='text-align:left'>SOURCE</th>"
        "<th style='text-align:left'>GROUP</th>"
        "<th>LINE</th>"
        "<th>SIDE</th>"
        "<th>P(MODEL)</th>"
        "<th>P(MKT)</th>"
        "<th>PRICE</th>"
        "<th>EDGE</th>"
        "<th>KELLY</th>"
        "<th>CONF</th>"
        "<th style='text-align:left'>NOTE</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def render_board_screen(
    *,
    st: object,
    board: pd.DataFrame,
    board_date: date | None,
    board_source: str | None,
    selected_pitcher_id: str | None,
    settings: DashboardSettings,
) -> None:
    """Render the board screen."""
    st.markdown(
        "<div class='strike-screen-head'>"
        "<div>"
        "<div class='strike-screen-title'>SLATE BOARD "
        f"<span class='strike-dim'>/ {escape(board_date.isoformat() if board_date else 'n/a')} / {len(board)} props</span>"
        "</div>"
        "<div class='strike-crumb'>"
        f"devig:{escape(settings.devig_method)} · edge≥{settings.edge_min:.1%} · "
        f"kelly×{settings.kelly_fraction:.2f} · bank {settings.bankroll_units:.0f}u"
        "</div>"
        "</div>"
        "<div class='strike-toolbar'>"
        f"<span class='strike-chip on'>{escape((board_source or 'no artifacts').replace('_', ' '))}</span>"
        f"<span class='strike-chip'>{settings.max_daily_exposure_units:.1f}u max daily exposure</span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    filter_columns = st.columns([1, 1, 3])
    side_filter = filter_columns[0].selectbox(
        "Side",
        ["ALL", "OVER", "UNDER"],
        index=0,
        key="board_side_filter",
    )
    show_rejected = filter_columns[1].toggle(
        "Show rejected",
        value=False,
        key="board_show_rejected",
    )
    display_mode = filter_columns[2].radio(
        "Rows",
        ["All line rows", "Grouped by pitcher"],
        index=0,
        horizontal=True,
        key="board_display_mode",
    )
    query = st.text_input(
        "Pitcher, team, or book",
        value="",
        placeholder="filter pitcher/team...",
        key="board_query",
    )

    metrics = current_slate_metrics(board)
    st.markdown(
        kpi_strip_html(
            [
                {"label": "plays cleared", "value": str(metrics["plays_cleared"])},
                {
                    "label": "total stake",
                    "value": f"{metrics['total_stake_units']:.2f}u",
                },
                {
                    "label": "expected units",
                    "value": f"{metrics['expected_units']:.2f}u",
                    "tone": "pos" if metrics["expected_units"] >= 0.0 else "neg",
                },
                {
                    "label": "avg edge",
                    "value": format_pct(metrics["avg_edge"], digits=1),
                },
                {
                    "label": "model",
                    "value": str(metrics["model_name"]),
                    "size": "sm",
                },
            ]
        ),
        unsafe_allow_html=True,
    )

    if board.empty:
        st.markdown(
            "<div class='strike-empty'>"
            "No <code>daily_candidates</code>, <code>edge_candidates</code>, or historical "
            "<code>walk_forward_backtest</code> reporting artifacts were found for this slate yet."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    filtered = board.copy()
    if side_filter != "ALL":
        filtered = filtered[filtered["side"].str.upper() == side_filter]
    if not show_rejected:
        filtered = filtered[filtered["cleared"]]
    if query:
        mask = (
            filtered["pitcher"].str.contains(query, case=False, na=False)
            | filtered["team"].str.contains(query, case=False, na=False)
            | filtered["opp"].str.contains(query, case=False, na=False)
            | filtered["sportsbook"].str.contains(query, case=False, na=False)
        )
        filtered = filtered[mask]

    if filtered.empty:
        st.markdown(
            "<div class='strike-empty'>No board rows matched the current filters.</div>",
            unsafe_allow_html=True,
        )
        return

    visible_rows = (
        group_board_by_pitcher(filtered)
        if display_mode == "Grouped by pitcher"
        else filtered
    )
    if display_mode == "Grouped by pitcher":
        st.markdown(
            "<div class='strike-crumb'>Grouped by pitcher: showing the current best "
            f"{len(visible_rows)} rows from {len(filtered)} book/line rows.</div>",
            unsafe_allow_html=True,
        )

    detail_rows = visible_rows.drop_duplicates("pitcher_id")
    pitcher_options = [str(row["pitcher_id"]) for _, row in detail_rows.iterrows()]
    pitcher_labels = {
        str(row["pitcher_id"]): (
            f"{row['pitcher']} · {str(row['side']).upper()} {float(row['line']):.1f} "
            f"· {row['note']}"
        )
        for _, row in detail_rows.iterrows()
    }
    detail_columns = st.columns([3, 1])
    selected_pitcher = detail_columns[0].selectbox(
        "Pitcher detail",
        pitcher_options,
        format_func=lambda pitcher_id: pitcher_labels[pitcher_id],
        key="board_pitcher_detail_select",
    )
    if detail_columns[1].button(
        "Open pitcher",
        key="board_open_pitcher_detail",
        type="secondary",
        use_container_width=True,
    ):
        set_dashboard_query_params(
            st,
            screen="pitcher",
            board_date=board_date,
            pitcher_id=selected_pitcher,
        )
        rerun_streamlit(st)

    st.markdown(
        _board_table_html(
            visible_rows,
            selected_pitcher_id=selected_pitcher_id,
        ),
        unsafe_allow_html=True,
    )
