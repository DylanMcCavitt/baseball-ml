"""MLflow registry screen renderer."""

from __future__ import annotations

from html import escape

from ..lib.data import DashboardSettings, format_pct, registry_rows


def _runs_table_html(rows: list[dict]) -> str:
    parts: list[str] = []
    for row in rows:
        stage = str(row["stage"]).lower()
        created = row["created"].astimezone().strftime("%Y-%m-%d %H:%M") if row["created"] else "n/a"
        brier_label = "n/a" if row["brier"] is None else f"{float(row['brier']):.3f}"
        logloss_label = "n/a" if row["logloss"] is None else f"{float(row['logloss']):.3f}"
        stage_tone = (
            "ok"
            if stage == "production"
            else "warn"
            if stage == "staging"
            else "bad"
            if stage == "archived"
            else ""
        )
        parts.append(
            "<tr>"
            f"<td>{escape(str(row['name']))} <span class='strike-dim'>{escape(str(row['run_id']))}</span></td>"
            f"<td><span class='strike-pill {escape(stage_tone)}'>{escape(str(row['stage']))}</span></td>"
            f"<td class='strike-num'>{escape(brier_label)}</td>"
            f"<td class='strike-num'>{escape(logloss_label)}</td>"
            f"<td class='strike-num'>{format_pct(row['clv'], digits=2)}</td>"
            f"<td class='strike-num'>{format_pct(row['roi'], digits=1)}</td>"
            f"<td class='strike-num'>{int(row['n'])}</td>"
            f"<td>{escape(created)}</td>"
            "</tr>"
        )
    return (
        "<div class='strike-grid-wrap'><table class='strike-grid'>"
        "<thead><tr>"
        "<th style='text-align:left'>RUN</th><th>STAGE</th><th>BRIER</th><th>LOG-LOSS</th>"
        "<th>CLV</th><th>ROI</th><th>N</th><th style='text-align:left'>CREATED</th>"
        "</tr></thead><tbody>"
        + "".join(parts)
        + "</tbody></table></div>"
    )


def _diff_panel_html(run_a: dict, run_b: dict) -> str:
    rows = [
        ("brier", run_a.get("brier"), run_b.get("brier"), False, 3),
        ("log-loss", run_a.get("logloss"), run_b.get("logloss"), False, 3),
        ("clv%", run_a.get("clv"), run_b.get("clv"), True, 3),
        ("roi%", run_a.get("roi"), run_b.get("roi"), True, 3),
        ("n", run_a.get("n"), run_b.get("n"), True, 0),
    ]
    parts: list[str] = [
        "<div class='strike-panel'><div class='strike-panel-head'><h4>DIFF</h4>"
        "<span class='strike-dim'>A → B</span></div>"
        f"<div class='strike-statusbar'><span>A {escape(str(run_a['name']))}</span><span class='strike-sep'>│</span>"
        f"<span>B {escape(str(run_b['name']))}</span></div>"
    ]
    for label, value_a, value_b, higher_better, digits in rows:
        if value_a is None or value_b is None:
            delta_text = "n/a"
            tone = ""
        else:
            delta = float(value_b) - float(value_a)
            good = delta > 0 if higher_better else delta < 0
            tone = "strike-pos" if good else "strike-neg"
            delta_text = f"{delta:+.{digits}f}"
        left = "n/a" if value_a is None else f"{float(value_a):.{digits}f}"
        right = "n/a" if value_b is None else f"{float(value_b):.{digits}f}"
        parts.append(
            "<div class='strike-statusbar'>"
            f"<span>{escape(label)}</span><span>{escape(left)}</span>"
            "<span class='strike-sep'>→</span>"
            f"<span>{escape(right)}</span>"
            f"<span class='{tone}'>{escape(delta_text)}</span>"
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def render_registry_screen(
    *,
    st: object,
    output_root: object,
    settings: DashboardSettings,
    stage_overrides: dict[str, str],
    on_stage_change: object,
) -> None:
    """Render the registry screen."""
    rows = registry_rows(output_root, settings=settings, stage_overrides=stage_overrides)
    st.markdown(
        "<div class='strike-screen-head'>"
        "<div>"
        f"<div class='strike-screen-title'>MLFLOW REGISTRY <span class='strike-dim'>/ {len(rows)} runs</span></div>"
        "<div class='strike-crumb'>experiment: mlb-props-stack-starter-strikeout-training · compare up to 2 runs</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    if not rows:
        st.markdown(
            "<div class='strike-empty'>No model evaluation summaries were found for the registry view.</div>",
            unsafe_allow_html=True,
        )
        return

    row_lookup = {str(row["run_id"]): row for row in rows}
    default_selection = [str(row["run_id"]) for row in rows[:2]]
    selected = st.multiselect(
        "Compare runs (up to 2)",
        options=list(row_lookup.keys()),
        default=default_selection,
        max_selections=2,
        format_func=lambda run_id: f"{row_lookup[run_id]['name']} · {run_id}",
        key="registry_selection",
    )
    st.markdown(_runs_table_html(rows), unsafe_allow_html=True)

    if len(selected) == 2:
        run_a = row_lookup[selected[0]]
        run_b = row_lookup[selected[1]]
        st.markdown(_diff_panel_html(run_a, run_b), unsafe_allow_html=True)
        action_columns = st.columns(2)
        if action_columns[0].button("promote B -> Production", key="registry_promote"):
            message = on_stage_change(run_b, "Production")
            st.success(message)
        if action_columns[1].button("archive A", key="registry_archive"):
            message = on_stage_change(run_a, "Archived")
            st.success(message)
    else:
        st.markdown(
            "<div class='strike-empty'>Select two runs to open the diff panel.</div>",
            unsafe_allow_html=True,
        )
