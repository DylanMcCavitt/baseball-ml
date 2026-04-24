"""Feature inspection screen renderer."""

from __future__ import annotations

from datetime import date
from html import escape

import pandas as pd

from ..lib.data import (
    DashboardSettings,
    feature_drift_rows,
    get_feature_importance,
    get_optional_feature_diagnostics,
    leakage_checks,
)
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


def _feature_list_html(features: list[str]) -> str:
    if not features:
        return "<span class='strike-dim'>none</span>"
    return "".join(
        f"<span class='strike-chip'>{escape(feature)}</span>"
        for feature in features
    )


def _status_class(status: str) -> str:
    if status == "active":
        return "ok"
    if status in {"missing_source", "excluded_below_coverage", "excluded_low_variance"}:
        return "warn"
    return "bad"


def _schema_summary_html(diagnostics: dict) -> str:
    active_model_run_id = diagnostics.get("active_model_run_id") or "n/a"
    source_model_run_id = diagnostics.get("source_model_run_id") or "n/a"
    target_date = diagnostics.get("target_date") or "n/a"
    return (
        "<div class='strike-kpi-strip'>"
        "<div class='strike-kpi'>"
        "<div class='strike-kpi-label'>active run</div>"
        f"<div class='strike-kpi-value sm'>{escape(str(active_model_run_id))}</div>"
        "</div>"
        "<div class='strike-kpi'>"
        "<div class='strike-kpi-label'>source run</div>"
        f"<div class='strike-kpi-value sm'>{escape(str(source_model_run_id))}</div>"
        "</div>"
        "<div class='strike-kpi'>"
        "<div class='strike-kpi-label'>target date</div>"
        f"<div class='strike-kpi-value sm'>{escape(str(target_date))}</div>"
        "</div>"
        "<div class='strike-kpi'>"
        "<div class='strike-kpi-label'>encoded features</div>"
        f"<div class='strike-kpi-value'>{int(diagnostics.get('encoded_feature_count') or 0)}</div>"
        "</div>"
        "<div class='strike-kpi'>"
        "<div class='strike-kpi-label'>active optional</div>"
        f"<div class='strike-kpi-value'>{int(diagnostics.get('optional_feature_count') or 0)}</div>"
        "</div>"
        "</div>"
    )


def _family_rows_html(rows: list[dict]) -> str:
    if not rows:
        return "<div class='strike-empty'>No optional-feature diagnostics were found.</div>"
    parts = [
        "<div class='strike-grid-wrap'><table class='strike-grid'>"
        "<thead><tr>"
        "<th style='text-align:left'>Family</th>"
        "<th>Status</th>"
        "<th>Active</th>"
        "<th>Source Train</th>"
        "<th>Target Date</th>"
        "<th style='text-align:left'>Reason</th>"
        "<th style='text-align:left'>Schema</th>"
        "</tr></thead><tbody>"
    ]
    for row in rows:
        status = str(row["status"])
        active_features = row.get("active_features") or []
        schema_notes = row.get("schema_notes") or []
        parts.append(
            "<tr>"
            f"<td style='text-align:left'>{escape(str(row['label']))}</td>"
            "<td>"
            f"<span class='strike-pill {_status_class(status)}'>{escape(status.replace('_', ' '))}</span>"
            "</td>"
            f"<td>{escape(str(len(active_features)))}</td>"
            f"<td>{escape(str(row.get('source_train_coverage_label') or 'n/a'))}</td>"
            f"<td>{escape(str(row.get('target_coverage_label') or 'n/a'))}</td>"
            f"<td style='text-align:left'>{escape(str(row.get('reason') or 'n/a'))}</td>"
            f"<td style='text-align:left'>{escape('; '.join(schema_notes) if schema_notes else 'current')}</td>"
            "</tr>"
        )
    parts.append("</tbody></table></div>")
    return "".join(parts)


def render_features_screen(
    *,
    st: object,
    output_root: object,
    board: pd.DataFrame,
    board_date: date | None,
    settings: DashboardSettings,
) -> None:
    """Render the feature inspection screen."""
    importance = get_feature_importance(output_root)
    diagnostics = get_optional_feature_diagnostics(
        output_root,
        target_date=board_date,
        run_id=settings.active_run_id,
    )
    drift = feature_drift_rows(output_root)
    checks = leakage_checks(board, settings=settings)
    st.markdown(
        "<div class='strike-screen-head'>"
        "<div>"
        "<div class='strike-screen-title'>FEATURE INSPECTION "
        "<span class='strike-dim'>/ schema · optional coverage · drift</span></div>"
        "<div class='strike-crumb'>current model · starter-strikeout baseline</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='strike-panel-head'><h4>ACTIVE MODEL SCHEMA</h4>"
        "<span class='strike-dim'>encoded feature selection from artifact</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(_schema_summary_html(diagnostics), unsafe_allow_html=True)
    schema_columns = st.columns([1, 1], gap="medium")
    with schema_columns[0]:
        st.markdown(
            "<div class='strike-panel-head'><h4>ACTIVE CORE FEATURES</h4></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            _feature_list_html(list(diagnostics.get("active_core_features") or [])),
            unsafe_allow_html=True,
        )
    with schema_columns[1]:
        st.markdown(
            "<div class='strike-panel-head'><h4>ACTIVE OPTIONAL FEATURES</h4></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            _feature_list_html(list(diagnostics.get("active_optional_features") or [])),
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='strike-panel-head'><h4>OPTIONAL FEATURE DIAGNOSTICS</h4>"
        "<span class='strike-dim'>source training run · current target date</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(_family_rows_html(list(diagnostics.get("family_rows") or [])), unsafe_allow_html=True)

    columns = st.columns([2, 1], gap="medium")
    with columns[0]:
        st.markdown(
            "<div class='strike-panel-head'><h4>GLOBAL IMPORTANCE</h4>"
            "<span class='strike-dim'>mean |importance| · sign from coefficient direction</span></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            feature_bars_fig(importance.to_dict("records")),
            width="stretch",
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
