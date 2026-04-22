"""Plotly chart helpers for the Streamlit dashboard."""

from __future__ import annotations

from math import sqrt

from plotly import graph_objects as go
from plotly.subplots import make_subplots

from .theme import COLORS, PLOTLY_TEMPLATE


def _base_figure() -> go.Figure:
    figure = go.Figure()
    figure.update_layout(template=PLOTLY_TEMPLATE)
    return figure


def pmf_fig(*, pmf_rows: list[dict[str, float | str]], line: float) -> go.Figure:
    """Return the strikeout PMF chart."""
    labels = [str(row["label"]) for row in pmf_rows]
    values = [float(row["p"]) for row in pmf_rows]
    colors = [str(row["color"]) for row in pmf_rows]
    figure = _base_figure()
    figure.add_bar(
        x=labels,
        y=values,
        marker={"color": colors},
        hovertemplate="K=%{x}<br>P=%{y:.2%}<extra></extra>",
    )
    if labels:
        marker_index = min(max(int(line + 0.5), 0), len(labels) - 1)
        figure.add_vline(
            x=marker_index,
            line_dash="dash",
            line_color=COLORS["accent"],
            annotation_text=f"line {line:.1f}",
            annotation_font={"color": COLORS["accent"], "family": "JetBrains Mono"},
        )
    figure.update_layout(
        height=280,
        bargap=0.12,
        showlegend=False,
        yaxis={"tickformat": ".0%", "title": "P(K=k)"},
        xaxis={"title": "Strikeouts"},
    )
    return figure


def probability_comparison_fig(
    *,
    model_probability: float,
    market_probability: float,
    edge_probability: float,
) -> go.Figure:
    """Return the model-vs-market comparison bars."""
    figure = _base_figure()
    figure.add_bar(
        x=[model_probability, market_probability, abs(edge_probability)],
        y=["p_model", "p_market", "edge"],
        orientation="h",
        marker={
            "color": [
                COLORS["pos"],
                COLORS["blue"],
                COLORS["accent"] if edge_probability >= 0.0 else COLORS["neg"],
            ]
        },
        hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    )
    figure.update_layout(
        height=220,
        showlegend=False,
        margin={"l": 88, "r": 20, "t": 12, "b": 18},
        xaxis={"tickformat": ".0%", "range": [0, 1]},
        yaxis={"categoryorder": "array", "categoryarray": ["edge", "p_market", "p_model"]},
    )
    return figure


def calibration_fig(calibration_rows: list[dict[str, float | int]]) -> go.Figure:
    """Return the reliability plot."""
    figure = _base_figure()
    if not calibration_rows:
        figure.update_layout(height=300)
        return figure
    max_sample = max(int(row["n"]) for row in calibration_rows)
    sizes = [
        max(8.0, 28.0 * sqrt(int(row["n"]) / max_sample))
        for row in calibration_rows
    ]
    figure.add_scatter(
        x=[float(row["pred_bin"]) for row in calibration_rows],
        y=[float(row["actual_rate"]) for row in calibration_rows],
        mode="markers+lines",
        marker={"size": sizes, "color": COLORS["accent"], "opacity": 0.9},
        line={"color": COLORS["accent"], "width": 1.5},
        hovertemplate=(
            "pred=%{x:.2%}<br>actual=%{y:.2%}<br>n=%{customdata}<extra></extra>"
        ),
        customdata=[int(row["n"]) for row in calibration_rows],
    )
    figure.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line={"color": COLORS["dim"], "dash": "dash"},
        hoverinfo="skip",
        showlegend=False,
    )
    figure.update_layout(
        height=300,
        showlegend=False,
        xaxis={"tickformat": ".0%", "title": "Predicted"},
        yaxis={"tickformat": ".0%", "title": "Actual", "range": [0, 1]},
    )
    return figure


def pnl_fig(series_rows: list[dict[str, float | str]]) -> go.Figure:
    """Return the walk-forward cumulative PnL and CLV chart."""
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.update_layout(template=PLOTLY_TEMPLATE)
    if not series_rows:
        figure.update_layout(height=320)
        return figure
    dates = [str(row["date"]) for row in series_rows]
    figure.add_scatter(
        x=dates,
        y=[float(row["cum_pnl_units"]) for row in series_rows],
        mode="lines",
        line={"color": COLORS["pos"], "width": 2},
        fill="tozeroy",
        fillcolor="rgba(46, 204, 113, 0.12)",
        name="Cum PnL",
        hovertemplate="%{x}<br>cum pnl=%{y:.2f}u<extra></extra>",
        secondary_y=False,
    )
    figure.add_scatter(
        x=dates,
        y=[float(row["clv_units"]) for row in series_rows],
        mode="lines",
        line={"color": COLORS["blue"], "width": 1.5, "dash": "dash"},
        name="CLV",
        hovertemplate="%{x}<br>cum clv=%{y:.3f}<extra></extra>",
        secondary_y=True,
    )
    figure.update_layout(height=320, hovermode="x unified", showlegend=False)
    figure.update_yaxes(title_text="Cum PnL (u)", secondary_y=False)
    figure.update_yaxes(title_text="Cum CLV delta", secondary_y=True)
    return figure


def scatter_fig(rows: list[dict[str, float]]) -> go.Figure:
    """Return the model-vs-market scatter."""
    figure = _base_figure()
    if not rows:
        figure.update_layout(height=300)
        return figure
    figure.add_scatter(
        x=[float(row["market_probability"]) for row in rows],
        y=[float(row["model_probability"]) for row in rows],
        mode="markers",
        marker={
            "size": 8,
            "color": [
                COLORS["pos"]
                if float(row["model_probability"]) > float(row["market_probability"])
                else COLORS["neg"]
                if float(row["model_probability"]) < float(row["market_probability"])
                else COLORS["dim"]
                for row in rows
            ],
            "opacity": 0.75,
        },
        hovertemplate=(
            "%{text}<br>p_market=%{x:.2%}<br>p_model=%{y:.2%}<extra></extra>"
        ),
        text=[str(row.get("label") or "") for row in rows],
        showlegend=False,
    )
    figure.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line={"color": COLORS["dim"], "dash": "dash"},
        hoverinfo="skip",
        showlegend=False,
    )
    figure.update_layout(
        height=300,
        xaxis={"tickformat": ".0%", "title": "p_market", "range": [0, 1]},
        yaxis={"tickformat": ".0%", "title": "p_model", "range": [0, 1]},
    )
    return figure


def roi_fig(series_rows: list[dict[str, float | str]]) -> go.Figure:
    """Return the rolling ROI sparkline chart."""
    figure = _base_figure()
    if not series_rows:
        figure.update_layout(height=180)
        return figure
    figure.add_scatter(
        x=[str(row["date"]) for row in series_rows],
        y=[float(row["roi"]) for row in series_rows],
        mode="lines",
        line={"color": COLORS["accent"], "width": 1.75},
        hovertemplate="%{x}<br>30d roi=%{y:.2%}<extra></extra>",
        showlegend=False,
    )
    figure.add_hline(y=0.0, line_color=COLORS["dim"], line_dash="dash")
    figure.update_layout(height=180, margin={"l": 42, "r": 16, "t": 18, "b": 28})
    figure.update_yaxes(tickformat=".0%", title="30d ROI")
    return figure


def feature_bars_fig(rows: list[dict[str, float | str]]) -> go.Figure:
    """Return the feature-importance bar chart."""
    figure = _base_figure()
    if not rows:
        figure.update_layout(height=300)
        return figure
    ordered = list(rows)[::-1]
    figure.add_bar(
        x=[float(row["importance"]) for row in ordered],
        y=[str(row["name"]) for row in ordered],
        orientation="h",
        marker={
            "color": [
                COLORS["pos"] if str(row["direction"]).startswith("+") else COLORS["neg"]
                for row in ordered
            ]
        },
        hovertemplate=(
            "%{y}<br>importance=%{x:.3f}<br>direction=%{customdata}<extra></extra>"
        ),
        customdata=[str(row["direction"]) for row in ordered],
        showlegend=False,
    )
    figure.update_layout(
        height=max(280, 36 * len(ordered)),
        margin={"l": 180, "r": 20, "t": 18, "b": 28},
    )
    figure.update_xaxes(title="Importance")
    figure.update_yaxes(title=None)
    return figure
