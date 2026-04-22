"""Theme constants and shared dashboard CSS."""

from __future__ import annotations

from html import escape


COLORS = {
    "bg": "#0a0b0d",
    "surface": "#111317",
    "surface_alt": "#16191e",
    "line": "#1e222a",
    "line_alt": "#262b35",
    "fg": "#d9dde3",
    "fg_alt": "#aeb4bd",
    "dim": "#6b7280",
    "accent": "#f5a524",
    "pos": "#2ecc71",
    "neg": "#ff4757",
    "blue": "#5aa9ff",
    "warn": "#f5a524",
}

SCREEN_LABELS = {
    "board": "BOARD",
    "pitcher": "PITCHER",
    "backtest": "BACKTEST",
    "registry": "MLFLOW",
    "features": "FEATURES",
    "config": "CONFIG",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "JetBrains Mono, ui-monospace, Menlo, monospace",
            "color": COLORS["fg"],
            "size": 11,
        },
        "xaxis": {
            "gridcolor": COLORS["line"],
            "zerolinecolor": COLORS["line"],
            "tickfont": {"color": COLORS["dim"]},
            "titlefont": {"color": COLORS["dim"]},
        },
        "yaxis": {
            "gridcolor": COLORS["line"],
            "zerolinecolor": COLORS["line"],
            "tickfont": {"color": COLORS["dim"]},
            "titlefont": {"color": COLORS["dim"]},
        },
        "hoverlabel": {
            "bgcolor": COLORS["surface_alt"],
            "bordercolor": COLORS["line_alt"],
            "font": {
                "family": "JetBrains Mono, ui-monospace, Menlo, monospace",
                "color": COLORS["fg"],
            },
        },
        "colorway": [
            COLORS["accent"],
            COLORS["pos"],
            COLORS["blue"],
            COLORS["neg"],
        ],
        "margin": {"l": 40, "r": 20, "t": 20, "b": 32},
    }
}


def build_theme_css() -> str:
    """Return the CSS injected into the Streamlit app."""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
  --bg: {COLORS["bg"]};
  --surface: {COLORS["surface"]};
  --surface2: {COLORS["surface_alt"]};
  --line: {COLORS["line"]};
  --line2: {COLORS["line_alt"]};
  --fg: {COLORS["fg"]};
  --fg2: {COLORS["fg_alt"]};
  --dim: {COLORS["dim"]};
  --accent: {COLORS["accent"]};
  --pos: {COLORS["pos"]};
  --neg: {COLORS["neg"]};
  --blue: {COLORS["blue"]};
  --warn: {COLORS["warn"]};
  --font-mono: "JetBrains Mono", ui-monospace, Menlo, monospace;
  --font-ui: "IBM Plex Sans", system-ui, sans-serif;
}}

[data-testid="stAppViewContainer"],
.stApp {{
  background: var(--bg);
  color: var(--fg);
}}

body {{
  font-family: var(--font-ui);
  font-size: 13px;
}}

[data-testid="stHeader"],
[data-testid="stToolbar"] {{
  background: transparent;
}}

[data-testid="stAppViewContainer"] > .main {{
  background: var(--bg);
}}

.block-container {{
  max-width: 1540px;
  padding-top: 0.75rem;
  padding-bottom: 2.5rem;
}}

.stMarkdown, .stAlert, .stText, label {{
  color: var(--fg);
}}

.stPlotlyChart {{
  border: 1px solid var(--line);
  background: var(--surface);
  padding: 0.35rem 0.5rem 0.15rem;
}}

.stPlotlyChart > div {{
  min-height: 0 !important;
}}

.strike-hide {{
  display: none;
}}

.strike-shell {{
  display: grid;
  gap: 0.9rem;
}}

.strike-ticker {{
  display: flex;
  align-items: center;
  gap: 18px;
  padding: 0.45rem 0.85rem;
  border: 1px solid var(--line);
  background: var(--surface);
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--fg2);
  white-space: nowrap;
  overflow-x: auto;
}}

.strike-brand {{
  color: var(--accent);
  font-weight: 600;
  letter-spacing: 0.12em;
}}

.strike-sep {{
  color: var(--line2);
}}

.strike-pill-live {{
  color: var(--pos);
}}

.strike-header {{
  display: grid;
  grid-template-columns: 260px 1fr auto;
  border: 1px solid var(--line);
  background: var(--surface);
}}

.strike-logo {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0.75rem 0.9rem;
  border-right: 1px solid var(--line);
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 0.14em;
}}

.strike-logo-mark {{
  width: 22px;
  height: 22px;
  display: grid;
  place-items: center;
  background: var(--accent);
  color: #111;
  font-weight: 700;
  clip-path: polygon(0 0, 100% 0, 100% 75%, 75% 100%, 0 100%);
}}

.strike-nav {{
  display: flex;
  flex-wrap: wrap;
}}

.strike-nav a {{
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  border-right: 1px solid var(--line);
  text-decoration: none;
  color: var(--fg2);
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.1em;
}}

.strike-nav a:hover {{
  background: var(--surface2);
  color: var(--fg);
}}

.strike-nav a.on {{
  color: var(--accent);
  box-shadow: inset 0 -2px 0 var(--accent);
}}

.strike-kbd {{
  color: var(--dim);
  font-size: 9px;
}}

.strike-header-right {{
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 0.75rem 0.9rem;
  border-left: 1px solid var(--line);
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--fg2);
}}

.strike-connected {{
  color: var(--pos);
  border: 1px solid var(--pos);
  padding: 0.1rem 0.45rem;
}}

.strike-screen-head {{
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 1rem;
  flex-wrap: wrap;
  margin-top: 0.35rem;
}}

.strike-screen-title {{
  font-family: var(--font-mono);
  font-size: 15px;
  letter-spacing: 0.08em;
  color: var(--fg);
  font-weight: 600;
}}

.strike-dim {{
  color: var(--dim);
}}

.strike-crumb {{
  margin-top: 0.15rem;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--dim);
}}

.strike-toolbar {{
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}}

.strike-chip {{
  display: inline-flex;
  align-items: center;
  border: 1px solid var(--line2);
  padding: 0.22rem 0.55rem;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.05em;
  color: var(--fg2);
  background: var(--surface);
}}

.strike-chip.on {{
  color: #111;
  background: var(--accent);
  border-color: var(--accent);
}}

.strike-kpi-strip {{
  display: grid;
  grid-auto-flow: column;
  grid-auto-columns: 1fr;
  border: 1px solid var(--line);
  background: var(--surface);
}}

.strike-kpi {{
  padding: 0.65rem 0.9rem;
  border-right: 1px solid var(--line);
}}

.strike-kpi:last-child {{
  border-right: none;
}}

.strike-kpi-label {{
  font-family: var(--font-mono);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--dim);
}}

.strike-kpi-value {{
  margin-top: 0.15rem;
  font-family: var(--font-mono);
  font-size: 20px;
  font-weight: 500;
  color: var(--fg);
}}

.strike-kpi-value.sm {{
  font-size: 13px;
}}

.strike-kpi-value.pos {{
  color: var(--pos);
}}

.strike-kpi-value.neg {{
  color: var(--neg);
}}

.strike-grid-wrap {{
  border: 1px solid var(--line);
  background: var(--surface);
  overflow-x: auto;
}}

.strike-grid {{
  width: 100%;
  border-collapse: collapse;
  font-family: var(--font-mono);
  font-size: 12px;
}}

.strike-grid th {{
  text-align: right;
  padding: 0.55rem 0.65rem;
  background: var(--surface2);
  border-bottom: 1px solid var(--line);
  color: var(--dim);
  font-size: 10px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}}

.strike-grid td {{
  padding: 0.5rem 0.65rem;
  border-bottom: 1px solid var(--line);
  color: var(--fg);
}}

.strike-grid tr.muted {{
  opacity: 0.45;
}}

.strike-grid tr.sel {{
  background: rgba(245, 165, 36, 0.08);
  box-shadow: inset 2px 0 0 var(--accent);
}}

.strike-grid a {{
  color: inherit;
  text-decoration: none;
}}

.strike-grid a:hover {{
  color: var(--accent);
}}

.strike-num {{
  text-align: right;
  font-variant-numeric: tabular-nums;
}}

.strike-pos {{
  color: var(--pos);
}}

.strike-neg {{
  color: var(--neg);
}}

.strike-blue {{
  color: var(--blue);
}}

.strike-side {{
  display: inline-block;
  border: 1px solid currentColor;
  padding: 0.08rem 0.4rem;
  font-size: 10px;
  letter-spacing: 0.08em;
}}

.strike-side.over {{
  color: var(--pos);
}}

.strike-side.under {{
  color: var(--blue);
}}

.strike-pill {{
  display: inline-block;
  border: 1px solid var(--line2);
  padding: 0.1rem 0.4rem;
  font-family: var(--font-mono);
  font-size: 9px;
  color: var(--dim);
  letter-spacing: 0.08em;
}}

.strike-pill.ok {{
  color: var(--pos);
  border-color: var(--pos);
}}

.strike-pill.warn {{
  color: var(--warn);
  border-color: var(--warn);
}}

.strike-pill.bad {{
  color: var(--neg);
  border-color: var(--neg);
}}

.strike-progress {{
  height: 8px;
  width: 80px;
  border: 1px solid var(--line2);
  background: var(--surface2);
}}

.strike-progress > span {{
  display: block;
  height: 100%;
}}

.strike-panel {{
  border: 1px solid var(--line);
  background: var(--surface);
  padding: 0.75rem 0.9rem 0.9rem;
}}

.strike-panel-head {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 0.5rem;
  margin-bottom: 0.65rem;
  padding-bottom: 0.35rem;
  border-bottom: 1px dashed var(--line);
}}

.strike-panel-head h4 {{
  margin: 0;
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.14em;
  color: var(--fg2);
}}

.strike-legend {{
  display: flex;
  gap: 0.9rem;
  flex-wrap: wrap;
  margin-top: 0.45rem;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--fg2);
}}

.strike-legend-box {{
  display: inline-block;
  width: 10px;
  height: 10px;
  margin-right: 0.3rem;
  vertical-align: -1px;
}}

.strike-mini-bars {{
  display: grid;
  gap: 0.45rem;
}}

.strike-mini-row {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) 88px 44px;
  gap: 0.5rem;
  align-items: center;
  font-family: var(--font-mono);
  font-size: 10px;
}}

.strike-mini-track {{
  height: 8px;
  border: 1px solid var(--line2);
  background: var(--surface2);
}}

.strike-mini-track > span {{
  display: block;
  height: 100%;
}}

.strike-guard-grid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.65rem;
}}

.strike-guard {{
  padding: 0.65rem;
  border: 1px solid var(--line2);
  background: var(--surface2);
}}

.strike-guard.bad {{
  border-color: var(--neg);
}}

.strike-guard-top {{
  display: flex;
  align-items: center;
  gap: 0.45rem;
  margin-bottom: 0.3rem;
}}

.strike-dot {{
  display: inline-block;
  width: 7px;
  height: 7px;
  background: var(--dim);
}}

.strike-dot.ok {{
  background: var(--pos);
  box-shadow: 0 0 6px var(--pos);
}}

.strike-dot.bad {{
  background: var(--neg);
}}

.strike-dot.warn {{
  background: var(--warn);
}}

.strike-statusbar {{
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  padding: 0.45rem 0.65rem;
  border: 1px solid var(--line);
  background: var(--surface);
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--fg2);
}}

.strike-empty {{
  border: 1px dashed var(--line2);
  padding: 1rem;
  background: var(--surface);
  color: var(--fg2);
}}

.strike-note-list {{
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  gap: 0.35rem;
  font-family: var(--font-mono);
  font-size: 11px;
}}

.strike-note-list li {{
  display: flex;
  align-items: center;
  gap: 0.45rem;
}}

.strike-formhint {{
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--dim);
}}

@media (max-width: 1100px) {{
  .strike-header {{
    grid-template-columns: 1fr;
  }}

  .strike-logo,
  .strike-header-right {{
    border: none;
    border-bottom: 1px solid var(--line);
  }}

  .strike-nav a {{
    flex: 1 0 auto;
  }}

  .strike-kpi-strip {{
    grid-auto-flow: row;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}

  .strike-guard-grid {{
    grid-template-columns: 1fr;
  }}
}}
</style>
"""


def kpi_strip_html(metrics: list[dict[str, str]]) -> str:
    """Render the custom KPI strip."""
    items: list[str] = []
    for metric in metrics:
        value_class = "strike-kpi-value"
        tone = metric.get("tone")
        size = metric.get("size")
        if tone:
            value_class += f" {escape(tone)}"
        if size == "sm":
            value_class += " sm"
        items.append(
            "<div class='strike-kpi'>"
            f"<div class='strike-kpi-label'>{escape(metric['label'])}</div>"
            f"<div class='{value_class}'>{escape(metric['value'])}</div>"
            "</div>"
        )
    return "<div class='strike-kpi-strip'>" + "".join(items) + "</div>"
