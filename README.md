# MLB Props Stack

An MLB props modeling stack focused on narrow, measurable sportsbook markets.

V1 is intentionally scoped to `pitcher strikeout props`, because strikeout outcomes
map cleanly to pitch-level process signals:

- pitcher whiff, CSW, zone, pitch mix, velocity, movement
- hitter strikeout tendencies by handedness
- expected batters faced, leash, and bullpen context
- park, weather, lineup, and umpire adjustments

## Why this repo exists

The goal is not to "predict baseball" in the abstract. The goal is to:

1. estimate event probabilities honestly
2. compare those probabilities to sportsbook prices
3. filter for edges that survive vig, variance, and bad calibration
4. backtest with the same information that would have been available at bet time

That is a math-and-process problem first, and a betting problem second.

## V1 Scope

- one market: pitcher strikeout props
- one modeling target: `P(K >= line + 0.5)` and adjacent ladder probabilities
- one output: edge-ranked candidate props with sizing guidance
- one evaluation loop: walk-forward historical backtest against real book lines

## Non-Goals

- full-game sides and totals
- same-game parlay optimization
- live execution bots
- reinforcement learning in the first modeling pass
- any claim that model outputs alone are "passive income"

## Stack Shape

- `src/mlb_props_stack/config.py`
  Runtime settings and model defaults.
- `src/mlb_props_stack/tracking.py`
  Reserved MLflow-compatible tracking config for later experiment logging.
- `src/mlb_props_stack/pricing.py`
  Odds conversion, expected value, devig, and Kelly sizing.
- `src/mlb_props_stack/markets.py`
  Core data models for props, projections, and decisions.
- `src/mlb_props_stack/edge.py`
  Edge detection and candidate ranking.
- `src/mlb_props_stack/backtest.py`
  Backtest policy and evaluation guardrails.
- `src/mlb_props_stack/dashboard/app.py`
  Placeholder module where the future Streamlit dashboard will live.
- `docs/architecture.md`
  Product and system architecture.
- `docs/modeling.md`
  Data, features, targets, and validation rules.

## Recommended Build Order

1. data contracts for games, pitchers, lines, and line moves
2. feature pipeline from Statcast + schedule + lineups
3. strikeout distribution model
4. calibration layer
5. pricing and edge detection
6. walk-forward backtest
7. paper-trading / tracking dashboard
8. later: RL for sizing or timing, not base prediction

## Development

This repo currently targets Python 3.9+ and uses a small standard-library-first
scaffold until data connectors and model dependencies are added.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python -m mlb_props_stack
```

If you already use `uv`, the equivalent commands are:

```bash
uv sync --dev
uv run pytest
uv run python -m mlb_props_stack
```

## Future Hooks

- `mlb_props_stack.tracking.TrackingConfig` is the reserved place for future
  MLflow tracking configuration.
- `mlb_props_stack.dashboard.app` is the reserved dashboard entrypoint for a
  later Streamlit UI.
- Neither MLflow nor Streamlit is installed in v1; the baseline only preserves
  clean seams for those additions.

## Risk

This project should be treated as a research and decision-support system.
Sportsbooks price efficiently enough that sloppy data handling, leakage, bad
calibration, and untracked line movement can erase a paper edge very quickly.
