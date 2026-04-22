# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Last completed issue branch: `feat/dashboard-ui-strike-ops`
- This slice delivers the first real Strike Ops dashboard workbench and adds a
  historical replay path so model iteration can happen off saved artifacts
- Current status: dashboard scaffolding is in place, but the local checkout
  still needs saved historical odds snapshots under `data/normalized/the_odds_api/`
  before historical replay can populate real board rows

## What Was Completed In This Slice

- `src/mlb_props_stack/dashboard/app.py`
  - replaces the placeholder with the multi-screen Streamlit Strike Ops shell
  - wires board, pitcher, backtest, registry, features, and config screens
  - persists dashboard controls to `user_config.toml`
- `src/mlb_props_stack/dashboard/lib/`
  - adds artifact-backed loaders, Plotly helpers, local MLflow helpers, and
    theme assets
  - board loading now falls back from `daily_candidates` -> `edge_candidates`
    -> latest `walk_forward_backtest` reporting rows
  - historical replay now labels the ticker as `HIST REPLAY` when the board is
    sourced from backtest artifacts
- `src/mlb_props_stack/dashboard/screens/`
  - adds the board, pitcher drill-down, backtest, registry, feature, and
    config screens
- `app.py` and `.streamlit/config.toml`
  - add the repo-root Streamlit entrypoint and local theme config
- `README.md` and `docs/architecture.md`
  - document the Strike Ops dashboard shape and the historical replay workflow
- `tests/test_dashboard_data.py`
  - adds focused coverage for replaying a historical board from backtest
    artifacts
- `tests/test_runtime_smokes.py` and `tests/test_paper_tracking.py`
  - update the dashboard runtime smoke coverage for the new multi-screen UI

## Files Changed

- `.streamlit/config.toml`
- `README.md`
- `app.py`
- `docs/architecture.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `pyproject.toml`
- `src/mlb_props_stack/dashboard/__init__.py`
- `src/mlb_props_stack/dashboard/app.py`
- `src/mlb_props_stack/dashboard/lib/__init__.py`
- `src/mlb_props_stack/dashboard/lib/data.py`
- `src/mlb_props_stack/dashboard/lib/mlflow_io.py`
- `src/mlb_props_stack/dashboard/lib/plots.py`
- `src/mlb_props_stack/dashboard/lib/theme.py`
- `src/mlb_props_stack/dashboard/screens/__init__.py`
- `src/mlb_props_stack/dashboard/screens/backtest.py`
- `src/mlb_props_stack/dashboard/screens/board.py`
- `src/mlb_props_stack/dashboard/screens/config.py`
- `src/mlb_props_stack/dashboard/screens/features.py`
- `src/mlb_props_stack/dashboard/screens/pitcher.py`
- `tests/test_dashboard_data.py`
- `tests/test_paper_tracking.py`
- `tests/test_runtime_smokes.py`
- `user_config.toml`
- `uv.lock`

## Verification

Commands run successfully on April 22, 2026:

```bash
uv sync --extra dev
uv run pytest tests/test_dashboard_data.py
uv run pytest tests/test_runtime_smokes.py
uv run pytest tests/test_paper_tracking.py
uv run pytest
uv run python -m mlb_props_stack
python3 -m compileall src tests app.py
```

Observed results:

- targeted dashboard tests passed
- full test suite passed: `69 passed`
- `uv run python -m mlb_props_stack`
  - printed the runtime summary successfully
- `python3 -m compileall src tests app.py`
  - completed successfully

## Recommended Next Issue

- Build or backfill the saved historical odds snapshot set under
  `data/normalized/the_odds_api/` and verify the dashboard historical replay
  loop against real replayable dates

Why this should go next:

- the dashboard can now replay historical backtest dates without a fresh Odds
  API call, but only if saved historical line snapshots already exist locally
- this checkout still has model and backtest summaries but no local
  `the_odds_api`, `daily_candidates`, or `edge_candidates` artifact trees
- the next useful milestone is a real `train -> backtest -> dashboard replay`
  workflow on saved historical pitcher strikeout markets

## Constraints And Open Questions

- The board replay fallback currently uses the latest walk-forward backtest run;
  if multiple backtest windows should be browsable side by side later, add a
  backtest-run selector rather than overloading the slate-date picker
- Historical replay is intentionally a fallback. Live board artifacts still
  take precedence when `daily_candidates` or `edge_candidates` exist
- The local checkout at handoff time does not include saved
  `data/normalized/the_odds_api/...` artifacts, so the board may still show
  empty states until those historical line snapshots are available
