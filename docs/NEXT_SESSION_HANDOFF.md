# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current issue branch:
  `dylanmccavitt2015/age-152-add-clv-roi-and-edge-bucket-reporting`
- `main` already includes the merged `AGE-143` docs work, `AGE-144` MLB
  metadata ingest, `AGE-145` sportsbook ingest, `AGE-146` Statcast feature
  tables, `AGE-147` starter strikeout baseline mean training, `AGE-148`
  count-distribution ladder probabilities, `AGE-149` probability calibration
  diagnostics, `AGE-150` replayable edge-candidate pricing rows, and `AGE-151`
  walk-forward backtest joins
- This branch adds `AGE-152`: chart-ready CLV, ROI, and edge-bucket reporting
  artifacts on top of the saved walk-forward backtest run

## What Was Completed In AGE-152

- `src/mlb_props_stack/backtest.py`
  - keeps `backtest_bets.jsonl` as the raw auditable backtest artifact
  - extends `WalkForwardBacktestResult` with reporting artifact paths
  - adds flat reporting helpers that derive:
    - `bet_reporting.jsonl`
    - `clv_summary.jsonl`
    - `roi_summary.jsonl`
    - `edge_bucket_summary.jsonl`
  - keeps CLV separate from realized ROI while making paper result vs
    market-beating result directly filterable
  - preserves daily plus overall reporting rows for CLV and ROI
  - keeps run-level `backtest_runs.jsonl` as the summary envelope and now
    includes pointers to the reporting artifacts
- `src/mlb_props_stack/cli.py`
  - extends the backtest CLI summary to print the new reporting artifact paths
- `tests/test_backtest.py`
  - verifies the new reporting tables on seeded backtest inputs
  - locks the split between raw backtest rows and flat dashboard rows
  - checks CLV, ROI, and edge-bucket summaries for both populated and empty
    placed-bet cases
- `tests/test_cli.py`
  - verifies the CLI summary now includes the new artifact paths
- `README.md`, `docs/architecture.md`, `docs/modeling.md`
  - document the new reporting outputs and their intended downstream use

## Verification Run

These commands were run successfully during AGE-152:

```bash
uv sync --extra dev
python3 -m compileall src tests
uv run pytest
uv run python -m mlb_props_stack
```

Local results:

- `uv run pytest`
  - `50 passed`
- `uv run python -m mlb_props_stack`
  - still prints the runtime summary cleanly
- seeded local demo backtest run
  - generated `bet_reporting.jsonl`, `clv_summary.jsonl`, `roi_summary.jsonl`,
    and `edge_bucket_summary.jsonl` under a temp `walk_forward_backtest` run

## Recommended Next Issue

- Handle moved-point closing-line references when the exact strikeout line
  disappears near first pitch

Why this should go next:

- `AGE-152` now gives daily and overall reporting on exact-line CLV and realized
  ROI, but CLV still falls back to missing when no same-line close snapshot
  exists
- the reporting layer is now in place, so the next slice can improve close-line
  reference quality without reopening the summary-table contract

## Constraints For The Next Worktree

- Start the next issue worktree from the current `main` after AGE-152 is
  merged.
- Keep the current cutoff rule:
  - select the latest exact-line snapshot with
    `captured_at <= commence_time - cutoff`
- Keep walk-forward calibration honest:
  - use `raw_vs_calibrated_probabilities.jsonl` for held-out headline backtest
    rows
  - do not fall back to the production calibrator inside
    `ladder_probabilities.jsonl` for reported CLV or ROI
- Preserve the current artifact split:
  - `backtest_bets.jsonl` for raw auditable row detail
  - `bet_reporting.jsonl` for flat dashboard consumption
  - `clv_summary.jsonl`, `roi_summary.jsonl`, and `edge_bucket_summary.jsonl`
    for chart-level rollups
- Keep `line_snapshot_id`, `feature_row_id`, `lineup_snapshot_id`, and
  `outcome_id` traceable on every non-skipped evaluated row.

## Open Questions

- How should moved-line close references be normalized when the later market is
  at `6.5` but the bet was placed at `5.5`?
- `projection_generated_at` still defaults to `features_as_of` for historical
  rows. A future issue may want a dedicated inference artifact that persists a
  separate pregame projection timestamp.
