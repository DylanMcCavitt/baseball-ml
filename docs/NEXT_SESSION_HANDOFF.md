# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current issue branch:
  `dylanmccavitt2015/age-151-build-walk-forward-backtest-with-timestamp-safe-joins`
- `main` already includes the merged `AGE-143` docs work, `AGE-144` MLB
  metadata ingest, `AGE-145` sportsbook ingest, `AGE-146` Statcast feature
  tables, `AGE-147` starter strikeout baseline mean training, `AGE-148`
  count-distribution ladder probabilities, `AGE-149` probability calibration
  diagnostics, and `AGE-150` replayable edge-candidate pricing rows
- This branch adds `AGE-151`: the first walk-forward backtest slice with
  cutoff-safe odds selection, honest held-out probability joins, and row-level
  freshness audits

## What Was Completed In AGE-151

- `src/mlb_props_stack/backtest.py`
  - keeps `BacktestPolicy` and `BACKTEST_CHECKLIST`
  - adds `WalkForwardBacktestResult`
  - adds `build_walk_forward_backtest()` which:
    - replays all saved AGE-145 odds runs for each requested official date
    - groups exact book lines by date, book, event, player, market, and line
    - selects the latest exact-line snapshot at or before the configured
      pregame cutoff
    - joins selected rows to the saved AGE-147 training dataset,
      AGE-149/150 held-out probability rows, and same-game starter outcomes
    - writes:
      - `data/normalized/walk_forward_backtest/start=..._end=.../run=.../backtest_bets.jsonl`
      - `data/normalized/walk_forward_backtest/start=..._end=.../run=.../backtest_runs.jsonl`
      - `data/normalized/walk_forward_backtest/start=..._end=.../run=.../join_audit.jsonl`
    - preserves late-only snapshots, training-split rows, and missing-join or
      missing-outcome rows as explicit skipped statuses
  - keeps CLV separate from realized ROI
  - uses held-out probabilities from
    `raw_vs_calibrated_probabilities.jsonl` instead of the production
    calibrator stored in `ladder_probabilities.jsonl`
- `src/mlb_props_stack/cli.py`
  - adds `build-walk-forward-backtest`
  - renders the new backtest summary including run id, model run id, cutoff,
    and output artifact paths
- `src/mlb_props_stack/config.py`
  - adds `backtest_cutoff_minutes_before_first_pitch` with a default of `30`
- `tests/test_backtest.py`
  - covers deterministic replay on seeded inputs
  - verifies latest-pre-cutoff snapshot selection
  - verifies late-only snapshots are rejected
  - verifies train-split rows are preserved as skipped
  - verifies backtest rows and join-audit rows preserve feature, lineup, and
    outcome traceability
- `tests/test_cli.py`
  - adds CLI coverage for `build-walk-forward-backtest`
- `README.md`, `docs/architecture.md`, `docs/modeling.md`
  - document the new backtest command
  - document `backtest_bets.jsonl`, `backtest_runs.jsonl`, and `join_audit.jsonl`
  - call out the walk-forward calibration rule and cutoff-safe snapshot
    selection rule

## Verification Run

These commands were run successfully during AGE-151:

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

## Recommended Next Issue

- Expand backtest reporting to handle moved-point closing-line references and
  richer per-date rollups

Why this should go next:

- `AGE-151` produces the first honest exact-line CLV and ROI rows, but CLV is
  still limited to cases where the same exact line remains available near first
  pitch
- the new backtest artifacts now make it practical to add better close-line
  matching and more operator-friendly reporting without reopening the core
  cutoff-safe join logic

## Constraints For The Next Worktree

- Start the next issue worktree from the current `main` after AGE-151 is
  merged.
- Keep the current cutoff rule:
  - select the latest exact-line snapshot with
    `captured_at <= commence_time - cutoff`
- Keep walk-forward calibration honest:
  - use `raw_vs_calibrated_probabilities.jsonl` for held-out headline backtest
    rows
  - do not fall back to the production calibrator inside
    `ladder_probabilities.jsonl` for reported CLV or ROI
- Preserve skipped rows and audit rows for:
  - late-only snapshots
  - train-split projections
  - missing joins or missing outcomes
- Keep `line_snapshot_id`, `feature_row_id`, `lineup_snapshot_id`, and
  `outcome_id` traceable on every non-skipped evaluated row.

## Open Questions

- How should CLV be defined when the same exact strikeout line disappears and
  only moved-point alternatives remain near first pitch?
- `projection_generated_at` still defaults to `features_as_of` for historical
  rows. A future issue may want a dedicated inference artifact that persists a
  separate pregame projection timestamp.
