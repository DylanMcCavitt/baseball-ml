# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Active issue branch: `feat/age-190-diagnose-missing-join-keys`
- This slice patches the walk-forward backtest join diagnostics so unresolved
  sportsbook rows no longer collapse into one generic `missing_join_keys`
  bucket
- Verification in this worktree used copied local normalized inputs from the
  canonical checkout under `/Users/dylanmccavitt/projects/nba-ml/data/normalized/`
  because the tracked `20260422T205727Z` and `20260422T205734Z` artifact folders
  in git only carried summary files, not the full join inputs

## What Was Completed In This Slice

- `src/mlb_props_stack/backtest.py`
  - classifies missing snapshot joins more precisely before the held-out
    probability lookup, including `unmatched_event_mapping`,
    `unresolved_pitcher_identity`, `missing_game_mapping`, and the existing
    downstream skip statuses
  - accumulates `skip_reason_counts` at run level and writes them into
    `WalkForwardBacktestResult` plus `backtest_runs.jsonl`
- `src/mlb_props_stack/cli.py`
  - prints `skip_reason_counts` in the walk-forward backtest CLI summary so
    zero-bet windows surface the real failure mode immediately
- `tests/test_backtest.py`
  - extends the backtest contract checks to assert run-level skip reason counts
  - adds a focused regression test with one stale unmatched odds row, one true
    `missing_line_probability` row, one below-threshold scored row, and one
    actionable row so scored-vs-skipped behavior stays explicit
- `tests/test_cli.py`
  - updates the CLI summary test to cover the new skip-reason output
- `docs/architecture.md`
  - documents that backtest rows now preserve explicit skipped-by-reason
    statuses and that `backtest_runs.jsonl` carries `skip_reason_counts`

## Files Changed

- `docs/architecture.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `src/mlb_props_stack/backtest.py`
- `src/mlb_props_stack/cli.py`
- `tests/test_backtest.py`
- `tests/test_cli.py`

## Verification

Commands run successfully on April 22, 2026:

```bash
uv sync --extra dev
uv run pytest tests/test_backtest.py tests/test_cli.py
uv run pytest
uv run python -m mlb_props_stack
uv run python -m mlb_props_stack train-starter-strikeout-baseline \
  --start-date 2026-04-18 \
  --end-date 2026-04-23 \
  --output-dir data
uv run python -m mlb_props_stack build-walk-forward-backtest \
  --start-date 2026-04-23 \
  --end-date 2026-04-23 \
  --output-dir data \
  --model-run-dir data/normalized/starter_strikeout_baseline/start=2026-04-18_end=2026-04-23/run=20260422T213008Z
```

Observed results:

- `uv run pytest tests/test_backtest.py tests/test_cli.py`
  - `11 passed`
- `uv run pytest`
  - `68 passed`
- `uv run python -m mlb_props_stack`
  - printed the runtime summary successfully
- training run `20260422T213008Z`
  - MLflow run ID: `4558b270f0e34098b9b327b305707cfa`
  - experiment: `mlb-props-stack-starter-strikeout-training`
  - `training_rows=108`
  - `starter_outcomes=108`
  - `held_out_status=beating_benchmark`
  - `held_out_model_rmse=2.322574`
  - `held_out_benchmark_rmse=2.518693`
  - `held_out_model_mae=1.960881`
  - `held_out_benchmark_mae=2.047866`
- backtest run `20260422T213013Z`
  - MLflow run ID: `dcee221b193b40ebaeef6f82557ee3cc`
  - experiment: `mlb-props-stack-walk-forward-backtest`
  - `snapshot_groups=139`
  - `actionable_bets=0`
  - `below_threshold=0`
  - `skipped=139`
  - `skip_reason_counts={"unmatched_event_mapping": 139}`
  - sample row in `backtest_bets.jsonl` now carries
    `evaluation_status=unmatched_event_mapping` with the explicit reason:
    the selected line snapshot still belongs to an unmatched Odds API event, so
    no honest MLB game join exists for backtest scoring

## Recommended Next Issue

- Tighten the Odds API target-date filtering / stale-unmatched-run cleanup so
  `2026-04-23` sportsbook rows map to the intended MLB slate instead of leaving
  the backtest window with `139` unresolved `unmatched_event_mapping` rows

Why this should go next:

- AGE-190 fixed the diagnosability problem in the backtest itself, but the real
  `2026-04-23` window is still operationally blocked upstream by stale unmatched
  sportsbook events
- until that ingest-side mismatch is fixed, the backtest will honestly explain
  the skip reason but still have no scoreable exact-line rows for that window

## Constraints And Open Questions

- The tracked historical artifact folders for `20260422T205727Z` and
  `20260422T205734Z` remain summary-only in git; future workers should not
  assume those directories contain the full training/backtest inputs needed to
  rerun joins locally
- The real `2026-04-23` odds data under `run=20260422T173024Z` still contains
  `139` normalized prop rows with `match_status="unmatched"`, `game_pk=null`,
  and `pitcher_mlb_id=null`; those are now reported honestly, but they still
  need an upstream resolution path
- A later local Odds API run for the same date (`run=20260422T190633Z`) exists
  with `0` normalized prop rows, so the next issue should confirm whether the
  book had no joinable pitcher-K markets yet or whether the current ingest is
  still filtering out rows it should keep
