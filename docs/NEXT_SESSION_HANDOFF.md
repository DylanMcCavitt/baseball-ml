# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main` at `f271684`
- Last completed issue: `AGE-198` on branch `dylan/adoring-hopper-e6b43f`
  (merged via [PR #24](https://github.com/DylanMcCavitt/baseball-ml/pull/24))
- This slice adds a per-date coverage diagnostic that reports row counts and
  feature/outcome/odds coverage ratios across every ingest, feature, and
  modeling artifact the scoring stack depends on
- Current status: the diagnostic is wired into the CLI and exits non-zero when
  coverage falls under the configured threshold, so it can gate
  `build-walk-forward-backtest` in future automation

## What Was Completed In This Slice

- `src/mlb_props_stack/data_alignment.py`
  - adds `ArtifactCounts`, `DateCoverageRow`, and `DataAlignmentReport`
    dataclasses plus the pure `build_date_coverage_rows` helper
  - counts per-date rows in `games.jsonl`, `probable_starters.jsonl`,
    `lineup_snapshots.jsonl`, `prop_line_snapshots.jsonl` (plus distinct
    `pitcher_mlb_id` coverage), the Statcast feature tables, and the latest
    baseline run's `training_dataset.jsonl`,
    `raw_vs_calibrated_probabilities.jsonl`, and `starter_outcomes.jsonl`
  - derives `feature_coverage`, `outcome_coverage`, and `odds_coverage`
    ratios and flags any date whose ratios fall below the threshold
  - renders a human-readable table plus a footer of failing dates
- `src/mlb_props_stack/cli.py`
  - adds the `check-data-alignment --start-date --end-date --threshold`
    subcommand
  - changes `main()` to return an `int` so non-zero exits flow out of
    `python -m mlb_props_stack`
- `src/mlb_props_stack/__main__.py`
  - propagates the CLI exit code via `sys.exit(main())`
- `README.md`
  - adds a one-line usage example for the new subcommand
- `tests/test_data_alignment.py`
  - adds focused tests for the pure helper, the filesystem orchestrator,
    the renderer, and the CLI exit-code surface on synthetic fixtures

## Files Changed

- `README.md`
- `src/mlb_props_stack/__main__.py`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/data_alignment.py`
- `tests/test_data_alignment.py`

## Verification

Commands run successfully before merge:

```bash
uv sync --extra dev
uv run pytest
uv run python -m mlb_props_stack
uv run python -m mlb_props_stack check-data-alignment \
  --start-date 2026-04-18 \
  --end-date 2026-04-23
```

Observed results:

- full test suite passed: `84 passed`
- `uv run python -m mlb_props_stack`
  - printed the runtime summary successfully
- `uv run python -m mlb_props_stack check-data-alignment --start-date 2026-04-18 --end-date 2026-04-23`
  - reproduced the all-skipped backtest window root cause for the current
    repo state (missing ingest, feature, and odds artifacts for every date)
  - exited with code `1`

## Recommended Next Issue

- Backfill the saved historical ingest, feature, and odds snapshot set under
  `data/normalized/mlb_stats_api/`, `data/normalized/statcast_search/`, and
  `data/normalized/the_odds_api/` for `2026-04-18` through `2026-04-23`, then
  rerun `check-data-alignment` and `build-walk-forward-backtest` to confirm
  the diagnostic flips to green and the backtest window produces non-zero
  actionable rows

Why this should go next:

- `check-data-alignment` now makes the missing-artifact root cause visible in
  seconds, but the local checkout still does not contain the historical
  ingest, feature, or odds snapshots required to actually train/backtest on
  `2026-04-18` through `2026-04-23`
- the Strike Ops dashboard replay path from the previous slice still depends
  on saved historical odds snapshots, so the same backfill unblocks both
  automated backtests and the dashboard replay loop
- once the coverage report passes for that window, the natural follow-up is
  to gate `build-walk-forward-backtest` on `check-data-alignment --threshold`
  so all-skipped windows surface as a precondition failure instead of an
  opaque skip-rate

## Constraints And Open Questions

- `feature_coverage` uses `pitcher_daily_features / probable_starters` as its
  denominator, and `odds_coverage` uses `unique pitcher_mlb_id with lines /
  probable_starters`. If probable starters for a slate are zero, both ratios
  are reported as `n/a` and the date is treated as failing so empty ingest
  surfaces explicitly rather than silently passing
- The report scans the latest normalized run per date for ingest artifacts
  and the latest baseline run whose `training_dataset.jsonl` contains the
  requested date for modeling artifacts. There is currently no explicit
  `--model-run-dir` override; add one only if a future slice needs to pin a
  specific baseline run for the report
- The diagnostic is read-only — it does not mutate or clean up artifacts.
  A separate archival / rotation path should land before running the CLI on
  very large historical windows, because it loads each JSONL file once per
  requested date to filter rows by `official_date`
