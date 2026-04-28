# Next Session Handoff

## Status

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Active issue: `AGE-312` - vertical slice: sportsbook walk-forward backtest report
- Current issue branch:
  `dylanmccavitt2015/age-312-vertical-slice-sportsbook-walk-forward-backtest-report`
- Base state: branch created from `origin/main` after `git fetch origin --prune`.
- PR: https://github.com/DylanMcCavitt/baseball-ml/pull/56
- Implementation state: code, docs, focused tests, full tests, compile check,
  module smoke, AGE-311 ML report regeneration, AGE-312 market-report runtime,
  and output artifact inspection are complete.

## What Changed In AGE-312

- Added `build-starter-strikeout-market-report`, a research-only sportsbook
  market report over the AGE-311 `starter_strikeout_ml_predictions.jsonl`
  artifact.
- Added `src/mlb_props_stack/market_report.py`, which adapts AGE-311 prediction
  rows into the existing `build_walk_forward_backtest` model-run contract rather
  than training or introducing a parallel model path.
- Updated `build_walk_forward_backtest` so it can read odds snapshots from a
  separate `--odds-input-dir` while writing outputs elsewhere. This lets issue
  worktrees consume read-only historical line artifacts without writing into the
  canonical checkout.
- Hardened the backtest timestamp guard with a new
  `missing_projection_timestamp` skip reason for matched rows that lack
  `features_as_of` or `projection_generated_at`.
- Updated `build-starter-strikeout-ml-report` predictions to include
  `feature_row_id`, `lineup_snapshot_id`, `features_as_of`,
  `projection_generated_at`, and `model_input_refs` when those inputs are
  present in the source artifacts.
- Updated `README.md`, `docs/architecture.md`, `docs/modeling.md`, and
  `docs/review_runtime_checks.md` with the market-report command, artifacts,
  and review expectations.

The report writes:

- `starter_strikeout_market_report.json`
- `starter_strikeout_market_report.md`
- `adapted_model_run/`

It also calls the existing walk-forward backtest path, which writes:

- `backtest_bets.jsonl`
- `bet_reporting.jsonl`
- `backtest_runs.jsonl`
- `join_audit.jsonl`
- `clv_summary.jsonl`
- `roi_summary.jsonl`
- `edge_bucket_summary.jsonl`

## Runtime Evidence

### AGE-311 Prediction Artifact Regeneration

Command:

```bash
rm -rf /tmp/age312-ml-report-historical && /opt/homebrew/bin/uv run python -m mlb_props_stack build-starter-strikeout-ml-report --start-date 2019-03-20 --end-date 2026-04-24 --output-dir /tmp/age312-ml-report-historical --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
```

Result:

- `rows=31729`
- `selected_candidate=validation_top_two_mean_blend`
- `held_out_rmse=2.480789`
- `held_out_mae=1.995703`
- Report path:
  `/tmp/age312-ml-report-historical/normalized/starter_strikeout_ml_report/start=2019-03-20_end=2026-04-24/run=20260428T210213Z/starter_strikeout_ml_report.json`
- Markdown path:
  `/tmp/age312-ml-report-historical/normalized/starter_strikeout_ml_report/start=2019-03-20_end=2026-04-24/run=20260428T210213Z/starter_strikeout_ml_report.md`
- Predictions path:
  `/tmp/age312-ml-report-historical/normalized/starter_strikeout_ml_report/start=2019-03-20_end=2026-04-24/run=20260428T210213Z/starter_strikeout_ml_predictions.jsonl`

Artifact inspection confirmed the regenerated predictions carry the new fields,
but the current canonical starter dataset still does not supply per-row
`features_as_of`, `lineup_snapshot_id`, or feature refs because the expected
feature-layer artifacts are absent.

### AGE-312 Historical Market Report

Command:

```bash
rm -rf /tmp/age312-market-report-historical && /opt/homebrew/bin/uv run python -m mlb_props_stack build-starter-strikeout-market-report --start-date 2019-03-20 --end-date 2026-04-24 --output-dir /tmp/age312-market-report-historical --ml-report-run-dir /tmp/age312-ml-report-historical/normalized/starter_strikeout_ml_report/start=2019-03-20_end=2026-04-24/run=20260428T210213Z --odds-input-dir /Users/dylanmccavitt/projects/nba-ml/data --cutoff-minutes-before-first-pitch 30
```

Result:

- `snapshot_groups=0`
- `scoreable_rows=0`
- `skipped_rows=0`
- `actionable_rows=0`
- `below_threshold_rows=0`
- `skip_reason_counts={}`
- Market report path:
  `/tmp/age312-market-report-historical/normalized/starter_strikeout_market_report/start=2019-03-20_end=2026-04-24/run=20260428T210339Z/starter_strikeout_market_report.json`
- Market report Markdown path:
  `/tmp/age312-market-report-historical/normalized/starter_strikeout_market_report/start=2019-03-20_end=2026-04-24/run=20260428T210339Z/starter_strikeout_market_report.md`
- Adapted model run:
  `/tmp/age312-market-report-historical/normalized/starter_strikeout_market_report/start=2019-03-20_end=2026-04-24/run=20260428T210339Z/adapted_model_run`
- Backtest bets path:
  `/tmp/age312-market-report-historical/normalized/walk_forward_backtest/start=2019-03-20_end=2026-04-24/run=20260428T210339Z/backtest_bets.jsonl`
- Join audit path:
  `/tmp/age312-market-report-historical/normalized/walk_forward_backtest/start=2019-03-20_end=2026-04-24/run=20260428T210339Z/join_audit.jsonl`
- CLV summary path:
  `/tmp/age312-market-report-historical/normalized/walk_forward_backtest/start=2019-03-20_end=2026-04-24/run=20260428T210339Z/clv_summary.jsonl`
- ROI summary path:
  `/tmp/age312-market-report-historical/normalized/walk_forward_backtest/start=2019-03-20_end=2026-04-24/run=20260428T210339Z/roi_summary.jsonl`

Artifact inspection confirmed:

- `adapted_model_run/raw_vs_calibrated_probabilities.jsonl`: `253,832` rows
- `adapted_model_run/training_dataset.jsonl`: `31,729` rows
- `backtest_bets.jsonl`: `0` rows
- `join_audit.jsonl`: `0` rows
- `/Users/dylanmccavitt/projects/nba-ml/data/normalized/the_odds_api` contains
  `0` `prop_line_snapshots.jsonl` files, so historical market coverage remains
  blocked by missing saved odds artifacts.

## Files Changed

- `src/mlb_props_stack/market_report.py`
- `src/mlb_props_stack/backtest.py`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/starter_ml_report.py`
- `tests/test_market_report.py`
- `tests/test_cli.py`
- `tests/test_starter_ml_report.py`
- `README.md`
- `docs/architecture.md`
- `docs/modeling.md`
- `docs/review_runtime_checks.md`
- `docs/NEXT_SESSION_HANDOFF.md`

## Verification

Commands run:

```bash
/opt/homebrew/bin/uv sync --extra dev
/opt/homebrew/bin/uv run pytest tests/test_market_report.py tests/test_starter_ml_report.py tests/test_cli.py -q
/opt/homebrew/bin/uv run pytest tests/test_market_report.py tests/test_starter_ml_report.py tests/test_cli.py tests/test_backtest.py -q
/opt/homebrew/bin/uv run pytest tests/test_runtime_smokes.py -q
/opt/homebrew/bin/uv run pytest
PYTHONPYCACHEPREFIX=/tmp/age312-pycache python3 -m compileall src tests
/opt/homebrew/bin/uv run python -m mlb_props_stack
git diff --check
```

Observed results:

- `uv sync --extra dev`: completed successfully.
- Focused market/report/CLI tests: `21 passed`.
- Focused market/report/CLI/backtest tests: `24 passed` after the
  timestamp-skip hardening update.
- Runtime smoke tests: `5 passed` with existing third-party MLflow/Pydantic
  warnings.
- Full suite: `231 passed` with existing third-party MLflow/Pydantic warnings.
- `compileall`: completed successfully.
- `python -m mlb_props_stack`: printed the runtime configuration banner.
- `git diff --check`: no whitespace errors.

## Recommended Next Issue

Restore or materialize the historical Odds API pitcher strikeout prop snapshots
under `data/normalized/the_odds_api/...`, then rerun
`build-starter-strikeout-market-report` against the AGE-311 prediction artifact
to produce non-empty scoreable/skipped market coverage.

## Constraints And Risks

- Do not treat the AGE-312 report as betting readiness. Current historical
  market coverage is `0` because saved odds snapshots are absent.
- Live/paper tracking remains blocked by missing market artifacts and by the
  broader research-only stage gates. No live or paper approval state changed in
  this issue.
- The current AGE-311 historical report still falls back to dense starter-game
  context because pitcher-skill, lineup-matchup, and workload/leash feature
  artifacts are absent. When those feature artifacts are restored, rerun the
  AGE-311 report so market-report rows can carry stronger per-row input refs and
  timestamps.
- Do not loosen `features_as_of <= projection_generated_at <= line captured_at`
  to recover rows. Missing projection timestamps now produce
  `missing_projection_timestamp`.
- Do not fabricate event or pitcher joins when sportsbook identity is
  uncertain; keep unresolved rows auditable.
