# Next Session Handoff

## Status

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Active issue: `AGE-312` - vertical slice: sportsbook walk-forward backtest report
- Current issue branch:
  `dylanmccavitt2015/age-312-market-report-fixes-v2`
- PR: replacement PR from the fresh branch, after PR #56 was closed unmerged
- Current state: Needs-fixes follow-up code is implemented, verified, committed,
  and ready to push/open as a replacement PR branch.

## What Changed In AGE-312

- Added `build-starter-strikeout-market-report`, a research-only sportsbook
  market report over the AGE-311 `starter_strikeout_ml_predictions.jsonl`
  artifact.
- Added `src/mlb_props_stack/market_report.py`, which adapts AGE-311 prediction
  rows into the existing `build_walk_forward_backtest` model-run contract rather
  than training or introducing a parallel model path.
- Updated `build_walk_forward_backtest` so it can read odds snapshots from a
  separate `--odds-input-dir` while writing outputs elsewhere.
- Hardened the backtest timestamp guard with a new
  `missing_projection_timestamp` skip reason for matched rows that lack
  `features_as_of` or `projection_generated_at`.
- Updated `build-starter-strikeout-ml-report` predictions to include
  `feature_row_id`, `lineup_snapshot_id`, `features_as_of`,
  `projection_generated_at`, and `model_input_refs` when those inputs are
  present in the source artifacts.
- Needs-fixes follow-up: `candidate_models._build_joined_rows()` can now join
  feature artifacts from multiple run directories across one report window.
  This allows season-sized feature materialization to feed one AGE-311 report
  instead of requiring one very large feature run.
- Needs-fixes follow-up: `build-lineup-matchup-features` now reuses sorted
  history date indexes instead of repeatedly sorting prior batter/pitcher
  histories for every starter row.
- Needs-fixes follow-up: `build-lineup-matchup-features` now fails loudly when
  the starter dataset or source pitch rows are missing, rather than writing an
  empty successful-looking feature artifact.

## Files Changed

- `src/mlb_props_stack/market_report.py`
- `src/mlb_props_stack/backtest.py`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/starter_ml_report.py`
- `src/mlb_props_stack/candidate_models.py`
- `src/mlb_props_stack/lineup_matchup_features.py`
- `tests/test_market_report.py`
- `tests/test_cli.py`
- `tests/test_starter_ml_report.py`
- `tests/test_candidate_models.py`
- `tests/test_lineup_matchup_features.py`
- `README.md`
- `docs/architecture.md`
- `docs/modeling.md`
- `docs/review_runtime_checks.md`
- `docs/NEXT_SESSION_HANDOFF.md`

## Runtime Evidence

### Original AGE-311 Prediction Artifact Regeneration

Command:

```bash
rm -rf /tmp/age312-ml-report-historical
/opt/homebrew/bin/uv run python -m mlb_props_stack build-starter-strikeout-ml-report --start-date 2019-03-20 --end-date 2026-04-24 --output-dir /tmp/age312-ml-report-historical --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
```

Result from the earlier PR pass:

- `rows=31729`
- `selected_candidate=validation_top_two_mean_blend`
- `held_out_rmse=2.480789`
- `held_out_mae=1.995703`
- Predictions path:
  `/tmp/age312-ml-report-historical/normalized/starter_strikeout_ml_report/start=2019-03-20_end=2026-04-24/run=20260428T210213Z/starter_strikeout_ml_predictions.jsonl`

### Original AGE-312 Historical Market Report

Command:

```bash
rm -rf /tmp/age312-market-report-historical
/opt/homebrew/bin/uv run python -m mlb_props_stack build-starter-strikeout-market-report --start-date 2019-03-20 --end-date 2026-04-24 --output-dir /tmp/age312-market-report-historical --ml-report-run-dir /tmp/age312-ml-report-historical/normalized/starter_strikeout_ml_report/start=2019-03-20_end=2026-04-24/run=20260428T210213Z --odds-input-dir /Users/dylanmccavitt/projects/nba-ml/data --cutoff-minutes-before-first-pitch 30
```

Result from the earlier PR pass:

- `snapshot_groups=0`
- `scoreable_rows=0`
- `skipped_rows=0`
- `actionable_rows=0`
- `below_threshold_rows=0`
- `skip_reason_counts={}`
- Backtest bets path:
  `/tmp/age312-market-report-historical/normalized/walk_forward_backtest/start=2019-03-20_end=2026-04-24/run=20260428T210339Z/backtest_bets.jsonl`

Artifact inspection confirmed:

- `adapted_model_run/raw_vs_calibrated_probabilities.jsonl`: `253,832` rows
- `adapted_model_run/training_dataset.jsonl`: `31,729` rows
- `backtest_bets.jsonl`: `0` rows
- `join_audit.jsonl`: `0` rows
- Historical market coverage remained blocked by missing saved Odds API
  `prop_line_snapshots.jsonl` artifacts.

### Needs-Fixes Feature Materialization Attempt

Completed before the workspace/data interruption:

```bash
/opt/homebrew/bin/uv run python -m mlb_props_stack build-pitcher-skill-features --start-date 2019-03-20 --end-date 2026-04-24 --output-dir /tmp/age312-feature-repair --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
```

Result:

- `feature_rows=31729`
- `pitch_rows=4684022`
- Feature path:
  `/tmp/age312-feature-repair/normalized/pitcher_skill_features/start=2019-03-20_end=2026-04-24/run=20260428T211833Z/pitcher_skill_features.jsonl`

Completed season-sized lineup chunks:

- 2019: `feature_rows=4858`, `batter_feature_rows=43452`, run
  `/tmp/age312-feature-repair/normalized/lineup_matchup_features/start=2019-03-20_end=2019-12-31/run=20260428T215727Z/`
- 2020: `feature_rows=1796`, `batter_feature_rows=16164`, run
  `/tmp/age312-feature-repair/normalized/lineup_matchup_features/start=2020-01-01_end=2020-12-31/run=20260428T215834Z/`
- 2021: `feature_rows=4858`, `batter_feature_rows=43722`, run
  `/tmp/age312-feature-repair/normalized/lineup_matchup_features/start=2021-01-01_end=2021-12-31/run=20260428T215940Z/`

Blocked:

- The canonical checkout at `/Users/dylanmccavitt/projects/nba-ml` currently
  contains only `.git`, and `git status --short --branch` there shows the
  tracked working tree files as deleted.
- Because this session is not allowed to edit the canonical checkout, the
  remaining 2022-2026 feature materialization and the final historical
  AGE-311/AGE-312 rerun are blocked until that checkout/data artifact is
  restored or another valid dataset artifact path is provided.
- A stale empty 2022 lineup artifact exists under `/tmp/age312-feature-repair`
  from before the missing-dataset hardening was added. Do not use it.

## Verification

Commands run in the recreated issue worktree:

```bash
/opt/homebrew/bin/uv sync --extra dev
/opt/homebrew/bin/uv run pytest tests/test_candidate_models.py tests/test_lineup_matchup_features.py tests/test_starter_ml_report.py tests/test_market_report.py -q
/opt/homebrew/bin/uv run pytest
PYTHONPYCACHEPREFIX=/tmp/age312-pycache python3 -m compileall src tests
/opt/homebrew/bin/uv run python -m mlb_props_stack
git diff --check
```

Observed results:

- `uv sync --extra dev`: completed successfully.
- Focused model/feature/report tests: `10 passed` with existing third-party
  MLflow/Pydantic warnings.
- Full suite: `233 passed` with existing third-party MLflow/Pydantic warnings.
- `compileall`: completed successfully.
- `python -m mlb_props_stack`: printed the runtime configuration banner.
- `git diff --check`: no whitespace errors.

## Recommended Next Step

Restore the canonical checkout/data artifact or provide a valid
`starter_strikeout_training_dataset` run path, then finish materializing
2022-2026 lineup chunks plus workload/leash features under
`/tmp/age312-feature-repair` or another artifact root. After that, rerun
`build-starter-strikeout-ml-report` from the chunked feature runs and then rerun
`build-starter-strikeout-market-report`.

## Constraints And Risks

- Do not treat the AGE-312 report as betting readiness. Current historical
  market coverage is still `0` because saved odds snapshots are absent.
- Live/paper tracking remains blocked by missing market artifacts and broader
  research-only stage gates. No live or paper approval state changed.
- Do not loosen `features_as_of <= projection_generated_at <= line captured_at`
  to recover rows.
- Do not fabricate event or pitcher joins when sportsbook identity is
  uncertain; keep unresolved rows auditable.
- Do not use empty feature artifacts as valid evidence. Missing dataset/source
  rows should fail the builder after this follow-up.
