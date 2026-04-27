# Next Session Handoff

## Status

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Active issue: `AGE-292` - validate model-only strikeout projections with
  walk-forward splits
- Current issue branch:
  `dylanmccavitt2015/age-292-validate-model-only-strikeout-projections-with-walk-forward`
- Base state: current `origin/main` at `4bd1429`
  (`AGE-291: train candidate strikeout model families (#49)`).
- Linear status was moved from `Todo` to `In Progress` at issue start.
- Implementation state: code, docs, focused tests, baseline checks, and a real
  model-only validation artifact run are complete. Remaining closeout steps are
  commit, push, PR creation, and moving Linear to `In Review`.

## What Changed In AGE-292

- Added `src/mlb_props_stack/model_validation.py`, a projection-only validation
  runner over the AGE-291 candidate model families.
- Added the CLI command:

```bash
uv run python -m mlb_props_stack validate-model-only-strikeouts \
  --start-date 2019-03-20 \
  --end-date 2026-04-24 \
  --output-dir data
```

- The command writes:
  - `validation_report.json`
  - `validation_report.md`
  - `validation_predictions.jsonl`
  - `reproducibility_notes.md`
- Headline validation uses rolling season walk-forward splits, not random
  splits:
  - train 2019-2022, validate 2023
  - train 2019-2023, validate 2024
  - train 2019-2024, validate 2025
  - train 2020-2025, validate 2026-to-date when present
- The report includes model-only MAE/RMSE, count-distribution log loss,
  common-line log loss and Brier, calibration by line bucket and confidence
  bucket, bias by pitcher tier, handedness, workload, rest/layoff, season, and
  rule environment, recency sensitivity, observed calibration-derived threshold
  proposals, and a go/no-go recommendation.
- The command intentionally excludes wagering, CLV, ROI, edge candidates,
  approval gates, and stake sizing.

## Real AGE-292 Data Evidence

Command:

```bash
rm -rf /tmp/age292-model-validation && UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run python -m mlb_props_stack validate-model-only-strikeouts --start-date 2019-03-20 --end-date 2026-04-24 --output-dir /tmp/age292-model-validation --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z --first-validation-season 2023
```

Result:

- run:
  `/tmp/age292-model-validation/normalized/model_only_walk_forward_validation/start=2019-03-20_end=2026-04-24/run=20260427T224022Z/`
- joined rows: `31,729`
- walk-forward splits: `4`
- held-out prediction rows: `15,358`
- recommendation: `no_go_betting_layer_still_blocked`
- headline MAE: `2.001726`
- headline RMSE: `2.492028`
- mean bias: `-0.060645`
- count log loss: `2.312353`
- common-line log loss: `0.483954`
- common-line Brier: `0.158808`
- observed confidence/line threshold status:
  `thresholds_observed_from_calibration`
- blocker: canonical data had `0` joined pitcher-skill, lineup-matchup, and
  workload/leash feature rows for the full validation window, so the report
  keeps betting-layer work blocked despite scoreable model-only calibration.
- inspected `validation_report.json`, `validation_report.md`, and
  `validation_predictions.jsonl`; no betting, CLV, ROI, edge, approval, or
  stake-sizing metrics were emitted.

## Files Changed

- `README.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/architecture.md`
- `docs/modeling.md`
- `docs/review_runtime_checks.md`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/model_validation.py`
- `tests/test_cli.py`
- `tests/test_model_validation.py`

## Verification

Commands run:

```bash
UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv sync --extra dev
UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run pytest tests/test_model_validation.py tests/test_cli.py::test_model_only_validation_cli_renders_output_summary -q
UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run pytest
PYTHONPYCACHEPREFIX=/tmp/age292-pycache python3 -m compileall src tests
UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run python -m mlb_props_stack
rm -rf /tmp/age292-model-validation && UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run python -m mlb_props_stack validate-model-only-strikeouts --start-date 2019-03-20 --end-date 2026-04-24 --output-dir /tmp/age292-model-validation --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z --first-validation-season 2023
```

Observed results:

- `uv sync --extra dev` completed successfully with `UV_CACHE_DIR=/tmp/uv-cache`.
- Focused tests: `2 passed`.
- Full suite: `222 passed` with existing third-party MLflow/Pydantic warnings.
- `python3 -m compileall src tests` completed successfully with
  `PYTHONPYCACHEPREFIX=/tmp/age292-pycache`.
- `python -m mlb_props_stack` printed the runtime configuration banner.
- The first focused test attempt failed before sync with
  `Failed to spawn: pytest`; this was the expected fresh-worktree missing-dev
  dependency state and was resolved by `uv sync --extra dev`.
- Real validation runtime completed successfully and produced the artifact run
  listed above.

## Recommended Next Issue

1. Keep betting-layer rebuild work blocked until full-window pitcher-skill,
   lineup-matchup, and workload/leash feature artifacts are available and the
   model-only validation is rerun with those feature joins.
2. Then proceed to `AGE-293` - scoreable historical market joins only after the
   model-only gate has a defensible go/no-go result with feature coverage.

## Constraints And Risks

- GitHub DNS initially failed from this workspace for both SSH fetch and `gh`
  API calls, so the branch started from the local completed AGE-291 branch.
  Fetch later recovered and `origin/main` advanced to the AGE-291 squash commit
  `4bd1429`; keep this branch based on that current remote main before PR.
- The real validation run used the full starter-game dataset only because the
  canonical checkout does not currently have full-window feature-layer artifacts.
  The report records that feature-coverage gap and keeps betting-layer work
  blocked.
- Do not resume pricing, approval-gate, paper-tracking, or dashboard reconnect
  work from this issue alone.
- Do not use v0 artifacts, metrics, feature assumptions, or code paths as
  current modeling evidence.

## Deferred Or Superseded Issues

- `AGE-293` waits for a model-only validation rerun with feature-layer coverage
  or an explicit human decision to continue with the documented blocker.
- `AGE-294` and `AGE-295` remain blocked behind `AGE-293`.
