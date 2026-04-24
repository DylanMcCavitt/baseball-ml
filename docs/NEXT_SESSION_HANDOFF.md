# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current branch: `main`
- Current `main`: synced to `origin/main` after the AGE-266 merge and handoff
  update
- Last completed issue: `AGE-266` - show active and excluded optional feature
  diagnostics in the dashboard
- Merged PR: <https://github.com/DylanMcCavitt/baseball-ml/pull/39>
- Status: AGE-266 is merged, CI passed, and the canonical local checkout is
  synced to `origin/main`
- Last completed prerequisite: `AGE-265` / PR #38 merged at `9b553b6`

## What Changed In This Slice

- Added active-schema diagnostics to the Feature Inspection screen:
  - active inference run ID
  - source training run ID
  - target date
  - encoded feature count
  - active core features
  - active optional features
- Added read-only optional-feature family diagnostics under
  `src/mlb_props_stack/dashboard/lib/data.py`.
- The diagnostics resolve the active target-date inference model first, then
  follow `source_model_run_id` back to the source baseline training run.
- The diagnostic table now reports, by optional feature family:
  - active feature count
  - source train coverage
  - current target-date coverage
  - status/reason (`active`, `missing_source`,
    `excluded_below_coverage`, `excluded_low_variance`, or
    `excluded_by_selection`)
  - stale/schema-mismatch notes when current artifacts expose old field names
- Added focused fixture coverage for the diagnostics loader using a synthetic
  inference run, source training run, and target-date feature artifacts.

## Current 2026-04-23 Diagnosis

Using canonical data at `/Users/dylanmccavitt/projects/nba-ml/data`:

- active inference run: `20260423T210236Z`
- source training run: `20260422T202712Z`
- encoded feature count: `11`
- active optional feature count: `0`
- split features: `excluded_below_coverage`
  - source train coverage: `0/60 (0%)`
  - source all-row coverage: `0/108 (0%)`
  - target-date coverage: `0/18 (0%)`
- lineup aggregate features: `missing_source`
  - reason: `missing_pregame_lineup`
  - source train coverage: `0/60 (0%)`
  - source all-row coverage: `0/108 (0%)`
  - target-date coverage: `0/18 (0%)`
- park factors: `missing_source`
  - reason: `missing_park_factor_source`
  - schema note: `park_factor present; expected park_k_factor`
- weather: `missing_source`
  - reason: `target-date weather source artifact missing`
  - schema note: `weather_wind_mph present; expected weather_wind_speed_mph`
- umpire: `missing_source`
  - reason: `target-date umpire source artifact missing`

This confirms the dashboard can now answer why optional fields are absent from
the active model without requiring artifact spelunking.

## Files Changed

- `docs/NEXT_SESSION_HANDOFF.md`
- `src/mlb_props_stack/dashboard/app.py`
- `src/mlb_props_stack/dashboard/lib/data.py`
- `src/mlb_props_stack/dashboard/screens/features.py`
- `tests/test_dashboard_data.py`

## Verification

Commands run successfully:

```bash
uv sync --extra dev
uv run pytest tests/test_dashboard_data.py tests/test_dashboard_app.py
uv run pytest tests/test_modeling.py
uv run pytest tests/test_runtime_smokes.py
uv run python -m mlb_props_stack
uv run pytest
python3 -m compileall src tests
MLB_PROPS_STACK_DATA_DIR=/Users/dylanmccavitt/projects/nba-ml/data uv run streamlit run src/mlb_props_stack/dashboard/app.py --server.port 8502 --server.headless true
```

Observed results after rebasing on merged AGE-265:

- dashboard/modeling/runtime suite passed: `19 passed`
- `tests/test_modeling.py`: `5 passed`
- `tests/test_runtime_smokes.py`: `4 passed`
- full test suite: `195 passed`
- `python3 -m compileall src tests` completed successfully
- `uv run python -m mlb_props_stack` printed the runtime configuration banner
- Streamlit launched on `http://localhost:8502`
- GitHub PR #39 `validate` check passed in CI after the AGE-265 rebase

Browser verification:

- Opened
  `http://localhost:8502/?screen=features&board_date=2026-04-23`
  in the Codex in-app browser after the AGE-265 rebase.
- Verified the Feature Inspection screen shows:
  - active run `20260423T210236Z`
  - source run `20260422T202712Z`
  - encoded features `11`
  - active optional `0`
  - weather as `missing source` with target-date coverage `0/18 (0%)`
  - umpire as `missing source` with target-date coverage `0/18 (0%)`
  - schema notes for `park_factor` and `weather_wind_mph`
- Clicked Board and Backtest nav controls and confirmed query params updated
  and the destination screens rendered.

The test runs still show the existing third-party MLflow/Pydantic deprecation
warnings.

## Recommended Next Issue

Work:

1. `AGE-267` - regenerate historical optional-feature artifacts with
   timestamp-valid coverage
   - Backfill/regenerate weather, umpire, park, handedness split, and lineup
     aggregate artifacts so optional columns can reach training rows.
2. `AGE-268` - train and compare expanded-feature baseline against core-only
   model
   - Run only after `AGE-267` proves optional feature coverage exists.
3. `AGE-262` - run the first approved-wager evidence refresh and readiness
   report
   - Use `evaluate-stage-gates` as the readiness report command.
4. `AGE-263` - close the sample-size gap for live-use stage gates
   - Broader evidence collection after the first coherent refresh.

Related but not on the critical path:

- `AGE-209` - add per-pitcher and per-game exposure caps to Kelly sizing
  - Keep queued unless grouped board work shows exposure/correlation sizing is
    still materially distorting the wager card before `AGE-262`.

## Constraints And Notes

- Keep the diagnostics read-only. This issue did not retrain, backfill, lower
  thresholds, or change feature-selection behavior.
- Use the canonical ignored data directory for live dashboard verification:
  `/Users/dylanmccavitt/projects/nba-ml/data`
- The active model remains core-only because the source training run did not
  contain enough populated optional feature coverage.
- The next data issue should improve timestamp-valid artifact coverage rather
  than changing the dashboard to hide the missing-source state.
