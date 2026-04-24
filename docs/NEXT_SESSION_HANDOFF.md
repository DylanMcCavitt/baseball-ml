# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current branch: `feat/age-268-expanded-feature-comparison`
- Current branch base: `origin/main` at `d3be83d`
- Active issue: `AGE-268` - train and compare expanded-feature baseline
  against core-only model
- PR: <https://github.com/DylanMcCavitt/baseball-ml/pull/43>
- Status: implementation and local verification are complete; PR review/merge
  closeout is still pending from this worktree.
- Canonical ignored data directory used for live model checks:
  `/Users/dylanmccavitt/projects/nba-ml/data`

## What Changed In This Slice

- Added explicit starter-strikeout training feature sets:
  - `--feature-set core`
    keeps the dense pitcher/workload schema and records optional families as
    excluded by the core feature-set guard.
  - `--feature-set expanded`
    keeps the dense core and admits optional numeric features only when they
    pass the configured coverage and variance gates.
- Persisted optional-feature selection diagnostics in `baseline_model.json`,
  `evaluation.json`, `evaluation_summary.json`, and `evaluation_summary.md`.
- Added `compare-starter-strikeout-baselines`, which:
  - trains core and expanded variants over the same date window,
  - runs pinned walk-forward backtests for each model run,
  - applies shared final wager gates to backtest reporting rows,
  - writes `model_comparison.json`, `model_comparison.md`, and
    `reproducibility_notes.md`.
- Hardened timestamp-shaped run ID creation so back-to-back training/backtest
  calls do not overwrite same-second artifact directories.
- Added `over_odds` and `under_odds` to backtest reporting rows so final wager
  gate checks can compute two-way hold from historical backtest artifacts.
- Updated README/modeling/runtime-check docs for the new feature-set switch and
  comparison report workflow.

## AGE-268 Real Artifact Result

Command run:

```bash
uv run python -m mlb_props_stack compare-starter-strikeout-baselines --start-date 2026-04-18 --end-date 2026-04-23 --output-dir /Users/dylanmccavitt/projects/nba-ml/data
```

Comparison artifacts:

- Report JSON:
  `/Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_model_comparison/start=2026-04-18_end=2026-04-23/run=20260424T164112Z/model_comparison.json`
- Report Markdown:
  `/Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_model_comparison/start=2026-04-18_end=2026-04-23/run=20260424T164112Z/model_comparison.md`
- Reproducibility notes:
  `/Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_model_comparison/start=2026-04-18_end=2026-04-23/run=20260424T164112Z/reproducibility_notes.md`

Model and backtest runs:

| Variant | Training run | Backtest run |
| --- | --- | --- |
| Core | `20260424T164050Z` | `20260424T164112Z` |
| Expanded | `20260424T164112Z` | `20260424T164113Z` |

Recommendation:

- `keep_core_only`
- Expanded activated optional features and improved held-out RMSE, but it
  worsened held-out MAE and calibrated ECE.
- Both variants produced zero scoreable backtest rows and zero final-gate
  approved wagers on this window, so there is no decision-quality evidence to
  promote the expanded model.

Key comparison metrics:

| Metric | Core | Expanded |
| --- | ---: | ---: |
| Held-out RMSE | `2.150421` | `2.108512` |
| Held-out MAE | `1.695608` | `1.722785` |
| Calibrated log loss | `0.243186` | `0.240320` |
| Calibrated ECE | `0.023741` | `0.027515` |
| Snapshot groups | `469` | `469` |
| Scoreable rows | `0` | `0` |
| Final-gate approved wagers | `0` | `0` |

Expanded active optional features:

- `pitcher_k_rate_vs_rhh`
- `pitcher_k_rate_vs_lhh`
- `pitcher_whiff_rate_vs_rhh`
- `pitcher_whiff_rate_vs_lhh`
- `park_k_factor`
- `park_k_factor_vs_rhh`
- `park_k_factor_vs_lhh`
- `weather_temperature_f`
- `weather_wind_speed_mph`
- `weather_humidity_pct`

Expanded excluded optional families:

- lineup aggregate metrics still had `0.0` coverage in the training rows:
  `projected_lineup_k_rate`,
  `projected_lineup_k_rate_vs_pitcher_hand`, `lineup_k_rate_vs_rhp`,
  `lineup_k_rate_vs_lhp`, `projected_lineup_chase_rate`,
  `projected_lineup_contact_rate`, and `lineup_continuity_ratio`.
- umpire rolling metrics still had `0.0` coverage:
  `ump_called_strike_rate_30d` and `ump_k_per_9_delta_vs_league_30d`.

Expanded backtest skip reasons:

- `unmatched_event_mapping`: `337`
- `late_snapshot_after_cutoff`: `74`
- `invalid_projection`: `48`
- `missing_projection`: `10`

The `invalid_projection` rows are timestamp guardrails doing their job:
examples had `features_as_of` after the selected line `captured_at`, so those
rows were rejected instead of scored.

## Files Changed

- `README.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/modeling.md`
- `docs/review_runtime_checks.md`
- `src/mlb_props_stack/backtest.py`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/model_comparison.py`
- `src/mlb_props_stack/modeling.py`
- `tests/test_backtest.py`
- `tests/test_cli.py`
- `tests/test_model_comparison.py`
- `tests/test_modeling.py`
- `tests/test_runtime_smokes.py`

## Verification

Commands run:

```bash
uv sync --extra dev
uv run pytest tests/test_cli.py tests/test_modeling.py tests/test_backtest.py tests/test_model_comparison.py tests/test_runtime_smokes.py
uv run python -m mlb_props_stack compare-starter-strikeout-baselines --start-date 2026-04-18 --end-date 2026-04-23 --output-dir /Users/dylanmccavitt/projects/nba-ml/data
uv run python -m mlb_props_stack check-data-alignment --start-date 2026-04-18 --end-date 2026-04-23 --output-dir /Users/dylanmccavitt/projects/nba-ml/data
uv run pytest tests/test_modeling.py tests/test_backtest.py tests/test_runtime_smokes.py tests/test_model_comparison.py
python3 -m compileall src tests
uv run pytest
uv run python -m mlb_props_stack
```

Observed results:

- Focused CLI/modeling/backtest/comparison/runtime-smoke tests:
  `23 passed` with the existing third-party MLflow/Pydantic warnings.
- Required focused tests:
  `13 passed` with the same existing third-party warnings.
- Full suite:
  `201 passed` with the same existing third-party warnings.
- `python3 -m compileall src tests` completed successfully.
- `uv run python -m mlb_props_stack` printed the runtime configuration banner.
- `check-data-alignment` exited nonzero because odds coverage remains below
  threshold on `2026-04-18` through `2026-04-22`.
  Feature and outcome coverage are now `100.0%` for every date in the window;
  `2026-04-23` passes alignment at current thresholds.

## Recommended Next Issue

1. `AGE-262` - run the first approved-wager evidence refresh and readiness
   report.
   - Use the AGE-268 comparison report above as the model-comparison input.
   - Expect `research_only` unless the readiness command is intentionally
     documenting the current zero-scoreable-row blocker.
   - Do not promote the expanded model from AGE-268 evidence.
2. Follow-up blocker to consider during or after `AGE-262`:
   - The model can train with 155 rows and optional weather/park/pitcher split
     features active, but the saved walk-forward window is still all-skipped.
   - The dominant blocker is odds/join quality:
     unmatched event mappings, late-only snapshots, missing projections, and
     timestamp-invalid projection rows.
3. `AGE-263` remains the broader sample-size and live-use stage-gate gap after
   the stack can produce scoreable backtest rows and approved wager evidence.

## Constraints And Notes

- Keep using `/Users/dylanmccavitt/projects/nba-ml/data` for canonical live
  dashboard/model checks.
- Do not loosen optional-feature coverage or variance thresholds to make sparse
  lineup or umpire families appear active.
- Keep the active model core-only until an expanded comparison improves
  decision-quality metrics without worsening calibration or timestamp safety.
- Treat scoreable backtest coverage as the blocker before declaring live
  readiness, even when held-out model metrics look reasonable.
