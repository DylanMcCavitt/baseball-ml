# Next Session Handoff

## Status

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current issue branch: `feat/age-290-workload-leash-features`
- Base state: `f15c732` (`origin/main`, AGE-289 merged)
- Active issue: `AGE-290` - build expected workload, leash, and injury-context
  features
- Parent rebuild track: `AGE-285` - rebuild pitcher strikeout projection model
  before betting layer
- Implementation state: AGE-290 code, docs, tests, and a real canonical
  artifact smoke are complete in this worktree.
- Canonical ignored data directory for live model checks:
  `/Users/dylanmccavitt/projects/nba-ml/data`

## What Changed In AGE-290

- Added `src/mlb_props_stack/workload_leash_features.py`, a standalone
  expected-opportunity feature builder over the AGE-287 starter-game dataset.
- Added the CLI command:

```bash
uv run python -m mlb_props_stack build-workload-leash-features \
  --start-date 2019-03-20 \
  --end-date 2026-04-24 \
  --output-dir /Users/dylanmccavitt/projects/nba-ml/data \
  --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
```

- The builder reads `starter_game_training_dataset.jsonl` plus the preserved
  AGE-287 direct Statcast `source_manifest.jsonl`.
- Feature rows are restricted to the requested date window while the source
  dataset count still records the full selected dataset run.
- Added workload and leash features:
  - recent 15-day and last-3-start pitch-count and batters-faced means
  - pitcher season pitch-count and batters-faced distributions
  - team season prior-starter leash tendency
  - times-through-order threshold rates for 18, 22, and 27 batters faced
  - expected pitch count and expected batters faced
- Added rest and role context:
  - short, standard, extra, long-layoff, and no-prior-start buckets
  - capped `rest_days_capped`, with no raw continuous `rest_days`
  - long-layoff unknown status separate from standard rest
  - IL and rehab return flags default false until a timestamp-valid source
    explicitly backs them
  - opener/bulk flags from prior short-start workload patterns when available
- Timestamp policy:
  - workload and role inputs use only games before each starter-game
    `official_date`
  - same-game `starter_strikeouts` is used only for report correlations
  - source pitch rows are reduced to inferred starting-pitcher appearances by
    first pitcher for the fielding team/game, so relief appearances do not
    contaminate starter leash
- Updated README, modeling docs, architecture docs, runtime-review docs, CLI
  tests, and focused workload/leash tests.

## Local AGE-290 Data Evidence

- Canonical AGE-287 dataset used for the smoke:
  `/Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z/`
- Real AGE-290 smoke command:

```bash
/opt/homebrew/bin/uv run python -m mlb_props_stack build-workload-leash-features --start-date 2019-03-20 --end-date 2019-03-22 --output-dir /Users/dylanmccavitt/projects/nba-ml/data --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
```

- Smoke artifact:
  `/Users/dylanmccavitt/projects/nba-ml/data/normalized/workload_leash_features/start=2019-03-20_end=2019-03-22/run=20260425T200939Z/`
- Smoke result:
  - `dataset_rows=31729`
  - `feature_rows=4`
  - `pitch_rows=689`
  - `pitchers=4`
  - long layoff rows: `0`
  - opener/bulk role rows: `0`
  - leakage policy status: `ok`
  - raw rest-days primary driver: `False`

## Files Changed

- `README.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/architecture.md`
- `docs/modeling.md`
- `docs/review_runtime_checks.md`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/workload_leash_features.py`
- `tests/test_cli.py`
- `tests/test_workload_leash_features.py`

## Verification

Commands run:

```bash
/opt/homebrew/bin/uv sync --extra dev
/opt/homebrew/bin/uv run pytest tests/test_workload_leash_features.py tests/test_cli.py -q
python3 -m compileall src tests
/opt/homebrew/bin/uv run python -m mlb_props_stack build-workload-leash-features --start-date 2019-03-20 --end-date 2019-03-22 --output-dir /Users/dylanmccavitt/projects/nba-ml/data --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
/opt/homebrew/bin/uv run pytest
/opt/homebrew/bin/uv run python -m mlb_props_stack
```

Observed results:

- First focused pytest attempt failed before setup because the fresh worktree
  did not have the dev extra installed; `/opt/homebrew/bin/uv sync --extra dev`
  fixed it.
- Focused workload/CLI tests: `17 passed`.
- `python3 -m compileall src tests` completed successfully.
- Real canonical AGE-290 CLI smoke succeeded and wrote the 2019-03-20 through
  2019-03-22 feature artifact listed above.
- Full suite: `216 passed` with existing third-party MLflow/Pydantic warnings.
- `python -m mlb_props_stack` printed the runtime configuration banner.

## Recommended Next Issue

1. `AGE-291` - candidate model families with distribution outputs.
2. Then continue through:
   - `AGE-292` - model-only walk-forward validation
   - `AGE-293` - scoreable historical market joins
   - `AGE-294` - rebuilt betting layer
   - `AGE-295` - dashboard and approved-wager UX reconnection

## Constraints And Risks

- Do not resume live-readiness, approved-wager evidence refresh, or dashboard
  reconnection work before the projection rebuild and validation gates pass.
- Do not use same-game target rows, same-game batting orders, or same-game
  workload outcomes as feature inputs.
- Keep workload/leash fields separate from pitcher skill and lineup matchup
  fields; this layer represents opportunity volume, not strikeout ability.
- Do not emit raw continuous `rest_days` as a primary model driver.
- Do not infer IL return, rehab return, or injury context from rest alone.
  Long layoffs remain unknown long-layoff context unless a timestamp-valid
  source explicitly labels the state.
- Keep opener/bulk flags source-backed. The landed builder uses prior
  short-start workload patterns only.
- Do not loosen optional-feature coverage or variance gates to force sparse
  lineup, workload, or umpire fields active.
- Treat scoreable backtest coverage as a hard blocker before any live-use claim.
- Preserve timestamp ordering:
  `features_as_of <= generated_at <= captured_at`.
- Treat v0 as obsolete historical residue. Do not use v0 artifacts, metrics,
  feature assumptions, or code paths as current modeling evidence.

## Deferred Or Superseded Issues

- `AGE-262` and `AGE-263` are canceled/superseded by `AGE-285`.
- `AGE-207` and `AGE-208` are canceled/superseded by `AGE-291`.
- `AGE-209` waits for `AGE-294`.
- `AGE-210` has its AGE-286 audit dependency satisfied but still waits for
  `AGE-291`.
- `AGE-212` waits for `AGE-293` / `AGE-294`.
