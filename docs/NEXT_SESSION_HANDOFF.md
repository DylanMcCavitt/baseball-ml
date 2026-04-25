# Next Session Handoff

## Status

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current issue branch: `feat/age-289-lineup-matchup-features`
- Base state: `5873ef8` (`origin/main`, AGE-288 merged)
- Active issue: `AGE-289` - build batter-by-batter lineup matchup feature set
- Parent rebuild track: `AGE-285` - rebuild pitcher strikeout projection model
  before betting layer
- Implementation state: AGE-289 code, docs, tests, and a real canonical
  artifact smoke are complete in this worktree.
- Canonical ignored data directory for live model checks:
  `/Users/dylanmccavitt/projects/nba-ml/data`

## What Changed In AGE-289

- Added `src/mlb_props_stack/lineup_matchup_features.py`, a standalone
  opponent-lineup matchup feature builder over the AGE-287 starter-game dataset.
- Added the CLI command:

```bash
uv run python -m mlb_props_stack build-lineup-matchup-features \
  --start-date 2019-03-20 \
  --end-date 2026-04-24 \
  --output-dir /Users/dylanmccavitt/projects/nba-ml/data \
  --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
```

- The builder reads `starter_game_training_dataset.jsonl` plus the preserved
  AGE-287 direct Statcast `source_manifest.jsonl`.
- Feature rows are restricted to the requested date window while the source
  dataset count still records the full selected dataset run.
- Added aggregate lineup features:
  - confirmed/projected/missing lineup status
  - lineup IDs and batter-history coverage
  - handedness-weighted lineup K vulnerability
  - contact, chase, whiff, and CSW rates
  - pitcher-arsenal-weighted lineup pitch-type weakness
- Added batter-by-batter slot features:
  - K%, K/PA context, and split versus pitcher throwing hand
  - contact%, chase%, whiff%, and CSW%
  - pitch-type weakness by batter
  - batting-order weight
  - sample-size-regressed K context
- Missingness is explicit:
  - no confirmed lineup
  - no projected lineup
  - incomplete batter history
- Timestamp policy:
  - confirmed lineup references are used only when carried by the starter-game
    artifact as pregame references
  - otherwise the fallback projection is the opponent team's most recent
    prior-game batting order
  - same-game batting orders and same-game target strikeouts are not feature
    inputs
- Updated README, modeling docs, architecture docs, runtime-review docs, CLI
  tests, and focused lineup matchup tests.

## Local AGE-289 Data Evidence

- Canonical AGE-287 dataset used for the smoke:
  `/Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z/`
- Real AGE-289 smoke command:

```bash
/opt/homebrew/bin/uv run python -m mlb_props_stack build-lineup-matchup-features --start-date 2019-03-20 --end-date 2019-03-22 --output-dir /Users/dylanmccavitt/projects/nba-ml/data --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
```

- Smoke artifact:
  `/Users/dylanmccavitt/projects/nba-ml/data/normalized/lineup_matchup_features/start=2019-03-20_end=2019-03-22/run=20260425T185924Z/`
- Smoke result:
  - `dataset_rows=31729`
  - `feature_rows=4`
  - `batter_feature_rows=18`
  - `pitch_rows=689`
  - confirmed lineups: `0`
  - projected lineups: `2`
  - no projection: `2`
  - leakage policy status: `ok`
- A 2024-04-01 through 2024-04-03 smoke was intentionally stopped during
  development because reading and normalizing the full preserved manifest was
  too slow before manifest filtering and indexing were added. The landed code
  filters manifest chunks after the requested end date and uses batter,
  pitcher, and team-lineup indexes.

## Files Changed

- `README.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/architecture.md`
- `docs/modeling.md`
- `docs/review_runtime_checks.md`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/lineup_matchup_features.py`
- `tests/test_cli.py`
- `tests/test_lineup_matchup_features.py`

## Verification

Commands run:

```bash
/opt/homebrew/bin/uv sync --extra dev
/opt/homebrew/bin/uv run pytest tests/test_lineup_matchup_features.py tests/test_cli.py -q
python3 -m compileall src tests
/opt/homebrew/bin/uv run python -m mlb_props_stack build-lineup-matchup-features --start-date 2019-03-20 --end-date 2019-03-22 --output-dir /Users/dylanmccavitt/projects/nba-ml/data --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
/opt/homebrew/bin/uv run pytest tests/test_runtime_smokes.py
/opt/homebrew/bin/uv run pytest
/opt/homebrew/bin/uv run python -m mlb_props_stack
```

Observed results:

- First focused pytest attempt failed before setup because the fresh worktree
  did not have the dev extra installed; `/opt/homebrew/bin/uv sync --extra dev`
  fixed it.
- Focused lineup/CLI tests: `16 passed`.
- Real canonical AGE-289 CLI smoke succeeded and wrote the 2019-03-20 through
  2019-03-22 feature artifact listed above.
- `python3 -m compileall src tests` completed successfully.
- Runtime smokes: `4 passed` with existing third-party MLflow/Pydantic
  warnings.
- Full suite: `212 passed` with existing third-party MLflow/Pydantic warnings.
- `python -m mlb_props_stack` printed the runtime configuration banner.

## Recommended Next Issue

1. `AGE-290` - workload, leash, and injury-context features.
2. Then continue through:
   - `AGE-291` - candidate model families with distribution outputs
   - `AGE-292` - model-only walk-forward validation
   - `AGE-293` - scoreable historical market joins
   - `AGE-294` - rebuilt betting layer
   - `AGE-295` - dashboard and approved-wager UX reconnection

## Constraints And Risks

- Do not resume live-readiness, approved-wager evidence refresh, or dashboard
  reconnection work before the projection rebuild and validation gates pass.
- Do not use same-game target rows or same-game batting orders as feature
  inputs.
- Keep confirmed lineup features separate from projected lineup fallback fields.
- Do not let missing lineup coverage silently become zeros.
- Do not reintroduce raw continuous `rest_days` as an unbounded primary driver.
- Treat pitcher and batter identity priors as shrinkage/context only, not
  memorization shortcuts.
- Do not loosen optional-feature coverage or variance gates to force sparse
  lineup or umpire fields active.
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
