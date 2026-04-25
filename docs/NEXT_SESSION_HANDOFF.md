# Next Session Handoff

## Status

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current issue branch: `feat/age-288-pitcher-skill-arsenal`
- Base state: `ef9e1e4` (`origin/main`, AGE-287 merged plus post-merge
  handoff update)
- Active issue: `AGE-288` - build pitcher skill and pitch arsenal feature set
- Parent rebuild track: `AGE-285` - rebuild pitcher strikeout projection model
  before betting layer
- Implementation state: AGE-288 code, docs, tests, and a real canonical
  artifact smoke are complete in this worktree.
- Canonical ignored data directory for live model checks:
  `/Users/dylanmccavitt/projects/nba-ml/data`

## What Changed In AGE-288

- Added `src/mlb_props_stack/pitcher_skill_features.py`, a standalone feature
  builder over the AGE-287 starter-game dataset.
- Added the CLI command:

```bash
uv run python -m mlb_props_stack build-pitcher-skill-features \
  --start-date 2019-03-20 \
  --end-date 2026-04-24 \
  --output-dir /Users/dylanmccavitt/projects/nba-ml/data \
  --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
```

- The builder reads `starter_game_training_dataset.jsonl` plus
  `source_manifest.jsonl` raw Statcast CSV paths from the selected AGE-287 run.
- Feature rows use only pitch rows whose `game_date` is before the starter-game
  `official_date`; same-game `starter_strikeouts` is used only inside the
  report correlation calculation and is not written into the feature table.
- Added pitcher-centered feature families:
  - K%, walk rate, K-BB%, strike rate, CSW%, SwStr%, whiff rate, called-strike
    rate, and putaway rate
  - career, season, recent 15-day, and last-3-start windows
  - pitch-type usage plus pitch-type whiff and CSW rates
  - velocity, spin, horizontal/vertical movement, release extension, and deltas
    versus the pitcher's prior baseline
  - sample-size-regressed pitcher skill priors recorded as shrinkage context,
    not raw pitcher-id memorization
  - capped rest context plus explicit rest buckets
- Rest/layoff correction:
  - raw continuous `rest_days` is not emitted as a primary model driver
  - `rest_days_capped` is capped at `14`
  - short rest, standard rest, extra rest, long layoff, and no-prior-start flags
    are explicit
  - generated reports assert long layoff has no unbounded positive numeric
    contribution
- The command writes:
  - `pitcher_skill_features.jsonl`
  - `feature_report.json`
  - `feature_report.md`
  - `reproducibility_notes.md`
- Updated README, modeling, architecture, runtime-review docs, CLI tests, and
  focused feature tests with the new artifact contract.

## Local AGE-288 Data Evidence

- Canonical AGE-287 dataset used for the smoke:
  `/Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z/`
- Real AGE-288 smoke command:

```bash
/opt/homebrew/bin/uv run python -m mlb_props_stack build-pitcher-skill-features --start-date 2024-04-01 --end-date 2024-04-03 --output-dir /Users/dylanmccavitt/projects/nba-ml/data --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
```

- Smoke artifact:
  `/Users/dylanmccavitt/projects/nba-ml/data/normalized/pitcher_skill_features/start=2024-04-01_end=2024-04-03/run=20260425T180426Z/`
- Smoke result:
  - `dataset_rows=31729`
  - `feature_rows=80`
  - `pitch_rows=4684022`
  - `pitchers=80`
  - leakage policy status: `ok`
  - rest policy: raw rest-days primary driver is `false`, cap is `14`, and
    long layoff has no unbounded positive numeric feature
- The smoke took about one minute because it reads all preserved raw Statcast
  chunks from the full AGE-287 run. A future optimization can filter manifest
  chunks by requested feature window plus required prior history, but the
  current runtime is acceptable for the rebuild artifact path.

## Files Changed

- `README.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/architecture.md`
- `docs/modeling.md`
- `docs/review_runtime_checks.md`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/pitcher_skill_features.py`
- `tests/test_cli.py`
- `tests/test_pitcher_skill_features.py`

## Verification

Commands run:

```bash
/opt/homebrew/bin/uv sync --extra dev
/opt/homebrew/bin/uv run pytest tests/test_pitcher_skill_features.py tests/test_cli.py
/opt/homebrew/bin/uv run python -m mlb_props_stack build-pitcher-skill-features --start-date 2024-04-01 --end-date 2024-04-03 --output-dir /Users/dylanmccavitt/projects/nba-ml/data --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
python3 -m compileall src tests
/opt/homebrew/bin/uv run pytest tests/test_runtime_smokes.py
/opt/homebrew/bin/uv run pytest
/opt/homebrew/bin/uv run python -m mlb_props_stack
```

Observed results:

- First focused pytest attempt failed before setup because the fresh worktree
  did not have the dev extra installed; `uv sync --extra dev` fixed it.
- Focused pitcher-feature and CLI tests: `14 passed`.
- Real canonical AGE-288 CLI smoke succeeded and wrote the 2024-04-01 through
  2024-04-03 feature artifact listed above.
- `python3 -m compileall src tests` completed successfully.
- Runtime smokes: `4 passed` with existing third-party MLflow/Pydantic
  warnings.
- Full suite: `208 passed` with existing third-party MLflow/Pydantic warnings.
- `python -m mlb_props_stack` printed the runtime configuration banner.

## Recommended Next Issue

1. `AGE-289` - batter-by-batter lineup matchup features.
2. Then continue through:
   - `AGE-290` - workload, leash, and injury-context features
   - `AGE-291` - candidate model families with distribution outputs
   - `AGE-292` - model-only walk-forward validation
   - `AGE-293` - scoreable historical market joins
   - `AGE-294` - rebuilt betting layer
   - `AGE-295` - dashboard and approved-wager UX reconnection

## Constraints And Risks

- Do not resume live-readiness, approved-wager evidence refresh, or dashboard
  reconnection work before the projection rebuild and validation gates pass.
- Do not use same-game target rows as feature inputs. AGE-288 feature rows must
  remain prior-games-only.
- Do not reintroduce raw continuous `rest_days` as an unbounded primary driver.
- Treat pitcher identity priors as shrinkage/context only, not a memorization
  shortcut.
- Do not loosen optional-feature coverage or variance gates to force sparse
  lineup or umpire fields active.
- Treat scoreable backtest coverage as a hard blocker before any live-use
  claim.
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
