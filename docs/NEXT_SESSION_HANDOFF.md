# Next Session Handoff

## Status

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current working branch: `feat/age-287-starter-game-dataset`
- Branch started from: `495e2b4` (`origin/main`, AGE-286 merged baseline v0
  audit)
- Current issue: `AGE-287` - build 5-7 season starter-game strikeout training dataset
- Parent rebuild track: `AGE-285` - rebuild pitcher strikeout projection model
  before betting layer
- Implementation state: AGE-287 code, docs, and canonical multi-season artifact
  build are complete locally; ready for PR review/merge.
- Canonical ignored data directory for future live model checks:
  `/Users/dylanmccavitt/projects/nba-ml/data`

## What Changed In AGE-287

- Added `src/mlb_props_stack/starter_dataset.py`, a standalone dataset builder
  for the projection rebuild.
- Added the CLI command:

```bash
uv run python -m mlb_props_stack build-starter-strikeout-dataset \
  --start-date 2019-03-20 \
  --end-date 2026-04-24 \
  --output-dir /Users/dylanmccavitt/projects/nba-ml/data \
  --chunk-days 3 \
  --max-fetch-workers 4
```

- Source behavior:
  - uses normalized Statcast feature runs when present
  - falls back to direct regular-season Baseball Savant Statcast pitch-log
    chunks when feature runs are absent
  - direct mode infers actual starters from the first pitcher used by each
    fielding team and counts same-game strikeout events only for the target
    label
- The command writes:
  - `starter_game_training_dataset.jsonl`
  - `coverage_report.json`
  - `coverage_report.md`
  - `missing_targets.jsonl`
  - `source_manifest.jsonl`
  - `schema_drift_report.json`
  - `timestamp_policy.md`
  - `reproducibility_notes.md`
- Added fixture-backed tests that verify:
  - rows are unique by `(official_date, game_pk, pitcher_id)`
  - same-game Statcast target matching lands in `starter_strikeouts`
  - missing targets are excluded and recorded
  - row counts by season/team and timestamp-policy status land in coverage
  - direct Statcast fallback works without feature-run artifacts
  - missing source dates and source chunk cap warnings are reported
- Updated README, modeling, architecture, and runtime-review docs with the new
  dataset contract.

## Local AGE-287 Data Evidence

- Canonical data directory inspected:
  `/Users/dylanmccavitt/projects/nba-ml/data`
- Only tracked static data files are currently present there:
  `data/static/park_factors/*` and `data/static/venues/*`.
- Running the new command against the canonical data path for
  `2026-04-18 -> 2026-04-23` produced an honest zero-row artifact because no
  normalized Statcast feature runs exist in that directory after the AGE-268
  cleanup.
- Investigation result: the zero-row artifact happened because the first
  implementation depended only on missing feature-run artifacts. Starter-game
  target data was available from direct regular-season Statcast pitch-log
  pulls.
- The fixed direct-mode build completed successfully:

```bash
/opt/homebrew/bin/uv run python -m mlb_props_stack build-starter-strikeout-dataset --start-date 2019-03-20 --end-date 2026-04-24 --output-dir /Users/dylanmccavitt/projects/nba-ml/data --chunk-days 3 --max-fetch-workers 4
```

- Canonical AGE-287 artifact:
  `/Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z/`
- Coverage:
  - `source_mode=direct_statcast_pitch_log`
  - `dataset_rows=31729`
  - `source_starter_rows=31732`
  - `missing_targets=3`
  - `excluded_starts=3`
  - `duplicate_source_rows=0`
  - `timestamp_violations=0`
  - `source_chunks=587`
  - `cap_warning_count=0`
  - `max_pitch_row_count=14358`, safely below the 25,000-row warning threshold
  - seasons represented: `2019`, `2020`, `2021`, `2022`, `2023`, `2024`,
    `2025`, `2026`
  - row counts by season: `2019=4858`, `2020=1796`, `2021=4858`,
    `2022=4859`, `2023=4860`, `2024=4856`, `2025=4860`, `2026=782`
- Missing targets:
  - `2022-05-22` STL away, Steven Matz
  - `2024-06-05` DET away, Kenta Maeda
  - `2024-08-27` CWS home, Garrett Crochet

## Prior AGE-286 Context

- Froze the current generated starter strikeout model label as
  `starter-strikeout-baseline-v0`.
- Added `docs/baseline_v0_audit.md`, which records:
  - what the current ridge baseline, global dispersion layer, calibrator, and
    artifact layout do
  - the preserved AGE-268 run IDs, split behavior, comparison metrics, zero
    scoreable rows, and zero approved wagers
  - the exact retained optional-feature activation and exclusion evidence
  - why `rest_days` is unsafe as a standalone continuous core feature for
    injury return, long layoff, role-change, and pitch-limit context
  - which old betting/readiness issues are blocked or superseded by the model
    rebuild
  - which v0 assumptions must not carry forward
- Updated `README.md`, `docs/modeling.md`, and `docs/architecture.md` so the
  current baseline is explicitly described as infrastructure-only, not the
  production or live-use projection model.
- Updated the seeded training runtime smoke expectation for the new generated
  `starter-strikeout-baseline-v0` label.

## AGE-268 Evidence Location

The detailed AGE-268 comparison evidence now lives in
`docs/baseline_v0_audit.md`. Important retained facts:

- Comparison command:

```bash
uv run python -m mlb_props_stack compare-starter-strikeout-baselines --start-date 2026-04-18 --end-date 2026-04-23 --output-dir /Users/dylanmccavitt/projects/nba-ml/data
```

- Run IDs:
  - Core training `20260424T164050Z`, core backtest `20260424T164112Z`
  - Expanded training `20260424T164112Z`, expanded backtest `20260424T164113Z`
- Current `_split_dates` behavior for the six-date window:
  - train: `2026-04-18` through `2026-04-21`
  - validation: `2026-04-22`
  - test: `2026-04-23`
- Preserved backtest counts:
  - snapshot groups: `469` core, `469` expanded
  - scoreable rows: `0` core, `0` expanded
  - final-gate approved wagers: `0` core, `0` expanded
- The generated AGE-268 files were intentionally deleted from the canonical
  checkout on 2026-04-24. Do not guess deleted training-row counts; AGE-287
  should persist durable multi-season row-count evidence.

## Files Changed

- `README.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/architecture.md`
- `docs/modeling.md`
- `docs/review_runtime_checks.md`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/starter_dataset.py`
- `tests/test_cli.py`
- `tests/test_starter_dataset.py`

## Verification

Commands run:

```bash
/opt/homebrew/bin/uv sync --extra dev
/opt/homebrew/bin/uv run pytest tests/test_modeling.py
/opt/homebrew/bin/uv run python -m mlb_props_stack train-starter-strikeout-baseline --start-date 2026-04-18 --end-date 2026-04-23 --output-dir /Users/dylanmccavitt/projects/nba-ml/data
/opt/homebrew/bin/uv run pytest tests/test_starter_dataset.py
/opt/homebrew/bin/uv run python -m mlb_props_stack build-starter-strikeout-dataset --start-date 2026-04-18 --end-date 2026-04-23 --output-dir /Users/dylanmccavitt/projects/nba-ml/data
/opt/homebrew/bin/uv run pytest tests/test_starter_dataset.py tests/test_cli.py
/opt/homebrew/bin/uv run python -m mlb_props_stack build-starter-strikeout-dataset --start-date 2024-04-01 --end-date 2024-04-01 --output-dir /Users/dylanmccavitt/projects/nba-ml/data --chunk-days 1 --max-fetch-workers 1
/opt/homebrew/bin/uv run python -m mlb_props_stack build-starter-strikeout-dataset --start-date 2025-04-01 --end-date 2025-04-01 --output-dir /Users/dylanmccavitt/projects/nba-ml/data --chunk-days 1 --max-fetch-workers 1
/opt/homebrew/bin/uv run python -m mlb_props_stack build-starter-strikeout-dataset --start-date 2026-04-01 --end-date 2026-04-01 --output-dir /Users/dylanmccavitt/projects/nba-ml/data --chunk-days 1 --max-fetch-workers 1
/opt/homebrew/bin/uv run python -m mlb_props_stack build-starter-strikeout-dataset --start-date 2019-03-20 --end-date 2026-04-24 --output-dir /Users/dylanmccavitt/projects/nba-ml/data --chunk-days 3 --max-fetch-workers 4
python3 -m compileall src tests
/opt/homebrew/bin/uv run pytest tests/test_runtime_smokes.py
/opt/homebrew/bin/uv run pytest
/opt/homebrew/bin/uv run python -m mlb_props_stack
```

Observed results:

- `uv sync --extra dev` succeeded using `/opt/homebrew/bin/uv`; plain `uv` was
  not on PATH in this worktree shell.
- Modeling tests before edits: `5 passed` with existing third-party
  MLflow/Pydantic warnings.
- Current training CLI against canonical data path failed with
  `FileNotFoundError: No AGE-146 Statcast feature runs were found inside the
  requested date range.` This is expected after AGE-268 artifact cleanup and is
  the exact pre-change behavior AGE-287 needed to account for.
- New starter dataset tests: `2 passed`.
- New dataset CLI against canonical data path succeeded and wrote an honest
  zero-row coverage artifact for `2026-04-18 -> 2026-04-23`: `requested_dates=6`,
  `source_dates=0`, `dataset_rows=0`, `missing_targets=0`, `excluded_starts=0`.
- One-day real direct-mode checks:
  - `2024-04-01`: `dataset_rows=28`
  - `2025-04-01`: `dataset_rows=26`
  - `2026-04-01`: `dataset_rows=30`
- Full direct-mode build for `2019-03-20 -> 2026-04-24` succeeded with
  `dataset_rows=31729`, `source_chunks=587`, `cap_warning_count=0`, and
  `timestamp_violations=0`.
- Focused dataset and CLI tests after direct fallback: `14 passed`.
- `python3 -m compileall src tests` completed successfully.
- Runtime smokes: `4 passed` with existing third-party MLflow/Pydantic
  warnings.
- Full suite: `205 passed` with existing third-party MLflow/Pydantic warnings.
- `python -m mlb_props_stack` printed the runtime configuration banner.

## Recommended Next Issue

1. `AGE-288` - pitcher skill and pitch arsenal features.
2. Then continue through:
   - `AGE-289` - batter-by-batter lineup matchup features
   - `AGE-290` - workload, leash, and injury-context features
   - `AGE-291` - candidate model families with distribution outputs
   - `AGE-292` - model-only walk-forward validation
   - `AGE-293` - scoreable historical market joins
   - `AGE-294` - rebuilt betting layer
   - `AGE-295` - dashboard and approved-wager UX reconnection

## Constraints And Risks

- Do not resume live-readiness, approved-wager evidence refresh, or dashboard
  reconnection work before the projection rebuild and validation gates pass.
- Do not loosen optional-feature coverage or variance gates to force sparse
  lineup or umpire fields active.
- Do not use `rest_days` as a standalone health, workload, or return-from-layoff
  proxy in the rebuild.
- Treat scoreable backtest coverage as a hard blocker before any live-use claim.
- Preserve timestamp ordering:
  `features_as_of <= generated_at <= captured_at`.
- Keep v0 artifacts and code paths useful as infrastructure, but do not promote
  v0 projection quality as decision-grade.

## Deferred Or Superseded Issues

- `AGE-262` and `AGE-263` are canceled/superseded by `AGE-285`.
- `AGE-207` and `AGE-208` are canceled/superseded by `AGE-291`.
- `AGE-209` waits for `AGE-294`.
- `AGE-210` has its AGE-286 audit dependency satisfied by this freeze but
  still waits for `AGE-291`.
- `AGE-212` waits for `AGE-293` / `AGE-294`.
