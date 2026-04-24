# Next Session Handoff

## Status

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current working branch: `feat/age-286-baseline-v0-audit`
- Branch started from: `78fb3d5` (`origin/main`, synced after the AGE-268
  merge and local cleanup)
- Current issue: `AGE-286` - audit and freeze current strikeout baseline as v0
- Parent rebuild track: `AGE-285` - rebuild pitcher strikeout projection model
  before betting layer
- Implementation state: AGE-286 changes are complete locally and ready for PR
  review/merge.
- Canonical ignored data directory for future live model checks:
  `/Users/dylanmccavitt/projects/nba-ml/data`

## What Changed In AGE-286

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
- `docs/baseline_v0_audit.md`
- `docs/modeling.md`
- `src/mlb_props_stack/modeling.py`
- `tests/test_runtime_smokes.py`

## Verification

Commands run:

```bash
/opt/homebrew/bin/uv sync --extra dev
/opt/homebrew/bin/uv run pytest tests/test_runtime_smokes.py
/opt/homebrew/bin/uv run pytest tests/test_modeling.py
/opt/homebrew/bin/uv run pytest tests/test_cli.py tests/test_edge.py tests/test_backtest.py tests/test_model_comparison.py
python3 -m compileall src tests
/opt/homebrew/bin/uv run pytest
/opt/homebrew/bin/uv run python -m mlb_props_stack
```

Observed results:

- `uv sync --extra dev` succeeded using `/opt/homebrew/bin/uv`; plain `uv` was
  not on PATH in this worktree shell.
- Runtime smokes: `4 passed` with existing third-party MLflow/Pydantic
  warnings.
- Modeling tests: `5 passed` with existing third-party MLflow/Pydantic
  warnings.
- Focused CLI/edge/backtest/model-comparison tests: `26 passed` with existing
  third-party MLflow/Pydantic warnings.
- `python3 -m compileall src tests` completed successfully.
- Full suite: `201 passed` with existing third-party MLflow/Pydantic warnings.
- `python -m mlb_props_stack` printed the runtime configuration banner.

## Recommended Next Issue

1. `AGE-287` - build 5-7 season starter-game strikeout training dataset.
2. Then continue through:
   - `AGE-288` - pitcher skill and pitch arsenal features
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
