# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Active issue branch: `feat/age-189-runtime-smokes`
- `main` already includes the merged `AGE-188` odds-ingest fix
- This branch contains the `AGE-189` runtime-smoke coverage work and is not
  merged yet

## What Was Completed In AGE-189

- `tests/test_runtime_smokes.py`
  - adds a deterministic dashboard file-entry smoke that executes
    `src/mlb_props_stack/dashboard/app.py` as `__main__` against seeded
    `daily_candidates` and `paper_results` artifacts
  - adds a seeded CLI smoke for `ingest-statcast-features` that proves the
    historical metadata backfill path can run from a post-lock metadata fixture
    without live network access
  - adds a seeded CLI smoke for `train-starter-strikeout-baseline` that writes
    real model artifacts from fixture-built AGE-146 feature rows and fake
    Statcast outcome pulls
- `.github/workflows/ci.yml`
  - adds an explicit `uv run pytest tests/test_runtime_smokes.py` step before
    the full pytest suite so runtime-entrypoint smoke coverage is first-class
    in CI
- `docs/review_runtime_checks.md`
  - documents the new runtime smoke command
  - requires runtime-entrypoint PRs to record whether the smoke suite passed in
    addition to the existing scenario-specific runtime commands
- `tests/__init__.py`
  - makes the repo test helpers importable so the smoke suite can reuse the
    seeded fixture builders already locked in adjacent test modules

## Files Changed

- `.github/workflows/ci.yml`
- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/review_runtime_checks.md`
- `tests/__init__.py`
- `tests/test_runtime_smokes.py`

## Verification

Commands run successfully on April 22, 2026:

```bash
uv sync --extra dev
uv run pytest tests/test_runtime_smokes.py
uv run python -m compileall src tests
uv run pytest
uv run python -m mlb_props_stack
```

Observed results:

- runtime smoke suite:
  - `3 passed`
  - dashboard smoke executed the file-entry path and rendered the seeded slate
    tables with the fake Streamlit shim
  - statcast CLI smoke selected the seeded historical metadata fixture and
    wrote feature tables without live MLB or Statcast access
  - training CLI smoke wrote `baseline_model.json` and `evaluation.json` from
    seeded fixture inputs
- `uv run python -m compileall src tests`
  - compiled all source and test modules successfully
- full pytest run:
  - `65 passed`
- `uv run python -m mlb_props_stack`
  - printed the runtime summary successfully

## Recommended Next Issue

- Extend the runtime smoke layer to the remaining artifact-heavy entrypoints
  that still rely only on full-suite coverage:
  - `ingest-odds-api-lines`
  - `build-edge-candidates`
  - `build-daily-candidates`
  - `build-walk-forward-backtest`

Why this should go next:

- `AGE-189` closes the highest-risk regressions called out in live review
  history, but the rest of the saved-artifact pipeline still depends on unit
  coverage plus manual review commands
- keeping each additional smoke narrow and seeded preserves determinism without
  turning CI into one giant integration test

## Constraints And Open Questions

- Keep `tests/test_runtime_smokes.py` deterministic and fixture-backed:
  - no real MLB Stats API access
  - no real Baseball Savant access
  - no real Odds API access or API keys
- The new smoke suite is additive. It does not replace:
  - `uv run streamlit run src/mlb_props_stack/dashboard/app.py` when a PR
    needs a real browser boot check
  - the changed live or seeded runtime command for whichever entrypoint a PR
    actually modifies
- If the helper builders in `tests/test_modeling.py` or
  `tests/test_statcast_feature_ingest.py` move, update
  `tests/test_runtime_smokes.py` imports in the same change so the smoke suite
  keeps using the canonical seeded fixtures
