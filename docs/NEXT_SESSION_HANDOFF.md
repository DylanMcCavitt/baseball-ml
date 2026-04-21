# Next Session Handoff

## Current State

- Repo: `baseball-ml`
- Default branch: `main`
- Current issue branch: `dylanmccavitt2015/age-142-define-prop-projection-pricing-and-backtest-contracts`
- `main` includes the repo-local `AGENTS.md` guide from commit `4027dbd`.
- `AGE-142` implementation is complete locally, pushed to GitHub, and open as PR #2:
  - `https://github.com/DylanMcCavitt/baseball-ml/pull/2`
- This branch now contains:
  - hardened prop / projection / decision / backtest contracts
  - explicit `PropSelectionKey` and `ProjectionInputRef` seams for later joins
  - expanded contract, pricing, and edge tests
  - a baseline GitHub Actions CI workflow in `.github/workflows/ci.yml`
  - corrected repo docs for `uv` development commands

## What Was Completed In AGE-142

- `src/mlb_props_stack/markets.py`
  - added invariant checks for blank IDs, numeric ranges, odds, and timestamps
  - added `PropSelectionKey`
  - added `ProjectionInputRef`
- `src/mlb_props_stack/edge.py`
  - enforced line/projection contract matching
  - enforced feature and generation timestamps relative to market capture
  - rejected boundary probabilities that cannot produce fair American odds
- `src/mlb_props_stack/backtest.py`
  - made future join-ref and rejected-prop expectations explicit
  - required at least one reporting output to stay enabled
- `src/mlb_props_stack/pricing.py`
  - tightened Kelly input validation
- `tests/test_contracts.py`
  - added focused contract validation coverage
- `tests/test_edge.py`
  - added mismatch and timestamp guardrail coverage
- `tests/test_pricing.py`
  - added invalid-input Kelly coverage
- `.github/workflows/ci.yml`
  - added baseline CI for locked dependency install, compile, tests, and CLI smoke
- `README.md`
  - documented the CI baseline
  - corrected `uv` usage to `uv sync --extra dev`
- `AGENTS.md`
  - corrected the repo verification sequence for the current `uv` setup

## Verification Run

These commands were run successfully from the issue branch:

```bash
uv sync --locked --extra dev
uv run pytest
uv run python -m compileall src tests
uv run python -m mlb_props_stack
```

## Recommended Next Issue

- `AGE-143` — `Document v1 architecture and modeling guardrails`

Why this should go next:

- the contracts are now explicit enough that the docs can describe the real v1
  interfaces instead of aspirational placeholders
- `docs/architecture.md` and `docs/modeling.md` should now be tightened against
  the actual repo seams, named data sources, and anti-leakage rules
- this keeps `Phase 1: Repo + Contracts` coherent before ingestion or modeling
  work starts

## Constraints For The Next Worktree

- Start from `main` after PR #2 is merged. Do not branch from this issue branch
  unless the PR is still open and you are intentionally stacking on top of it.
- Keep the standard-library-first posture until a later issue explicitly needs
  data, model, or experiment dependencies.
- Keep `python -m mlb_props_stack` working.
- Use the corrected `uv` flow for local verification:
  - `uv sync --extra dev`
  - `uv run pytest`
  - `uv run python -m mlb_props_stack`
- Keep CI narrow and honest. The current workflow is a bootstrap baseline, not
  a full training or deployment pipeline.

## Open Questions

- Once PR #2 merges, confirm that the new GitHub Actions workflow is enabled and
  reports status on subsequent PRs.
- If `AGE-143` wants exact endpoint naming for future data sources, use the real
  source candidates that will likely matter next:
  - Statcast / Baseball Savant history
  - MLB schedule / probable starters / lineup metadata
  - sportsbook prop line snapshots and line movement captures
