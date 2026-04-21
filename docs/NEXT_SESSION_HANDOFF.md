# Next Session Handoff

## Current State

- Repo: `baseball-ml`
- Local branch: `main`
- `AGE-141` is merged to remote `main` and marked `Done` in Linear.
- Current local `main` matches remote merge commit `a1bcd3b`.
- Baseline is in place:
  - installable Python package
  - documented `uv` and venv-based local commands
  - placeholder tracking/dashboard seams
  - Python 3.11 baseline locked in `pyproject.toml`

## Recommended Next Issue

- `AGE-142` — `Define prop, projection, pricing, and backtest contracts`
- Linear branch name: `dylanmccavitt2015/age-142-define-prop-projection-pricing-and-backtest-contracts`
- Why this should go next:
  - it is still in `Phase 1: Repo + Contracts`
  - later data/model issues depend on contract stability
  - the repo already has initial contract-like models, so this is a tight follow-up slice instead of a greenfield jump

## What Already Exists

- `src/mlb_props_stack/markets.py`
  - has first-pass `PropLine`, `PropProjection`, and `EdgeDecision` dataclasses
- `src/mlb_props_stack/backtest.py`
  - has first-pass `BacktestPolicy`
- `src/mlb_props_stack/pricing.py`
  - has odds conversion / devig / EV / fair odds / Kelly helpers
- `tests/test_pricing.py`
  - has happy-path pricing tests
- `tests/test_edge.py`
  - has a simple projection-to-decision test

## Why AGE-142 Is Not Actually Done Yet

- The issue requires explicit constraints around:
  - probability ranges
  - timestamp usage
  - no ad hoc missing fields for later slices
- Current models are mostly shape-only dataclasses.
- There is not yet clear runtime validation for:
  - probability bounds on projections
  - negative or malformed lines
  - timestamp ordering expectations where relevant
  - compatibility placeholders for future game / lineup / feature-row joins
- Current tests prove basic behavior, but not contract enforcement.

## Suggested Scope For The Next Session

1. Tighten the typed contracts in `src/mlb_props_stack/markets.py`.
2. Add invariant enforcement in `__post_init__` where appropriate.
3. Decide the minimum compatibility seam for later issues:
   - either small typed placeholder models for game / lineup / feature row references
   - or explicit IDs / key contracts that later tables must satisfy
4. Expand tests so contract failures are intentional and explicit.
5. Keep the slice narrow. Do not start real ingestion, persistence, or model training here.

## Concrete Implementation Targets

- In `PropLine`:
  - validate non-empty IDs / names / market
  - validate line is numeric and sane
  - validate odds are non-zero
- In `PropProjection`:
  - validate `over_probability` and `under_probability` are each in `[0, 1]`
  - validate they sum to approximately `1.0` or document why loose tolerance is allowed
  - validate identifiers / market / line align with the contract intent
- In `EdgeDecision`:
  - validate `side` is one of `over` / `under`
  - validate `edge_pct` and `stake_fraction` are sane
- In `BacktestPolicy`:
  - keep it small, but make the intended honesty flags explicit and stable
- Add one small compatibility seam for later inputs:
  - likely a typed key/identity contract for `game`, `lineup`, or `feature_row`
  - do not overbuild a full schema layer

## Files Most Likely To Change

- `src/mlb_props_stack/markets.py`
- `src/mlb_props_stack/backtest.py`
- `src/mlb_props_stack/pricing.py`
- `tests/test_edge.py`
- `tests/test_pricing.py`
- likely new tests such as:
  - `tests/test_contracts.py`

## Verification

Run the issue packet command first:

```bash
uv run --with pytest pytest
```

Also run:

```bash
uv run python -m mlb_props_stack
```

## Guardrails

- Preserve the current standard-library-first posture.
- Do not add MLflow or Streamlit dependencies in this slice.
- Do not start the Odds API or Statcast ingestion work; that belongs to later issues.
- Keep the PR small and focused on contracts plus tests.

## Good End State For AGE-142

- Contract models are clearly typed and enforce the important invariants.
- Tests cover both valid and invalid cases.
- Later issues can consume the contracts without inventing fields on the fly.
- The repo still feels like a clean bootstrap, not a half-started data platform.

## Likely Issue After AGE-142

- `AGE-143` — `Document v1 architecture and modeling guardrails`
- Note:
  - the repo already has starter docs, but they likely still need tightening against the issue's exact requirement for named sources/endpoints and explicit anti-leakage rules
  - treat `AGE-143` as a docs-hardening slice after contract stabilization
