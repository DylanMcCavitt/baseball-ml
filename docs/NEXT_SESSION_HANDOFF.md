# Next Session Handoff

## Current State

- Repo: `baseball-ml`
- Default branch: `main`
- Current issue branch: `feat/age-143-document-v1-architecture-and-modeling-guardrails`
- `main` already includes the merged `AGE-142` contract and CI baseline work from
  PR #2.
- `AGE-143` is complete locally on this branch. The repo now has tighter docs
  for:
  - the v1 product boundary
  - the allowed upstream source systems and named endpoints
  - the current contract spine in `src/mlb_props_stack`
  - the timestamp and leakage rules that future ingestion and modeling work must
    satisfy

## What Was Completed In AGE-143

- `docs/architecture.md`
  - rewrote the architecture doc around the actual repo seams instead of
    aspirational layers
  - named the v1 trusted sources:
    - Baseball Savant Statcast Search CSV
    - Baseball Savant CSV docs
    - MLB Stats API schedule / probable starter hydration
    - MLB Stats API lineup hydration
    - MLB Stats API `game/{gamePk}/feed/live`
    - sportsbook strikeout prop snapshots as `PropLine` inputs
  - documented current contract modules:
    - `config.py`
    - `markets.py`
    - `pricing.py`
    - `edge.py`
    - `backtest.py`
  - made the timestamp chain explicit:
    - `features_as_of <= generated_at <= captured_at`
  - added explicit non-goals and live-use caveats
- `docs/modeling.md`
  - rewrote the modeling doc as guardrails rather than a loose plan
  - mapped the decision target directly to `PropLine`, `PropProjection`, and
    `EdgeDecision`
  - documented allowed source usage and prohibited usage for:
    - Statcast CSV inputs
    - MLB Stats API schedule and lineup inputs
    - sportsbook line snapshots
  - made lineup handling, pricing integrity, and backtest leakage rules
    unambiguous
  - clarified that the repo is not yet committed to a heavy modeling dependency
    such as XGBoost or LightGBM

## Verification Run

These commands were run successfully from the issue branch:

```bash
uv sync --extra dev
uv run pytest
uv run python -m mlb_props_stack
```

Doc-specific verification was also checked directly:

- named sources and endpoints are present in `docs/architecture.md` and
  `docs/modeling.md`
- timestamp rules explicitly mention:
  - `features_as_of`
  - `generated_at`
  - `captured_at`
- leakage language now explicitly forbids substituting closing lines for missing
  pregame snapshots

## Recommended Next Issue

- `AGE-144` — `Ingest schedule, probable starters, and confirmed lineups`

Why this should go next:

- `AGE-143` locked the trusted MLB Stats API sources and the timestamp rules for
  lineup snapshots
- `AGE-144` is the first issue that turns those documented sources into actual
  normalized records
- later issues depend on these joins:
  - `AGE-145` needs deterministic game metadata to map sportsbook events to MLB
    games
  - `AGE-146` needs lineup snapshots and game context to build timestamp-valid
    feature tables

## Constraints For The Next Worktree

- Start from `main` after this docs branch is merged.
- Keep the standard-library-first posture unless the issue explicitly expands
  dependencies.
- Preserve `python -m mlb_props_stack` as a working local entrypoint.
- When implementing `AGE-144`, preserve raw fetch timestamps on every MLB Stats
  API snapshot.
- Normalize lineup snapshots in a way that can satisfy
  `ProjectionInputRef.lineup_snapshot_id` later without retrofitting IDs.
- Do not use `feed/live` data as pregame truth after first pitch. Capture the
  snapshot time and keep pregame and in-game use cases separate.

## Open Questions

- `AGE-144` should settle the exact normalized record shapes for:
  - `games`
  - `probable_starters`
  - `lineup_snapshots`
- `AGE-145` already names The Odds API as the first sportsbook source, but the
  MLB game-to-odds event matching key still needs to be implemented and tested
  against real daily data.
- Weather and umpire inputs remain optional until a future issue introduces a
  timestamp-valid source for them.
