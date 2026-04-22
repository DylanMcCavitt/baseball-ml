# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current issue branch:
  `dylanmccavitt2015/age-146-build-statcast-derived-pitcher-and-opponent-strikeout`
- `main` already includes the merged `AGE-143` docs work, `AGE-144` MLB
  metadata ingest, and `AGE-145` sportsbook ingest.
- This branch adds the first Statcast-derived feature-table build for the
  pitcher strikeout model, but it has not been merged to `main` yet from this
  handoff snapshot.

## What Was Completed In AGE-146

- `src/mlb_props_stack/ingest/statcast_features.py`
  - added a stdlib-only `StatcastSearchClient`
  - added reproducible Baseball Savant Statcast CSV URL construction for
    targeted pitcher and batter pulls
  - sends Statcast requests with an explicit browser-style `User-Agent` header
    so the live CSV export endpoint accepts the pull
  - loads the latest normalized MLB metadata run for the target date from:
    - `games.jsonl`
    - `probable_starters.jsonl`
    - `lineup_snapshots.jsonl`
  - falls back to the latest pregame-valid MLB metadata run when a newer
    same-date metadata run was captured after first pitch
  - preserves raw CSV pulls under:
    - `data/raw/statcast_search/date=YYYY-MM-DD/player_type=.../player_id=.../`
  - writes normalized feature artifacts under:
    - `data/normalized/statcast_search/date=YYYY-MM-DD/run=.../`
  - emits:
    - `pull_manifest.jsonl`
    - `pitch_level_base.jsonl`
    - `pitcher_daily_features.jsonl`
    - `lineup_daily_features.jsonl`
    - `game_context_features.jsonl`
  - dedupes repeated pitch rows across pitcher and batter pulls into one
    traceable `pitch_level_base` seam
  - enforces the current pregame lineup rule:
    - only lineup snapshots with `captured_at <= commence_time` are allowed
    - late lineup snapshots stay explicit as missing instead of leaking
      post-lock information
  - keeps weather and park factor explicit as missing with status fields rather
    than silently backfilling them
- `src/mlb_props_stack/ingest/__init__.py`
  - exported the new Statcast feature-build surface
- `src/mlb_props_stack/cli.py`
  - added:
    - `ingest-statcast-features --date YYYY-MM-DD [--output-dir ...] [--history-days ...]`
  - prints a feature-build summary with history window, pull counts, row counts,
    and normalized artifact paths
- `tests/test_statcast_feature_ingest.py`
  - added end-to-end coverage for:
    - reproducible pitcher and batter pull URLs
    - raw CSV pull writing and pull-manifest output
    - deduped `pitch_level_base` rows across pitcher and batter pulls
    - explicit missing-lineup handling when the only lineup snapshot is late
    - explicit missing weather / park factor status fields
- `tests/test_cli.py`
  - added CLI coverage for the new Statcast feature-build command
- `README.md`
  - documented the new feature-build command, raw/normalized output layout, and
    current missing-input behavior
- `docs/architecture.md`
  - updated the source-adapter table and current artifact-shape section to
    include AGE-146
- `docs/modeling.md`
  - documented the current feature-table outputs and the pregame lineup / missing
    weather guardrails

## Verification Run

These commands were run successfully during AGE-146:

```bash
uv sync --extra dev
python3 -m compileall src tests
uv run pytest
uv run python -m mlb_props_stack
```

Local results:

- `uv run pytest`
  - `36 passed`
- `uv run python -m mlb_props_stack`
  - still prints the runtime summary cleanly

Not run during AGE-146:

```bash
uv run python -m mlb_props_stack ingest-statcast-features --date 2026-04-21
```

Reason:

- the local session verified the new path through deterministic tests instead of
  writing a live feature run against Baseball Savant; a real smoke run still
  needs same-date MLB metadata artifacts plus a decision on where that live
  output should land

## Recommended Next Issue

- `AGE-147` — `Build starter strikeout expectation baseline model`

Why this should go next:

- AGE-146 now provides concrete `pitcher_daily_features`,
  `lineup_daily_features`, and `game_context_features` tables keyed to the same
  slate metadata seam
- AGE-147 can consume those feature tables directly to build the first
  reproducible expected-strikeout baseline before pricing, ladder probabilities,
  or walk-forward backtests

## Constraints For The Next Worktree

- Start the next issue worktree from the current `main` after AGE-146 is merged.
- Treat `pitch_level_base` as the trace seam from raw Statcast CSV rows to any
  later model-training row; do not bypass it with ad hoc historical joins.
- Keep the history window pregame-safe:
  - no Statcast row from the evaluated official date belongs in the daily
    feature tables
- Keep lineup honesty:
  - only lineup snapshots with `captured_at <= commence_time` are allowed into
    model inputs unless a later issue explicitly changes that rule
- Keep missing weather and park factor explicit until a timestamp-valid source
  lands. Do not silently replace the current null-plus-status contract.
- If AGE-147 needs wider or different feature windows, expand them deliberately
  in code or config instead of embedding hidden assumptions in the model layer.

## Open Questions

- A real Statcast smoke run is still needed against live Baseball Savant CSV
  pulls to confirm the current query template behaves cleanly for the targeted
  pitcher and batter workflow.
- The repo still does not have a timestamp-valid weather source for first pitch.
  Weather stays intentionally missing until that source is chosen.
- Park factor is currently just a reserved slot in `game_context_features`; if
  it becomes important for the baseline model, it should come from a named
  source in a dedicated follow-up slice rather than a hardcoded constant.
