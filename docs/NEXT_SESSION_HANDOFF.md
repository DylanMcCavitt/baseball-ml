# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Active issue branch: `feat/age-188-fix-odds-api-joins`
- `main` already includes the merged runtime-review follow-up after `AGE-153`
- This branch contains the `AGE-188` odds-ingest fix and is not merged yet

## What Was Completed In AGE-188

- `src/mlb_props_stack/ingest/odds_api.py`
  - keeps `event_game_mappings.jsonl` as the audit trail for matched and
    unmatched Odds API events
  - stops unmatched event mappings from flowing into
    `prop_line_snapshots.jsonl`
  - adds explicit ingest counts for:
    - skipped unmatched events
    - matched events with no pitcher-strikeout markets
    - resolved vs unresolved pitcher prop rows
- `src/mlb_props_stack/cli.py`
  - surfaces the new Odds API ingest summary counts so runtime review can tell
    the difference between a join regression and honest upstream market
    unavailability
- `src/mlb_props_stack/env.py`
  - adds a stdlib-only repo `.env` loader that does not override already-set
    environment variables
  - in a git worktree, falls back from the current worktree root to a sibling
    checkout from the same repo so one ignored `.env` in the canonical checkout
    still works in fresh issue worktrees
  - now merges candidate `.env` files in priority order instead of stopping at
    the first file, so a partial worktree-local `.env` does not block fallback
    to the canonical checkout's `ODDS_API_KEY`
  - correctly strips quoted values even when they carry trailing inline comments
- `src/mlb_props_stack/ingest/odds_api.py`
  - now calls the repo `.env` loader before reading `ODDS_API_KEY`
- `tests/test_odds_api_ingest.py`
  - adds a regression that reproduces the same-team dual-event shape from the
    live April 22 review:
    - one unmatched event with prop markets
    - one matched event for the target date
  - verifies that only the matched event persists prop rows and that those rows
    carry `game_pk` plus `pitcher_mlb_id`
  - verifies that `OddsAPIClient()` can load `ODDS_API_KEY` from the repo env
    path when the shell environment is missing it
- `tests/test_cli.py`
  - locks the new ingest summary fields
- `tests/test_env.py`
  - locks direct `.env` parsing plus worktree fallback to a sibling `main`
    checkout
- `README.md`
  - documents the audit-vs-scoring split between
    `event_game_mappings.jsonl` and `prop_line_snapshots.jsonl`
  - documents the repo-local `.env` loading behavior for worktree-based runs
- `docs/review_runtime_checks.md`
  - updates the runtime-review guidance so future threads know that
    `matched_events_without_props` can indicate honest upstream emptiness rather
    than another join failure

## Files Changed

- `README.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/review_runtime_checks.md`
- `src/mlb_props_stack/__init__.py`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/env.py`
- `src/mlb_props_stack/ingest/odds_api.py`
- `tests/test_cli.py`
- `tests/test_env.py`
- `tests/test_odds_api_ingest.py`

## Verification

Commands run successfully on April 22, 2026:

```bash
uv sync --extra dev
uv run pytest tests/test_env.py tests/test_odds_api_ingest.py
uv run pytest
uv run python -m mlb_props_stack
uv run python -m compileall src tests
env -u ODDS_API_KEY uv run python - <<'PY'
from mlb_props_stack.ingest.odds_api import OddsAPIClient
client = OddsAPIClient()
print('api_key_loaded', bool(client.api_key))
PY
PYTHONPATH=/tmp/age188-odds-check.EziJdd/stub ODDS_API_KEY=stub-key \
  uv run python -m mlb_props_stack ingest-odds-api-lines \
  --date 2026-04-21 \
  --output-dir /tmp/age188-odds-check.EziJdd
```

Observed results:

- focused pytest run:
  - `10 passed`
- full pytest run:
  - `62 passed`
- `uv run python -m mlb_props_stack`
  - printed the runtime summary successfully
- `uv run python -m compileall src tests`
  - compiled all source and test modules successfully
- direct env fallback runtime check:
  - `env -u ODDS_API_KEY ... OddsAPIClient()` printed `api_key_loaded True`
    from the worktree, proving the ignored canonical-checkout `.env` is now
    usable without exporting the key first
- seeded CLI ingest runtime check:
  - summary reported `candidate_events=2`, `matched_events=1`,
    `unmatched_events=1`, `skipped_unmatched_events=1`,
    `matched_events_without_props=0`, and `prop_line_snapshots=2`
  - `event_game_mappings.jsonl` kept both matched and unmatched events
  - `prop_line_snapshots.jsonl` only contained the matched event and both rows
    had non-null `game_pk` plus `pitcher_mlb_id`

## Recommended Next Issue

- `AGE-189`
  Add fixture-backed runtime smoke coverage for dashboard boot and representative
  pipeline entrypoints

Why this should go next:

- `AGE-188` fixed the ingest contract so unmatched events no longer poison
  `prop_line_snapshots`
- the remaining gap is faster runtime detection for dashboard and pipeline
  regressions before merge

## Constraints And Open Questions

- Keep the current audit split:
  - unmatched events belong in `event_game_mappings.jsonl`
  - only matched target-date events belong in `prop_line_snapshots.jsonl`
- Do not synthesize `pitcher_mlb_id` values when the matched event's player
  name cannot be resolved honestly against probable starters
- For future live-slate reviews, inspect `matched_events_without_props` before
  treating an empty slate as a join bug
- April 22, 2026 live raw payloads for several matched April 23 events returned
  `bookmakers: []`; if a later issue tries to make future slates non-empty
  earlier in the day, verify upstream market availability first
