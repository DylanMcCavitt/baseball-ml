# Next Session Handoff

## Status

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Active issue: `AGE-293` - fix scoreable historical market joins for
  strikeout prop backtests
- Current issue branch:
  `dylanmccavitt2015/age-293-fix-scoreable-historical-market-joins-for-strikeout-prop`
- Base state: `HEAD` and `origin/main` both started at
  `14932bf021a8dce6100f0fb5dfc22b1a6cdb48ea`.
- PR: https://github.com/DylanMcCavitt/baseball-ml/pull/51
- Implementation state: code, focused/runtime verification, commit, push, and
  PR creation are complete. Linear is ready to remain in `In Review` for human
  review.

## What Changed In AGE-293

- Updated `build_walk_forward_backtest()` snapshot grouping so resolved
  historical line snapshots are grouped by the stable MLB contract when both
  `game_pk` and `pitcher_mlb_id` are present:
  official date, sportsbook, market, exact line, MLB game, and MLB pitcher.
- Kept unresolved or partially mapped snapshots on source identifiers instead
  of fabricating joins.
- Preserved alternate lines as separate groups by keeping exact line in the
  grouping key.
- Changed selected contract metadata to use the selected cutoff-safe snapshot
  when a group contains historical source `event_id` or `player_id` drift.
- Added explicit projection timestamp validation against the selected line
  snapshot:
  `features_as_of <= selected captured_at` and
  `projection_generated_at <= selected captured_at`.
- Added `timestamp_invalid_projection` as a separate skip reason before pricing
  analysis can run.
- Added join-audit fields for selected match status, selected game mapping,
  selected pitcher mapping, and projection timestamp status.
- Hardened mapping checks so blank mapping fields are treated as missing.

## Coverage Evidence

- Fixture-backed runtime smoke:
  `/tmp/age293-backtest-runtime` contains a representative historical
  `2026-04-20` replay with source event/player drift across snapshots for the
  same resolved MLB game and pitcher.
- CLI result:
  `snapshot_groups=2`, `actionable_bets=1`, `skipped=1`,
  `skip_reason_counts={"unmatched_event_mapping": 1}`.
- Inspected artifacts under:
  `/tmp/age293-backtest-runtime/normalized/walk_forward_backtest/start=2026-04-20_end=2026-04-20/run=20260428T005335Z/`
  - `backtest_bets.jsonl`: one `actionable` row and one
    `unmatched_event_mapping` row.
  - `join_audit.jsonl`: scoreable row has selected game/pitcher mapping true
    and `projection_timestamp_status=ok`; unmatched row has both selected
    mapping flags false.
  - `backtest_runs.jsonl`: row counts and skip reasons match the CLI summary.

## Historical Market Coverage Caveat

- This worktree has no persisted real historical Odds API market artifacts
  beyond static data.
- The canonical checkout at `/Users/dylanmccavitt/projects/nba-ml` also has no
  `data/normalized/the_odds_api` or `data/normalized/starter_strikeout_baseline`
  artifacts available during this session.
- Therefore no real MLB season/date window can be certified here as having
  sufficient market coverage for betting-layer validation.
- The corrected path is verified for covered fixture data on `2026-04-20`.
  Real season/date certification still requires backfilling or restoring
  historical Odds API line artifacts plus a compatible starter-strikeout
  baseline run.

## Files Changed

- `src/mlb_props_stack/backtest.py`
- `tests/test_backtest.py`
- `docs/NEXT_SESSION_HANDOFF.md`

## Verification

Commands run:

```bash
/opt/homebrew/bin/uv sync --extra dev
/opt/homebrew/bin/uv run pytest tests/test_backtest.py -q
/opt/homebrew/bin/uv run pytest tests/test_runtime_smokes.py -q
/opt/homebrew/bin/uv run pytest
PYTHONPYCACHEPREFIX=/tmp/age293-pycache python3 -m compileall src tests
/opt/homebrew/bin/uv run python -m mlb_props_stack
/opt/homebrew/bin/uv --project /Users/dylanmccavitt/.codex/worktrees/symphony-nba-ml/AGE-293 run python -m mlb_props_stack build-walk-forward-backtest --start-date 2026-04-20 --end-date 2026-04-20 --output-dir /tmp/age293-backtest-runtime --cutoff-minutes-before-first-pitch 30
```

Observed results:

- `uv sync --extra dev`: completed successfully.
- Focused backtest tests: `5 passed` with existing third-party
  MLflow/Pydantic warnings.
- Runtime smoke tests: `5 passed` with existing third-party MLflow/Pydantic
  warnings.
- Full test suite: `224 passed` with existing third-party MLflow/Pydantic
  warnings.
- `compileall`: completed successfully.
- `python -m mlb_props_stack`: printed the runtime configuration banner.
- Fixture-backed CLI backtest: completed successfully with one scoreable row
  and explicit unmatched-event audit output.

## Recommended Next Issue

1. Backfill or restore real historical Odds API pitcher strikeout line
   artifacts and compatible baseline probability artifacts for the intended
   validation window.
2. Rerun `build-walk-forward-backtest` on that real covered window and record
   which dates/seasons produce scoreable rows.
3. Keep downstream betting-layer validation blocked until real historical
   coverage is certified and model-only validation has a defensible go/no-go
   result with feature coverage.

## Constraints And Risks

- Do not treat the fixture-backed `2026-04-20` smoke as betting evidence.
  It only proves the corrected historical join path and audit behavior.
- Do not loosen timestamp guards. The implementation now rejects projections
  whose `features_as_of` or `projection_generated_at` lands after the selected
  sportsbook snapshot.
- Do not fabricate event or pitcher joins when `game_pk` or `pitcher_mlb_id`
  is missing or blank; those rows stay skipped with explicit reasons.
- Do not resume pricing, approval-gate, paper-tracking, dashboard, or live
  ingest work from this issue alone.
