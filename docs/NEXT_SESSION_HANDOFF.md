# Next Session Handoff

## Current State

- Repo: `baseball-ml`
- Default branch: `main`
- Current issue branch:
  `feat/age-144-ingest-schedule-probable-starters-and-confirmed-lineups`
- `main` already includes the merged `AGE-143` docs work from PR #3.
- `AGE-144` is complete locally on this branch. The repo now has a working
  stdlib-only MLB Stats API ingest path for:
  - one-day schedule pulls keyed by `gamePk`
  - probable starter normalization
  - per-team lineup snapshots from `feed/live`
  - raw JSON snapshot persistence
  - normalized JSONL outputs for `games`, `probable_starters`, and
    `lineup_snapshots`

## What Was Completed In AGE-144

- `src/mlb_props_stack/ingest/mlb_stats_api.py`
  - added the first real source adapter in the repo
  - fetches:
    - `schedule?sportId=1&date=...&hydrate=probablePitcher(note),team`
    - `game/{gamePk}/feed/live`
  - preserves raw schedule and feed payloads under `data/raw/mlb_stats_api/...`
  - normalizes typed records for:
    - `games`
    - `probable_starters`
    - `lineup_snapshots`
  - emits a deterministic `odds_matchup_key` from:
    - `official_date`
    - away team abbreviation
    - home team abbreviation
    - UTC `commence_time`
- `src/mlb_props_stack/ingest/__init__.py`
  - exported the new ingest API surface for CLI and future adapters
- `src/mlb_props_stack/cli.py`
  - kept the no-arg runtime summary intact
  - added:
    - `ingest-mlb-metadata --date YYYY-MM-DD [--output-dir ...]`
  - prints the run id plus artifact paths and record counts
- `tests/test_mlb_stats_api_ingest.py`
  - added normalization and filesystem-write coverage for AGE-144
- `tests/test_cli.py`
  - added CLI coverage for the new ingest command
- `README.md`
  - documented the new ingest command and output layout
- `docs/architecture.md`
  - documented the new ingest seam and the normalized artifact shape

## Verification Run

These commands were run successfully from the issue branch:

```bash
uv sync --extra dev
uv run pytest
uv run python -m mlb_props_stack
uv run python -m mlb_props_stack ingest-mlb-metadata --date 2026-04-21 --output-dir /tmp/mlb-props-stack-age144-check
```

The live ingest smoke run against the real MLB Stats API produced:

- `games=15`
- `probable_starters=30`
- `lineup_snapshots=30`

Row-count verification on the normalized outputs confirmed:

- one `games` row per schedule game
- two `probable_starters` rows per game
- two `lineup_snapshots` rows per game

Sample normalized rows were checked directly for:

- `captured_at`
- `game_pk`
- `odds_matchup_key`
- ordered lineup entries and player names

## Recommended Next Issue

- `AGE-145` — `Ingest sportsbook pitcher strikeout lines and capture snapshots`

Why this should go next:

- AGE-144 now produces the MLB-side join material AGE-145 needs:
  - `gamePk`
  - away and home team metadata
  - UTC `commence_time`
  - deterministic `odds_matchup_key`
- the next blocker is the sportsbook side of the same join:
  - raw Odds API event and market snapshots
  - normalized `prop_line_snapshots`
  - mapping Odds API event ids back to MLB `gamePk`

## Constraints For The Next Worktree

- Start from `main` after this ingest branch is merged.
- Keep the standard-library-first posture unless the issue explicitly expands
  dependencies.
- Preserve `python -m mlb_props_stack` as a working local entrypoint.
- Reuse `odds_matchup_key` exactly as implemented in
  `src/mlb_props_stack/ingest/mlb_stats_api.py`; do not invent a second game
  matching key on the sportsbook side.
- Preserve repeated sportsbook snapshots instead of overwriting prior line
  states; AGE-145 needs replayable line history.
- Keep sportsbook event ids and MLB `gamePk` separate. The mapping table should
  bridge them, not collapse them into one field.

## Open Questions

- Some pregame `feed/live` captures still have one team lineup populated and the
  other empty. AGE-145 and AGE-146 should treat `is_confirmed=False` snapshots as
  real captured state, not as ingest failures.
- AGE-145 still has to confirm how strictly The Odds API `commence_time` matches
  MLB schedule `gameDate` in practice for doubleheaders and late schedule moves.
- Weather and umpire inputs remain optional until a future issue introduces a
  timestamp-valid source for them.
