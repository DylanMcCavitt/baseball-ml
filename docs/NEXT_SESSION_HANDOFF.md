# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Last completed issue branch:
  `dylanmccavitt2015/age-145-ingest-sportsbook-pitcher-strikeout-lines-and-capture`
- `main` already includes the merged `AGE-143` docs work and the merged
  `AGE-144` MLB metadata ingest work.
- `main` now also includes the merged `AGE-145` sportsbook ingest work from
  PR #5.
- The repo now has a working The Odds API ingest path for sportsbook pitcher
  strikeout props that reuses the latest MLB metadata artifacts for the same
  official date.

## What Was Completed In AGE-145

- `src/mlb_props_stack/ingest/odds_api.py`
  - added a stdlib-only `OddsAPIClient` with:
    - `GET /v4/sports/baseball_mlb/events`
    - `GET /v4/sports/baseball_mlb/events/{eventId}/odds?regions=us&markets=pitcher_strikeouts&oddsFormat=american`
  - loads the latest normalized MLB metadata run for the target date from:
    - `games.jsonl`
    - `probable_starters.jsonl`
  - preserves raw Odds API event snapshots under:
    - `data/raw/the_odds_api/date=YYYY-MM-DD/events/`
  - preserves raw event-odds snapshots under:
    - `data/raw/the_odds_api/date=YYYY-MM-DD/event_odds/event_id=.../`
  - writes normalized sportsbook artifacts under:
    - `data/normalized/the_odds_api/date=YYYY-MM-DD/run=.../`
  - emits:
    - `event_game_mappings.jsonl`
    - `prop_line_snapshots.jsonl`
  - reuses the existing `odds_matchup_key` contract to map Odds API `event_id`
    values back to MLB `gamePk`
  - resolves `player_id` from probable starters when the event-to-game join is
    clean, and falls back to deterministic `odds-player:...` ids when the join
    is still unresolved
  - keeps unmatched event rows instead of discarding them so schedule / time
    mismatches can be inspected later
- `src/mlb_props_stack/ingest/__init__.py`
  - exported the new sportsbook ingest surface
- `src/mlb_props_stack/cli.py`
  - added:
    - `ingest-odds-api-lines --date YYYY-MM-DD [--output-dir ...] [--api-key ...]`
  - prints a sportsbook ingest summary with event counts, mapping counts, and
    normalized artifact paths
- `tests/test_odds_api_ingest.py`
  - added end-to-end coverage for:
    - writing raw and normalized sportsbook artifacts
    - preserving prior run directories and raw snapshots across repeated ingests
    - unmatched event mappings and deterministic fallback player ids
- `tests/test_cli.py`
  - added CLI coverage for the new sportsbook ingest command
- `README.md`
  - documented the new sportsbook ingest command, env var requirement, and
    output layout
- `docs/architecture.md`
  - updated the trusted sportsbook source, source-adapter table, and current
    ingest artifact shape to include AGE-145
- `docs/modeling.md`
  - documented current `player_id` resolution behavior for sportsbook snapshots

## Verification Run

These commands were run successfully during AGE-145:

```bash
uv sync --extra dev
python3 -m compileall src tests
uv run pytest
uv run python -m mlb_props_stack
```

Local results:

- `uv run pytest`
  - `30 passed`
- `uv run python -m mlb_props_stack`
  - still prints the runtime summary cleanly

Not run during AGE-145:

```bash
uv run python -m mlb_props_stack ingest-odds-api-lines --date 2026-04-21
```

Reason:

- `ODDS_API_KEY` was not present in the environment during this session, so the
  live sportsbook smoke run could not be exercised against the real API.

## Recommended Next Issue

- `AGE-146` — `Build Statcast-derived pitcher and opponent strikeout feature tables`

Why this should go next:

- AGE-144 now provides timestamped MLB game, probable-starter, and lineup
  artifacts.
- AGE-145 now provides replayable sportsbook line snapshots plus event-to-game
  mappings for the same official date.
- AGE-146 can build the first real feature tables against those upstream seams
  without guessing about market timestamps or event identity.

## Constraints For The Next Worktree

- Start the next issue worktree from the current `main`.
- `ingest-odds-api-lines` expects an existing MLB metadata run for the same date
  under `data/normalized/mlb_stats_api/...`; do not bypass that dependency by
  inventing a second game-matching path.
- Keep Odds API `event_id` and MLB `gamePk` separate. The mapping table is the
  bridge and should remain explicit.
- Preserve repeated sportsbook snapshots. The run-specific normalized directories
  and timestamped raw files are required for replayable line history.
- Keep unmatched event mappings in the normalized output until the actual live
  mismatch pattern is understood.
- Keep `odds-player:...` ids as a fallback only. When the sportsbook event maps
  cleanly and the probable-starter name matches, prefer the MLB pitcher id.

## Open Questions

- A real-key smoke run is still needed to confirm how often The Odds API
  `commence_time` misses the exact MLB `odds_matchup_key` join in practice,
  especially for doubleheaders and late schedule changes.
- If real sportsbook payloads show frequent player-name variants beyond the
  current probable-starter names, the repo may need an explicit alias layer
  before model training consumes these records.
- `AGE-146` still needs to decide the exact timestamp-valid weather source.
  Missing weather is currently allowed and should stay explicit rather than
  silently backfilled.
