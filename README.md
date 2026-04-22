# MLB Props Stack

An MLB props modeling stack focused on narrow, measurable sportsbook markets.

V1 is intentionally scoped to `pitcher strikeout props`, because strikeout outcomes
map cleanly to pitch-level process signals:

- pitcher whiff, CSW, zone, pitch mix, velocity, movement
- hitter strikeout tendencies by handedness
- expected batters faced, leash, and bullpen context
- park, weather, lineup, and umpire adjustments

## Why this repo exists

The goal is not to "predict baseball" in the abstract. The goal is to:

1. estimate event probabilities honestly
2. compare those probabilities to sportsbook prices
3. filter for edges that survive vig, variance, and bad calibration
4. backtest with the same information that would have been available at bet time

That is a math-and-process problem first, and a betting problem second.

## V1 Scope

- one market: pitcher strikeout props
- one modeling target: `P(K >= line + 0.5)` and adjacent ladder probabilities
- one output: edge-ranked candidate props with sizing guidance
- one evaluation loop: walk-forward historical backtest against real book lines

## Non-Goals

- full-game sides and totals
- same-game parlay optimization
- live execution bots
- reinforcement learning in the first modeling pass
- any claim that model outputs alone are "passive income"

## Stack Shape

- `src/mlb_props_stack/config.py`
  Runtime settings and model defaults.
- `src/mlb_props_stack/tracking.py`
  Reserved MLflow-compatible tracking config for later experiment logging.
- `src/mlb_props_stack/pricing.py`
  Odds conversion, expected value, devig, and Kelly sizing.
- `src/mlb_props_stack/markets.py`
  Core data models for props, projections, and decisions.
- `src/mlb_props_stack/ingest/mlb_stats_api.py`
  MLB Stats API adapters for schedule, probable starters, and lineup snapshots.
- `src/mlb_props_stack/ingest/statcast_features.py`
  Targeted Statcast pulls plus normalized pitcher, lineup, and game-context
  feature tables for one slate date.
- `src/mlb_props_stack/edge.py`
  Edge detection and candidate ranking.
- `src/mlb_props_stack/backtest.py`
  Backtest policy and evaluation guardrails.
- `src/mlb_props_stack/dashboard/app.py`
  Placeholder module where the future Streamlit dashboard will live.
- `docs/architecture.md`
  Product and system architecture.
- `docs/modeling.md`
  Data, features, targets, and validation rules.

## Recommended Build Order

1. data contracts for games, pitchers, lines, and line moves
2. feature pipeline from Statcast + schedule + lineups
3. strikeout distribution model
4. calibration layer
5. pricing and edge detection
6. walk-forward backtest
7. paper-trading / tracking dashboard
8. later: RL for sizing or timing, not base prediction

## Development

This repo currently targets Python 3.11+ and uses a small standard-library-first
scaffold until data connectors and model dependencies are added.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python -m mlb_props_stack
```

If you already use `uv`, the equivalent commands are:

```bash
uv sync --extra dev
uv run pytest
uv run python -m mlb_props_stack
```

## MLB Metadata Ingest

`AGE-144` adds the first real source adapter for MLB game context.

Fetch one schedule day, persist raw snapshots, and write normalized JSONL outputs:

```bash
uv run python -m mlb_props_stack ingest-mlb-metadata --date 2026-04-21
```

By default that writes:

- raw schedule snapshots under `data/raw/mlb_stats_api/date=YYYY-MM-DD/schedule/`
- raw `feed/live` snapshots under `data/raw/mlb_stats_api/date=YYYY-MM-DD/feed_live/game_pk=.../`
- normalized `games.jsonl`, `probable_starters.jsonl`, and `lineup_snapshots.jsonl`
  under `data/normalized/mlb_stats_api/date=YYYY-MM-DD/run=.../`

The normalized game rows include an `odds_matchup_key` built from:

- `official_date`
- away team abbreviation
- home team abbreviation
- UTC `commence_time`

That key is the bridge AGE-145 now uses when mapping sportsbook events back to
MLB `gamePk` records.

## Sportsbook Line Ingest

`AGE-145` adds the first sportsbook source adapter via The Odds API MLB
event and event-odds endpoints.

This command expects a prior MLB metadata run for the same date under
`data/normalized/mlb_stats_api/...`, because it reuses the latest `games.jsonl`
and `probable_starters.jsonl` artifacts to map Odds API `event_id` values back
to MLB `gamePk` records and resolve pitcher identities when possible.

Fetch current pitcher strikeout lines for one official date:

```bash
ODDS_API_KEY=YOUR_KEY uv run python -m mlb_props_stack ingest-odds-api-lines --date 2026-04-21
```

By default that writes:

- raw events snapshots under `data/raw/the_odds_api/date=YYYY-MM-DD/events/`
- raw event-odds snapshots under
  `data/raw/the_odds_api/date=YYYY-MM-DD/event_odds/event_id=.../`
- normalized `event_game_mappings.jsonl` and `prop_line_snapshots.jsonl` under
  `data/normalized/the_odds_api/date=YYYY-MM-DD/run=.../`

Each normalized `prop_line_snapshots` row preserves:

- the sportsbook and source `event_id`
- the mapped MLB `gamePk` when the `odds_matchup_key` join succeeds
- the exact two-way line and `market_last_update`
- the ingest `captured_at` timestamp for replayable line history

## Statcast Feature Build

`AGE-146` adds the first feature-table build that turns MLB metadata plus
targeted Statcast pulls into model-ready daily rows.

This command expects a prior MLB metadata run for the same target date under
`data/normalized/mlb_stats_api/...`, because it uses the latest pregame-valid
metadata run for the slate and only accepts lineup snapshots whose
`captured_at` is still on or before the scheduled `commence_time`.

Build one slate's feature tables from the previous `--history-days` official
dates of Statcast history:

```bash
uv run python -m mlb_props_stack ingest-statcast-features --date 2026-04-21
```

By default that writes:

- raw Statcast CSV pulls under
  `data/raw/statcast_search/date=YYYY-MM-DD/player_type=.../player_id=.../`
- a pull manifest under
  `data/normalized/statcast_search/date=YYYY-MM-DD/run=.../pull_manifest.jsonl`
- a normalized `pitch_level_base.jsonl` table
- `pitcher_daily_features.jsonl`
- `lineup_daily_features.jsonl`
- `game_context_features.jsonl`

The current feature build is intentionally explicit about missing inputs:

- lineups captured after first pitch are treated as unavailable for pregame
  feature rows
- weather stays null with `missing_weather_source` until a timestamp-valid
  source is added
- park factor stays null with `missing_park_factor_source` rather than being
  silently backfilled

## CI

GitHub Actions runs the repo baseline checks on pull requests to `main` and on
pushes to `main`:

- `uv sync --locked --extra dev`
- `python -m compileall src tests`
- `uv run pytest`
- `uv run python -m mlb_props_stack`

That keeps the current scaffold honest without pretending the repo already has a
full training or deployment pipeline. As model, feature, and data workflows land,
CI can grow from this baseline instead of starting as placeholder ceremony.

## Future Hooks

- `mlb_props_stack.tracking.TrackingConfig` is the reserved place for future
  MLflow tracking configuration.
- `mlb_props_stack.dashboard.app` is the reserved dashboard entrypoint for a
  later Streamlit UI.
- Neither MLflow nor Streamlit is installed in v1; the baseline only preserves
  clean seams for those additions.

## Risk

This project should be treated as a research and decision-support system.
Sportsbooks price efficiently enough that sloppy data handling, leakage, bad
calibration, and untracked line movement can erase a paper edge very quickly.
