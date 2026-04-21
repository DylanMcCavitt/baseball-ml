# Architecture

## V1 Boundary

This repository is not a generic baseball prediction project.

V1 is a narrow decision system for one market only:

- pitcher strikeout props

The product question is simple:

1. estimate the probability that a starting pitcher finishes over or under a
   posted strikeout line
2. compare that estimate to the actual sportsbook price that was available at
   decision time
3. rank only the props that still clear vig, threshold, and sizing rules
4. evaluate those decisions with timestamp-valid walk-forward backtests

## Trusted Upstream Sources

The system is allowed to trust these source families for v1.

| Source family | Named endpoint or source | What it supplies | What it must not be used for |
| --- | --- | --- | --- |
| Pitch-level history | Baseball Savant Statcast Search CSV via `https://baseballsavant.mlb.com/statcast_search` and CSV export from `https://baseballsavant.mlb.com/statcast_search/csv` | pitch outcomes, pitch type, velocity, movement, whiff / called-strike context, pitcher and batter IDs | any row dated after the feature cutoff for the evaluated prop |
| Metric dictionary | Baseball Savant CSV docs at `https://baseballsavant.mlb.com/csv-docs` | field meanings for Statcast CSV columns | inventing derived metrics without documenting the transformation |
| Schedule and probable starters | MLB Stats API `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=YYYY-MM-DD&hydrate=probablePitcher` | game IDs, official dates, teams, venues, probable starters | replacing real pregame snapshots with postgame truth |
| Pregame or confirmed lineups | MLB Stats API `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=YYYY-MM-DD&hydrate=lineups` | lineup snapshots keyed to the game | pretending a confirmed lineup existed before it was published |
| Game context and official game feed | MLB Stats API `https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live` | final lineup state, official game metadata, scorebook context, player IDs | pregame feature generation once the feed contains in-game or postgame information |
| Sportsbook prices | sportsbook-specific pitcher strikeout prop snapshots captured into the repo's future `PropLine`-shaped records | sportsbook name, posted line, over odds, under odds, capture timestamp | backfilling missing pregame prices with closing lines or generic market averages |

The sportsbook integration is intentionally not pinned to one vendor yet. What
matters in v1 is that every decision is backed by a real pregame snapshot with
the fields required by `PropLine`.

## Repo Contract Spine

The current codebase already defines the seams that future ingestion and
modeling work must honor.

| Layer | Current module | Current responsibility |
| --- | --- | --- |
| Runtime defaults | `src/mlb_props_stack/config.py` | market name, edge threshold, Kelly fraction, bankroll cap, timezone |
| Prop and projection contracts | `src/mlb_props_stack/markets.py` | `PropLine`, `PropProjection`, `EdgeDecision`, `PropSelectionKey`, `ProjectionInputRef` |
| Source adapters | `src/mlb_props_stack/ingest/mlb_stats_api.py` | fetch schedule and `feed/live` payloads, preserve raw snapshots, normalize `games`, `probable_starters`, and `lineup_snapshots` |
| Pricing math | `src/mlb_props_stack/pricing.py` | American odds conversion, devig, fair odds, expected value, fractional Kelly |
| Decision layer | `src/mlb_props_stack/edge.py` | match line and projection contracts, enforce timestamp order, emit the best actionable side |
| Evaluation guardrails | `src/mlb_props_stack/backtest.py` | walk-forward policy flags and the baseline honesty checklist |
| Reserved future seams | `src/mlb_props_stack/tracking.py`, `src/mlb_props_stack/dashboard/app.py` | tracking and dashboard entrypoints only, not full implementations yet |

The most important current contract boundary is the join between
`PropLine.selection_key` and `PropProjection.selection_key`. Future adapters can
change where data comes from, but they cannot silently weaken that join key or
drop the timestamp fields that make the backtest honest.

## System Flow

```text
Baseball Savant Statcast Search CSV
        +
MLB Stats API schedule / probable starters / lineups
        +
Sportsbook strikeout prop snapshots
        ->
normalized contract records
        ->
feature row + lineup snapshot references
        ->
prop projection
        ->
devig + EV + Kelly sizing
        ->
edge-ranked candidate decision
        ->
walk-forward evaluation with CLV and ROI reported separately
```

In code terms, that flow should eventually materialize as:

1. source adapters produce timestamped records
2. those records are normalized into `PropLine` plus a versioned feature row and
   lineup snapshot
3. the model emits a `PropProjection`
4. `evaluate_projection()` in `src/mlb_props_stack/edge.py` compares the
   projection to the market
5. `BacktestPolicy` and `BACKTEST_CHECKLIST` define which historical runs are
   considered valid

## Current AGE-144 Output Shape

The first ingest slice now writes both raw and normalized artifacts locally.

- raw schedule payloads:
  `data/raw/mlb_stats_api/date=YYYY-MM-DD/schedule/captured_at=...json`
- raw `feed/live` payloads:
  `data/raw/mlb_stats_api/date=YYYY-MM-DD/feed_live/game_pk=.../captured_at=...json`
- normalized runs:
  `data/normalized/mlb_stats_api/date=YYYY-MM-DD/run=.../`

The normalized files are:

- `games.jsonl`
  one row per `gamePk` with team metadata, venue, schedule status, and
  `odds_matchup_key`
- `probable_starters.jsonl`
  one row per team side per game with the captured probable starter state
- `lineup_snapshots.jsonl`
  one row per team side per `feed/live` capture with `lineup_snapshot_id`,
  `captured_at`, ordered batter IDs, and detailed lineup entries

`odds_matchup_key` is currently:

- `official_date`
- away team abbreviation
- home team abbreviation
- UTC `commence_time`

joined into one deterministic string. That is the bridge for AGE-145 when Odds
API events have to be matched back to MLB Stats API games without relying on
shared vendor IDs.

## Timestamp Authority

V1 documentation is only useful if the timestamps are unambiguous.

- `features_as_of`
  newest timestamp of any feature input referenced by `ProjectionInputRef`
- `generated_at`
  when the model output was produced
- `captured_at`
  when the sportsbook line snapshot was recorded

Current code already enforces:

- `features_as_of <= generated_at`
- `features_as_of <= captured_at`
- `generated_at <= captured_at`

That means the model may not use a lineup, pitch log, or market state that did
not exist when the price snapshot was captured.

## Non-Goals

These stay out of scope unless a later issue changes the boundary explicitly.

- full-game sides or totals
- same-game parlay optimization
- live betting or in-game decisioning
- automated bet placement
- portfolio optimization across many correlated markets
- dependency-heavy platform work such as MLflow, Streamlit, Plotly, schedulers,
  or storage systems during the bootstrap phase

## Live-Use Caveats

This repo should be treated as a research and decision-support system, not an
execution bot.

- No output is trustworthy unless the upstream line snapshot is real and
  timestamped.
- CLV and realized ROI answer different questions and must never be collapsed
  into one metric.
- If a prop cannot be tied back to a specific lineup snapshot and feature row,
  it should not be promoted to live tracking.
- A model that looks profitable only after substituting later lineups, later
  prices, or closing lines is invalid, even if the headline ROI is positive.
