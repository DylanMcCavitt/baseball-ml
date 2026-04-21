# Modeling Guardrails

## V1 Market

The only in-scope market is:

- pitcher strikeout props

Typical two-way lines are `3.5`, `4.5`, `5.5`, and `6.5`, with alternate
ladders derived from the same strikeout-count distribution.

## Decision Target

The repo is trying to answer one decision question per posted line:

- what is the probability that this pitcher finishes over or under the posted
  strikeout line at the time this sportsbook snapshot was captured?

That maps directly to the current contracts:

- `PropLine`
  one sportsbook snapshot for one pitcher strikeout market
- `PropProjection`
  the model estimate for that exact contract
- `EdgeDecision`
  the priced action after devig, EV, and sizing

## Allowed Source Inventory

### 1. Baseball Savant Statcast Search CSV

Named source:

- `https://baseballsavant.mlb.com/statcast_search`
- CSV export shape served from `https://baseballsavant.mlb.com/statcast_search/csv`
- field dictionary at `https://baseballsavant.mlb.com/csv-docs`

Allowed uses:

- pitch-type usage
- whiff and called-strike skill
- velocity and movement baselines
- batter strikeout tendency splits
- rolling pitcher and opponent process metrics

Not allowed:

- any plate appearance or pitch that happened after `features_as_of`
- leaking same-day outcomes from games that had not started at decision time
- treating postgame aggregates as if they were known pregame

### 2. MLB Stats API

Named sources:

- `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=YYYY-MM-DD&hydrate=probablePitcher`
- `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=YYYY-MM-DD&hydrate=lineups`
- `https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live`

Allowed uses:

- official game IDs and dates
- probable starter identity
- venue and schedule context
- lineup snapshots keyed to game and timestamp
- player ID resolution across sources

Not allowed:

- using the live game feed for pregame features after first pitch
- silently replacing a projected lineup snapshot with a later confirmed lineup
- pulling realized in-game stats into a pregame feature row

### 3. Sportsbook strikeout prop snapshots

Named source class:

- sportsbook-specific over or under pitcher strikeout snapshots captured into
  `PropLine`-compatible records

Required fields:

- `sportsbook`
- `event_id`
- `player_id`
- `player_name`
- `market`
- `line`
- `over_odds`
- `under_odds`
- `captured_at`

Current ingest behavior:

- AGE-145 resolves `player_id` from the latest MLB probable-starter artifact
  when the sportsbook event maps cleanly to one `gamePk`
- if that join is still unresolved, the ingest keeps the snapshot with a
  deterministic name-based fallback `player_id` so the raw market history is
  preserved instead of discarded

Allowed uses:

- direct market comparison
- devig and implied probability estimation
- CLV tracking relative to later snapshots

Not allowed:

- substituting a later or closing number for a missing pregame snapshot
- averaging multiple books together without preserving the original book line
- evaluating model edges against a line the bettor could not actually access

## Feature Families

Feature work should stay legible and traceable to the source inventory above.

### Pitcher process

- rolling strikeout rate
- rolling CSW rate
- rolling whiff rate
- zone rate
- chase induced
- pitch-type mix
- pitch-type whiff rates
- velocity and movement deltas

Primary source:

- Baseball Savant Statcast Search CSV

### Opponent strikeout profile

- expected lineup strikeout rate by handedness
- swing aggression and chase profile
- contact quality and whiff susceptibility by pitch type
- concentration of strikeout-prone bats near the top of the order

Primary sources:

- Baseball Savant Statcast Search CSV
- MLB Stats API lineup snapshots

### Usage and leash

- recent pitch count
- recent batters faced
- inning depth
- removal risk tied to team or bullpen state

Primary sources:

- Baseball Savant Statcast Search CSV
- MLB Stats API schedule and game context

### Environment

- park
- rest
- travel
- weather when a trustworthy pregame source is added later
- umpire when a trustworthy pregame source is added later

Primary sources:

- MLB Stats API schedule and venue context

Weather and umpire adjustments stay optional until they can be sourced with
their own timestamp-valid snapshots. They are not excuses to inject vague or
manually remembered context.

## Model Shape

The docs should define the modeling job without pretending the implementation
choice is already settled.

Required modeling behavior:

1. produce an expected strikeout signal for a pitcher-game matchup
2. convert that expectation into an over or under probability for the posted
   line
3. keep calibration explicit and testable out of sample

Candidate implementation path:

- baseline: simple count model or other standard-library-friendly prototype
- later production candidate: tree-based regressor or classifier in a dedicated
  issue that expands dependencies deliberately
- calibration: explicit post-model calibration on out-of-fold or walk-forward
  predictions

This repo is not committed to XGBoost, LightGBM, or any other dependency yet.
The hard requirement is honest probability estimation, not a specific library.

## Leakage Rules

These rules are mandatory, not suggestions.

### Timestamp ordering

For every evaluated prop:

- `ProjectionInputRef.features_as_of <= PropProjection.generated_at`
- `ProjectionInputRef.features_as_of <= PropLine.captured_at`
- `PropProjection.generated_at <= PropLine.captured_at`

If that ordering cannot be proven, the record is invalid.

### Lineup handling

- Every evaluated projection must point to a specific `lineup_snapshot_id`.
- If a lineup is projected rather than confirmed, that projected snapshot must
  still be versioned and timestamped.
- A later confirmed lineup cannot be swapped into an earlier decision record.

### Training and backtest windows

- training data for an evaluated day must stop before that day begins
- feature windows may summarize prior games only
- no feature may incorporate results from the evaluated game itself
- rejected props should be stored so threshold changes can be audited later

### Pricing integrity

- compare the model only to the actual book line snapshot that existed at
  decision time
- report CLV separately from realized ROI
- keep vig in the pricing path instead of treating the board as fair by default

## Promotion Criteria

A model is not ready for live tracking until it can demonstrate all of the
following on held-out, walk-forward evaluation:

- stable calibration across multiple line buckets
- positive median CLV
- positive ROI over a meaningful sample
- no dependence on one short time window or one narrow team cluster
- sensible degradation when vig assumptions tighten

If those conditions are not met, the system is still research code, even if a
few backtest slices look good.
