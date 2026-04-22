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

## Current AGE-146 And AGE-147 Feature And Training Artifacts

The current feature build writes four concrete artifacts for one target date:

- `pitch_level_base`
  normalized pitch rows from targeted Statcast pulls for the slate's probable
  starters and any pregame-valid opponent lineup hitters
- `pitcher_daily_features`
  one row per probable starter with rolling strikeout rate, whiff rate, CSW,
  pitch mix, release-speed deltas, release-extension deltas, recent workload,
  and rest
- `lineup_daily_features`
  one row per probable starter with opponent-lineup strikeout, chase, contact,
  continuity, and confirmation fields
- `game_context_features`
  one row per probable starter with venue, home/away, rest, and expected leash
  proxies

Current implementation details that later issues must preserve unless they
explicitly replace them:

- the Statcast history window stops at the prior official date, never the
  evaluated date itself
- lineup features only use lineup snapshots whose `captured_at` is still on or
  before the scheduled first pitch
- lineup rows stay explicit when the lineup is missing or late; they do not
  silently swap in a later confirmed lineup
- weather and park factor currently remain null with explicit status fields
  until a timestamp-valid source is added

AGE-147 now builds the first reproducible starter strikeout training dataset on
top of those feature tables.

Training command:

```bash
uv run python -m mlb_props_stack train-starter-strikeout-baseline \
  --start-date 2026-04-01 \
  --end-date 2026-04-20
```

Current AGE-147 behavior:

- joins `pitcher_daily_features`, `lineup_daily_features`, and
  `game_context_features` by `official_date`, `gamePk`, and `pitcher_id`
- fetches same-day Statcast pitcher rows and counts strikeout events on final
  pitches to define the official starter strikeout target
- saves train, validation, and test splits by date in `date_splits.json`
- writes a naive benchmark based on
  `pitcher_k_rate * expected_leash_batters_faced`
- trains a deterministic ridge-style linear regressor with only pregame-valid
  feature fields
- reports RMSE, MAE, and Spearman rank correlation plus coefficient-based
  feature importance in `evaluation.json`

AGE-148 adds the first explicit count-distribution layer on top of that mean:

- the ridge baseline still owns the expected strikeout mean
- a single global negative-binomial dispersion parameter is then fit from the
  training rows using method-of-moments
- half-strikeout ladder probabilities are derived from that fitted count
  distribution and written per starter-game to `ladder_probabilities.jsonl`
- `evaluation.json` now compares the fitted negative-binomial layer against a
  Poisson fallback using the same mean predictions on train, validation, test,
  and combined held-out rows

AGE-149 adds explicit probability calibration on top of those raw ladders:

- out-of-fold ladder-event probabilities are generated with expanding
  date-ordered fits so each predicted date only sees prior data
- an isotonic probability calibrator is fit from those out-of-fold rows and
  stored in `probability_calibrator.json`
- `raw_vs_calibrated_probabilities.jsonl` keeps honest held-out rows with both
  raw and calibrated over/under probabilities for diagnostics
- `calibration_summary.json` records reliability bins plus Brier, log-loss, and
  expected calibration error for raw vs calibrated probabilities
- `ladder_probabilities.jsonl` now carries both the raw ladder and a calibrated
  ladder so later pricing work can consume the calibrated side directly

AGE-150 turns those saved ladders into replayable pricing rows:

- `ladder_probabilities.jsonl` now also carries `feature_row_id`,
  `lineup_snapshot_id`, `features_as_of`, and a conservative
  `projection_generated_at` so each saved ladder row can become a
  contract-valid `PropProjection`
- `build-edge-candidates --date YYYY-MM-DD` joins the latest saved line
  snapshots for that date to the latest saved ladder run containing that date
- `edge_candidates.jsonl` stores raw and calibrated line probabilities, no-vig
  market probabilities, EV, fair odds, capped Kelly sizing, and an evaluation
  status for every replayable line snapshot

AGE-151 turns those pricing seams into the first honest historical evaluation:

- `build-walk-forward-backtest --start-date YYYY-MM-DD --end-date YYYY-MM-DD`
  replays all saved odds runs for each evaluated date and selects the latest
  exact-line snapshot at or before the configured pregame cutoff
- `backtest_bets.jsonl` stores the selected cutoff-safe line snapshot,
  feature-row and lineup refs, honest held-out over or under probabilities,
  final outcome joins, and realized decision result
- `backtest_runs.jsonl` stores window-level ROI, CLV, and edge-bucket summaries
- `join_audit.jsonl` stores cutoff compliance, train-window freshness, and
  outcome traceability for both kept and rejected rows
- headline backtest rows use held-out calibrated probabilities from
  `raw_vs_calibrated_probabilities.jsonl`, not the production calibrator stored
  alongside `ladder_probabilities.jsonl`

## Model Shape

The docs should define the modeling job without pretending the implementation
choice is already settled.

Required modeling behavior:

1. produce an expected strikeout signal for a pitcher-game matchup
2. convert that expectation into an over or under probability for the posted
   line
3. keep calibration explicit and testable out of sample

Current implementation path:

- benchmark baseline:
  `pitcher_k_rate * expected_leash_batters_faced`
- current trainable baseline:
  deterministic ridge-style linear regression implemented in the standard
  library so the repo stays dependency-light during bootstrap
- current count-distribution layer:
  a fitted global negative-binomial dispersion parameter with variance
  `mean + alpha * mean^2`, used to turn the ridge mean into over or under line
  probabilities and adjacent ladder rungs
- later production candidate:
  tree-based regressor or classifier in a dedicated issue that expands
  dependencies deliberately
- calibration:
  explicit post-model calibration on out-of-fold or walk-forward predictions

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

Current AGE-150 historical candidate generation uses
`projection_generated_at = features_as_of` as the conservative default until a
dedicated live inference artifact exists.

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

Current AGE-147 training matrix intentionally excludes:

- IDs and keys such as `official_date`, `gamePk`, `pitcher_id`, and feature-row
  identifiers
- `features_as_of`
- the target column `starter_strikeouts`
- the naive benchmark output itself

### Pricing integrity

- compare the model only to the actual book line snapshot that existed at
  decision time
- use the latest exact-line snapshot that existed at or before the configured
  pregame cutoff; later snapshots should be preserved as rejected rows, not
  substituted into the decision record
- report CLV separately from realized ROI
- keep vig in the pricing path instead of treating the board as fair by default
- keep headline backtest calibration honest by using held-out calibration rows
  rather than the production calibrator fit across the full out-of-fold table
- keep rejected or skipped rows so later threshold or join changes can be
  audited against the same saved market history

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
