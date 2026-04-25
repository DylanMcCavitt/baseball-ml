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

AGE-287 adds the standalone starter-game strikeout dataset build that the
projection rebuild should use before model fitting. It uses existing normalized
feature runs when available; when the canonical data directory has no feature
runs, it falls back to direct regular-season Baseball Savant Statcast pitch-log
chunks. Direct mode infers actual starters from the first pitcher used by each
fielding team, counts same-game strikeout events only for the
`starter_strikeouts` target label, and writes durable coverage artifacts:

```bash
uv run python -m mlb_props_stack build-starter-strikeout-dataset \
  --start-date 2019-03-20 \
  --end-date 2026-04-24 \
  --output-dir /Users/dylanmccavitt/projects/nba-ml/data \
  --chunk-days 3 \
  --max-fetch-workers 4
```

Output path:

`data/normalized/starter_strikeout_training_dataset/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../`

Key artifacts:

- `starter_game_training_dataset.jsonl`
  one row per `(official_date, gamePk, pitcher_id)` starter-game with pitcher,
  team/opponent, home/away, target strikeouts, starter role review fields,
  source references, season/month, pitch-clock era, and league environment
- `coverage_report.json` and `coverage_report.md`
  row counts by season/month/team, source dates without pitch rows, missing
  targets, excluded starts, starter role edge cases, source freshness, chunk cap
  warnings, and timestamp violations
- `schema_drift_report.json`
  field-level non-null coverage for the built artifact
- `missing_targets.jsonl`
  source starter rows excluded because a same-game target could not be derived
- `source_manifest.jsonl`
  one row per direct Statcast chunk with source URL, raw path, pitch-row count,
  dataset-row count, missing-target count, and cap-warning status
- `timestamp_policy.md`
  the issue-local rule that pregame feature references remain features and
  same-game Statcast pulls are target labels only

The landed AGE-287 canonical artifact lives at:

`data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z/`

Coverage summary:

- `31,729` dataset rows
- seasons represented: `2019`, `2020`, `2021`, `2022`, `2023`, `2024`, `2025`, `2026`
- row counts: `4858`, `1796`, `4858`, `4859`, `4860`, `4856`, `4860`, `782`
- `587` source chunks
- `0` source chunks at the 25,000-row cap-warning threshold
- `3` missing-target exclusions
- `0` duplicate source rows
- `0` timestamp violations

AGE-147 still builds the original reproducible starter strikeout training
dataset as a side effect of fitting the frozen v0 baseline on top of those
feature tables.

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
  feature fields, anchored on a dense pitcher/workload core and only adding
  optional numeric fields when the train window actually has enough populated,
  non-constant data to support them
- reports RMSE, MAE, and Spearman rank correlation plus coefficient-based
  feature importance in `evaluation.json`
- writes `evaluation_summary.json` and `evaluation_summary.md` so each run has
  a readable held-out benchmark comparison, calibration snapshot,
  optional-feature activation/exclusion table, top-feature table, and
  previous-run delta on the same date window when available

The training CLI now has an explicit `--feature-set` switch:

- `core`
  uses the dense pitcher/workload schema only and records every optional family
  as excluded by the core feature-set guard
- `expanded`
  uses the dense core plus any optional numeric feature that passes coverage
  and variance gates

`AGE-268` adds the comparison command:

```bash
uv run python -m mlb_props_stack compare-starter-strikeout-baselines \
  --start-date 2026-04-18 \
  --end-date 2026-04-23 \
  --output-dir /Users/dylanmccavitt/projects/nba-ml/data
```

That command trains `core` and `expanded` variants over the same window, runs a
pinned walk-forward backtest for each model run, applies the shared final wager
approval gates to the backtest reporting rows, and writes
`model_comparison.json` plus `model_comparison.md` under
`data/normalized/starter_strikeout_model_comparison/...`.

AGE-286 freezes this current path as `starter-strikeout-baseline-v0`. The
freeze is an audit label, not a production promotion: the ridge baseline,
global dispersion layer, calibrator, artifact layout, and comparison CLI remain
useful infrastructure, but the projection itself is not trusted for live
betting or readiness claims. See `docs/baseline_v0_audit.md` for the retained
AGE-268 evidence, rest-days risk, optional-feature coverage gaps, and
assumptions that must not carry into the rebuild.

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
- the training CLI now surfaces held-out RMSE / MAE for the model and the naive
  benchmark directly, instead of forcing every review to open raw JSON first

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

AGE-151 and AGE-152 turn those pricing seams into the first honest historical
evaluation plus the first dashboard-ready reporting outputs:

- `build-walk-forward-backtest --start-date YYYY-MM-DD --end-date YYYY-MM-DD`
  replays all saved odds runs for each evaluated date and selects the latest
  exact-line snapshot at or before the configured pregame cutoff
- `backtest_bets.jsonl` stores the selected cutoff-safe line snapshot,
  feature-row and lineup refs, honest held-out over or under probabilities,
  final outcome joins, and realized decision result
- `bet_reporting.jsonl` stores a flat per-bet reporting table for Plotly and
  Streamlit, including paper result, CLV status, edge bucket, and
  model-vs-market scatter fields
- `backtest_runs.jsonl` stores window-level metadata plus pointers to the
  generated reporting artifacts
- `join_audit.jsonl` stores cutoff compliance, train-window freshness, and
  outcome traceability for both kept and rejected rows
- `clv_summary.jsonl` stores daily and overall CLV rows so paper winners and
  market-beating bets can be separated directly
- `roi_summary.jsonl` stores daily and overall realized stake, profit, and ROI
  rows
- `edge_bucket_summary.jsonl` stores one realized summary row per configured
  edge bucket
- headline backtest rows use held-out calibrated probabilities from
  `raw_vs_calibrated_probabilities.jsonl`, not the production calibrator stored
  alongside `ladder_probabilities.jsonl`

AGE-153 adds the first target-date inference and paper-tracking layer:

- `generate_starter_strikeout_inference_for_date(target_date=...)`
  loads the latest saved baseline run whose historical coverage still ends
  before the requested slate, then scores one target date from the latest
  AGE-146 feature rows
- `data/normalized/starter_strikeout_inference/date=YYYY-MM-DD/run=.../ladder_probabilities.jsonl`
  stores those target-date raw and calibrated ladder probabilities
- `build-daily-candidates --date YYYY-MM-DD`
  uses that target-date inference output plus the latest saved line snapshots
  to write `daily_candidates.jsonl` and refresh `paper_results.jsonl`
- `daily_candidates.jsonl`
  stores ranked scored props for the slate, including actionable and
  below-threshold rows, plus the final wager-gate decision that separates raw
  actionable edges from approved wagers
- `paper_results.jsonl`
  stores only the final approved paper bets from the latest sheet per date,
  with same-line CLV where available and pending vs settled result status

AGE-154 adds the first explicit experiment-tracking layer:

- `TrackingConfig` now defines a local MLflow store at
  `file:./artifacts/mlruns`
- training runs log into
  `mlb-props-stack-starter-strikeout-training`
- walk-forward backtests log into
  `mlb-props-stack-walk-forward-backtest`
- `evaluation_summary.json` now stores the MLflow run ID, experiment name, and
  the exact rerun command for the saved training slice
- `backtest_runs.jsonl` now stores the associated MLflow run ID for each
  backtest window summary row
- both training and backtest run directories now include
  `reproducibility_notes.md` so the exact CLI inputs can be relaunched later

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
  `starter-strikeout-baseline-v0`, a deterministic ridge-style linear
  regression implemented in the standard library so the repo stays
  dependency-light during bootstrap; this is infrastructure-only and not a
  production or live-use betting model
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

Current AGE-153 target-date inference writes a separate
`inference_generated_at` timestamp for the run itself, while
`projection_generated_at` still stays equal to `features_as_of` so the saved
projection timestamp remains conservative against the quoted line snapshot.

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

The exact go or no-go thresholds now live in `docs/stage_gates.md`.

High-level rule:

- the repo stays `research_only` until held-out model quality, walk-forward
  coverage, paper-tracked sample size, CLV, and ROI all pass together
- positive short-run ROI alone never promotes the system
- next-market expansion requires a stricter second-stage sample than live-use
  discussion on the current pitcher strikeout market
