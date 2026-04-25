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
| Sportsbook prices | The Odds API MLB event and event-odds endpoints for `pitcher_strikeouts`, normalized into replayable `prop_line_snapshots` | sportsbook name, event id, posted line, over odds, under odds, capture timestamp | backfilling missing pregame prices with closing lines or generic market averages |

The first sportsbook adapter is now The Odds API, but the important v1 rule has
not changed: every decision still has to be backed by a real pregame snapshot
with the fields required by `PropLine`.

## Repo Contract Spine

The current codebase already defines the seams that future ingestion and
modeling work must honor.

| Layer | Current module | Current responsibility |
| --- | --- | --- |
| Runtime defaults | `src/mlb_props_stack/config.py` | market name, edge threshold, Kelly fraction, bankroll cap, timezone, devig mode (`per_book` / `tightest_book` / `consensus`) |
| Prop and projection contracts | `src/mlb_props_stack/markets.py` | `PropLine`, `PropProjection`, `EdgeDecision`, `PropSelectionKey`, `ProjectionInputRef` |
| Source adapters | `src/mlb_props_stack/ingest/mlb_stats_api.py`, `src/mlb_props_stack/ingest/odds_api.py`, `src/mlb_props_stack/ingest/statcast_features.py` | fetch schedule, `feed/live`, sportsbook event-odds payloads, and targeted Statcast CSV pulls; preserve raw snapshots; normalize `games`, `probable_starters`, `lineup_snapshots`, `event_game_mappings`, `prop_line_snapshots`, `pitch_level_base`, `pitcher_daily_features`, `lineup_daily_features`, and `game_context_features` |
| Starter-game dataset | `src/mlb_props_stack/starter_dataset.py` | build the standalone one-row-per-starter-game strikeout target dataset plus coverage, missing-target, schema-drift, timestamp-policy, and reproducibility artifacts for the projection rebuild |
| Pitcher skill features | `src/mlb_props_stack/pitcher_skill_features.py` | build prior-only pitcher skill, pitch arsenal, shrinkage, recent-form, and capped rest-bucket features over the starter-game dataset |
| Lineup matchup features | `src/mlb_props_stack/lineup_matchup_features.py` | build prior-only batter-by-batter and aggregate opponent-lineup matchup features over the starter-game dataset |
| Workload and leash features | `src/mlb_props_stack/workload_leash_features.py` | build prior-only expected opportunity, pitch-count, batters-faced, rest-bucket, team-leash, and role-context features over the starter-game dataset |
| Obsolete pre-rebuild modeling | `src/mlb_props_stack/modeling.py` | legacy v0 training path retained only until the rebuild replaces it; do not use `starter-strikeout-baseline-v0` as current performance evidence or as the rebuild benchmark |
| Pricing math | `src/mlb_props_stack/pricing.py` | American odds conversion, per-book and consensus devig, book hold, fair odds, expected value, fractional Kelly, and capped bankroll sizing |
| Decision layer | `src/mlb_props_stack/edge.py` | match line and projection contracts, enforce timestamp order, score no-vig edges, and write replayable `edge_candidates` rows |
| Evaluation guardrails | `src/mlb_props_stack/backtest.py` | cutoff-safe snapshot selection, walk-forward backtest joins, join-audit artifacts, chart-ready reporting tables, and the baseline honesty checklist |
| Tracking and dashboard seams | `src/mlb_props_stack/tracking.py`, `src/mlb_props_stack/dashboard/app.py` | reserved tracking config plus the Strike Ops Streamlit workbench for board review, pitcher drill-down, backtests, registry, feature inspection, and config controls |

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
starter strikeout training dataset + pitcher skill feature layer + expectation baseline
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
5. `build_edge_candidates_for_date()` writes auditable edge rows keyed by line
   snapshot and model version
6. `generate_starter_strikeout_inference_for_date()` scores one target-date
   slate from the latest historical baseline run that ends before that date
7. `build_daily_candidate_workflow()` writes ranked `daily_candidates` plus
   cumulative `paper_results` for recent paper tracking
8. `build_walk_forward_backtest()` selects the latest exact-line snapshot at or
   before the configured cutoff, joins it to held-out probabilities and final
   outcomes, and writes replayable backtest artifacts
9. `BacktestPolicy` and `BACKTEST_CHECKLIST` define which historical runs are
   considered valid

## Current AGE-144, AGE-145, And AGE-146 Output Shape

The first three ingest slices now write both raw and normalized artifacts locally.

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

AGE-145 adds:

- raw Odds API event snapshots:
  `data/raw/the_odds_api/date=YYYY-MM-DD/events/captured_at=...json`
- raw event-odds snapshots:
  `data/raw/the_odds_api/date=YYYY-MM-DD/event_odds/event_id=.../captured_at=...json`
- normalized runs:
  `data/normalized/the_odds_api/date=YYYY-MM-DD/run=.../`

The normalized sportsbook files are:

- `event_game_mappings.jsonl`
  one row per target-date Odds API event candidate with the source `event_id`,
  the derived `odds_matchup_key`, and the mapped MLB `gamePk` when the join
  succeeds
- `prop_line_snapshots.jsonl`
  one row per sportsbook, pitcher, line, and capture time with paired
  over/under odds, `market_last_update`, the source `event_id`, and the mapped
  `gamePk` when available

AGE-146 adds:

- raw Statcast CSV pulls:
  `data/raw/statcast_search/date=YYYY-MM-DD/player_type=.../player_id=.../captured_at=...csv`
- normalized runs:
  `data/normalized/statcast_search/date=YYYY-MM-DD/run=.../`

The normalized Statcast feature files are:

- `pull_manifest.jsonl`
  one row per targeted pitcher or batter pull with the exact CSV URL, history
  window, raw path, and row count
- `pitch_level_base.jsonl`
  one normalized pitch row per unique `(gamePk, at_bat_number, pitch_number,
  pitcher, batter)` across all raw pulls, with chase/contact flags preserved for
  feature tracing
- `pitcher_daily_features.jsonl`
  one row per probable starter with rolling strikeout, whiff, CSW, pitch mix,
  velocity, extension, workload, and rest features
- `lineup_daily_features.jsonl`
  one row per probable starter keyed to the opponent lineup snapshot when a
  pregame-valid lineup exists
- `game_context_features.jsonl`
  one row per probable starter with venue, rest, expected leash proxies, and
  explicit missing-value markers for unsourced park-factor and weather fields

AGE-287 adds the standalone projection-rebuild dataset artifact:

- `data/normalized/starter_strikeout_training_dataset/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../starter_game_training_dataset.jsonl`
  one row per `(official_date, gamePk, pitcher_id)` starter-game with target
  strikeouts, starter role review fields, source references, season/month,
  pitch-clock era, and league environment fields. The builder uses feature-run
  references when available and otherwise falls back to direct regular-season
  Statcast pitch-log chunks.
- `coverage_report.json` and `coverage_report.md`
  row counts by season/month/team, source dates without pitch rows, missing
  targets, excluded starts, short-start role edge cases, source freshness, cap
  warnings, and timestamp-policy status
- `missing_targets.jsonl`
  source starter rows excluded because same-game Statcast outcomes could not
  derive the target label
- `source_manifest.jsonl`
  direct Statcast chunk source URLs, raw paths, pitch-row counts, dataset-row
  counts, missing-target counts, and cap-warning status
- `schema_drift_report.json`
  field-level non-null coverage for the built dataset
- `timestamp_policy.md`
  explicit rule that outcome pulls supply labels only and must not feed
  pregame features
- `reproducibility_notes.md`
  deterministic rerun command for the requested date window

AGE-288 adds the first projection-rebuild feature layer on top of the
starter-game artifact:

- `data/normalized/pitcher_skill_features/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../pitcher_skill_features.jsonl`
  one row per starter-game with only prior-game pitcher skill and arsenal
  fields: K%, K-BB%, walk, strike, CSW, SwStr, whiff, called-strike, putaway,
  pitch-type usage, pitch-type whiff/CSW, velocity, spin, movement, release
  extension, recent-form windows, and shrinkage-context fields
- `feature_report.json` and `feature_report.md`
  coverage, missingness, variance, top correlations by season, leakage-policy
  status, and rest-policy status
- `reproducibility_notes.md`
  exact rerun command and the rule that same-game target rows are not feature
  inputs

Rest context in this layer is capped and bucketed. Raw continuous `rest_days`
is not exposed as an unbounded primary driver; long layoffs, no prior starts,
short rest, standard rest, and extra rest are explicit flags.

AGE-289 adds the opponent-lineup matchup layer on top of the same starter-game
artifact:

- `data/normalized/lineup_matchup_features/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../lineup_matchup_features.jsonl`
  one row per starter-game with explicit confirmed/projected/missing lineup
  status, batter-history coverage, handedness-weighted lineup K vulnerability,
  contact, chase, whiff, CSW, and pitcher-arsenal-weighted pitch-type weakness
- `batter_matchup_features.jsonl`
  one row per batting-order slot with prior-only batter K%, K/PA, handedness
  splits, contact/chase/whiff/CSW rates, pitch-type weakness, and
  sample-size-regressed K context
- `feature_report.json` and `feature_report.md`
  missingness, coverage, variance, season correlations, leakage status, and
  artifact paths
- `reproducibility_notes.md`
  exact rerun command and the rule that same-game batting orders are not
  pregame feature inputs

Confirmed lineup features remain separate from projected lineup features. When
the rebuild dataset does not carry a pregame lineup snapshot, the fallback is
the opponent team's most recent prior-game batting order, labeled
`projected_from_prior_team_game`. Rows with no projection or incomplete batter
history keep explicit status fields instead of zero-filled numeric features.

AGE-290 adds the expected-opportunity workload and leash layer on top of the
same starter-game artifact:

- `data/normalized/workload_leash_features/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../workload_leash_features.jsonl`
  one row per starter-game with prior-only recent pitch counts, recent batters
  faced, season workload distributions, team starter leash tendency, expected
  pitch count, expected batters faced, times-through-order threshold rates,
  capped rest buckets, and source-backed opener/bulk role context
- `feature_report.json` and `feature_report.md`
  coverage, variance, top correlations by season, rest policy, role-context
  source counts, leakage status, and artifact paths
- `reproducibility_notes.md`
  exact rerun command and the rule that same-game target outcomes are not
  workload inputs

This layer is intentionally separate from pitcher skill and lineup matchup
features. It describes how many opportunities a starter is expected to receive,
not how often the starter creates strikeouts per opportunity. Raw continuous
`rest_days` is not emitted; long layoffs are separate from standard rest and
stay labeled as unknown layoff context unless a timestamp-valid source
explicitly labels IL, rehab, or role-change state.

AGE-147 adds:

- raw same-day outcome pulls:
  `data/raw/statcast_search_outcomes/date=YYYY-MM-DD/player_id=.../captured_at=...csv`
- normalized baseline training runs:
  `data/normalized/starter_strikeout_baseline/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../`

The normalized model-training files are:

- `training_dataset.jsonl`
  one joined starter-game row keyed by `official_date`, `gamePk`, and
  `pitcher_id`, with explicit feature-row and lineup references preserved
- `starter_outcomes.jsonl`
  one observed same-game strikeout total per starter derived from same-day
  Statcast rows
- `date_splits.json`
  saved train, validation, and test dates
- `baseline_model.json`
  serialized ridge-style linear baseline coefficients plus the encoded feature
  schema used at train time
- `evaluation.json`
  RMSE, MAE, and Spearman rank correlation for both the naive benchmark and the
  trainable baseline, plus coefficient-based feature importance

AGE-148 extends those same baseline runs with:

- a fitted global negative-binomial dispersion parameter stored inside
  `baseline_model.json`
- `ladder_probabilities.jsonl`
  one row per starter-game with the predicted mean and half-strikeout ladder
  over or under probabilities
- held-out count-distribution metrics in `evaluation.json`
  so the fitted distribution can be compared against a Poisson fallback using
  the same mean predictions

AGE-149 adds a calibration layer on top of those ladder probabilities:

- `probability_calibrator.json`
  a stored isotonic calibrator fit from out-of-fold ladder-event probabilities
- `raw_vs_calibrated_probabilities.jsonl`
  honest held-out ladder-event rows with raw and calibrated probabilities side
  by side
- `calibration_summary.json`
  reliability bins plus Brier, log-loss, and calibration-error diagnostics that
  later tracking and dashboard code can log directly
- `ladder_probabilities.jsonl`
  now preserves both the raw ladder and a calibrated ladder so downstream
  pricing work can consume calibrated probabilities without recomputing the
  calibration layer

AGE-150 adds the first saved decision artifact on top of those calibrated
probabilities:

- `ladder_probabilities.jsonl`
  now also carries `feature_row_id`, `lineup_snapshot_id`, `features_as_of`,
  and a conservative `projection_generated_at` so downstream pricing code can
  materialize contract-valid `PropProjection` objects
- `data/normalized/edge_candidates/date=YYYY-MM-DD/run=.../edge_candidates.jsonl`
  stores one auditable pricing row per line snapshot and model version,
  including actionable candidates, below-threshold rows, and skipped rows that
  failed join or timestamp requirements

AGE-151 and AGE-152 add the first saved historical evaluation artifact and the
first dashboard-ready reporting artifacts on top of those decision seams:

- `data/normalized/walk_forward_backtest/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../backtest_bets.jsonl`
  stores the selected cutoff-safe line snapshot, model refs, honest held-out
  probabilities, final outcome, and realized decision result for each exact
  line group, or an explicit skipped-by-reason status when the line cannot be
  joined honestly
- `data/normalized/walk_forward_backtest/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../bet_reporting.jsonl`
  stores a flat per-bet table with paper-result, CLV, edge-bucket, and
  model-vs-market scatter fields broken out for downstream dashboards
- `data/normalized/walk_forward_backtest/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../backtest_runs.jsonl`
  stores the run-level summary for one requested historical window, including
  the associated MLflow run ID, rerun metadata, and `skip_reason_counts` so a
  zero-bet window still explains why rows were skipped
- `data/normalized/walk_forward_backtest/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../join_audit.jsonl`
  stores the freshness and cutoff audit for every kept or rejected backtest row,
  including explicit join-failure statuses such as `unmatched_event_mapping`,
  `missing_line_probability`, and `missing_lineup_snapshot_id`
- `data/normalized/walk_forward_backtest/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../clv_summary.jsonl`
  stores daily and overall CLV summaries so paper winners and market-beating
  bets can be separated explicitly
- `data/normalized/walk_forward_backtest/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../roi_summary.jsonl`
  stores daily and overall realized stake, profit, and ROI rows
- `data/normalized/walk_forward_backtest/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../edge_bucket_summary.jsonl`
  stores one realized summary row per configured edge bucket

AGE-153 adds the first target-date operating loop on top of those same seams:

- `data/normalized/starter_strikeout_inference/date=YYYY-MM-DD/run=.../`
  stores target-date ladder probabilities scored from the latest historical
  baseline run whose end date is still before the requested slate
- `data/normalized/daily_candidates/date=YYYY-MM-DD/run=.../daily_candidates.jsonl`
  stores the ranked current sheet, including actionable and below-threshold
  scored props for that slate, plus the final shared wager-gate decision used
  by both paper tracking and the dashboard board
- `data/normalized/paper_results/date=YYYY-MM-DD/run=.../paper_results.jsonl`
  stores only final approved paper bets from the latest daily sheet per date,
  with pending vs settled status and same-line CLV where the exact close exists

The Streamlit dashboard now reads those AGE-153 artifacts directly:

- `daily_candidates`
  powers the current-slate table and slate-level counts
- `paper_results`
  powers recent paper performance and per-bet result review
- `evaluation_summary.json` and `calibration_summary.json`
  power the backtest KPI strip, reliability plot, and feature inspection
- `roi_summary.jsonl`, `clv_summary.jsonl`, and `bet_reporting.jsonl`
  power the walk-forward charts and model-vs-market scatter when those artifacts
  exist
- the local MLflow tracking store
  powers the registry screen and run-diff actions

The first backtest slice intentionally uses held-out probabilities from
`raw_vs_calibrated_probabilities.jsonl` rather than the production calibrator
embedded in `ladder_probabilities.jsonl`, so reported CLV and ROI stay aligned
to walk-forward evaluation instead of leaking future calibration data.

The lineup guardrail in AGE-146 is intentionally strict:

- only lineup snapshots with `captured_at <= commence_time` are allowed into the
  pregame feature tables
- if the latest lineup snapshot is late, the feature row keeps a missing-lineup
  status instead of silently leaking a post-lock lineup

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

Current AGE-150 historical edge builds use `features_as_of` as the conservative
`generated_at` timestamp until a dedicated pregame inference runner persists its
own projection snapshot times.

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

- Promotion status is controlled by the numeric stage gates in
  `docs/stage_gates.md`, not by one good-looking run.
- No output is trustworthy unless the upstream line snapshot is real and
  timestamped.
- CLV and realized ROI answer different questions and must never be collapsed
  into one metric.
- If a prop cannot be tied back to a specific lineup snapshot and feature row,
  it should not be promoted to live tracking.
- A model that looks profitable only after substituting later lineups, later
  prices, or closing lines is invalid, even if the headline ROI is positive.
