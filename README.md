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
  Odds conversion, expected value, per-book and consensus devig, book
  hold, and Kelly sizing.
- `src/mlb_props_stack/markets.py`
  Core data models for props, projections, and decisions.
- `src/mlb_props_stack/ingest/mlb_stats_api.py`
  MLB Stats API adapters for schedule, probable starters, and lineup snapshots.
- `src/mlb_props_stack/ingest/statcast_features.py`
  Targeted Statcast pulls plus normalized pitcher, lineup, and game-context
  feature tables for one slate date.
- `src/mlb_props_stack/modeling.py`
  Date-split starter strikeout dataset assembly, naive benchmark, and the first
  reproducible trainable baseline model.
- `src/mlb_props_stack/edge.py`
  Edge detection and candidate ranking.
- `src/mlb_props_stack/paper_tracking.py`
  Target-date inference, daily candidate sheet generation, and cumulative paper
  result refreshes.
- `src/mlb_props_stack/backtest.py`
  Backtest policy and evaluation guardrails.
- `src/mlb_props_stack/dashboard/app.py`
  Multi-screen Streamlit workbench for the live board, pitcher detail,
  backtests, MLflow registry review, feature inspection, and config tuning.
- `docs/architecture.md`
  Product and system architecture.
- `docs/modeling.md`
  Data, features, targets, and validation rules.
- `docs/stage_gates.md`
  Numeric go or no-go thresholds for live usage discussion and next-market
  expansion.

## Recommended Build Order

1. data contracts for games, pitchers, lines, and line moves
2. feature pipeline from Statcast + schedule + lineups
3. starter strikeout expectation baseline model
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

If `ODDS_API_KEY` is not already exported in the shell, the CLI will also try to
load a repo-local `.env` without overriding existing environment variables. In a
git worktree, it first checks the current worktree root and then falls back to a
sibling checkout from the same repo, which lets one ignored local `.env` in the
canonical checkout keep working across fresh issue worktrees.

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

Multiple books that quote the same pitcher-line pair land as separate rows
(one row per `(sportsbook, captured_at)`) so the edge builder can devig
them independently. To narrow ingest to a specific sharp subset:

```bash
ODDS_API_KEY=YOUR_KEY uv run python -m mlb_props_stack \
  ingest-odds-api-lines --date 2026-04-21 --bookmakers pinnacle,circa
```

`StackConfig.devig_mode` selects how those rows are priced downstream
(`per_book` is the default, `tightest_book` picks the lowest-hold book at
each line, and `consensus` averages the no-vig probabilities across every
book). The resolved books land on each scored candidate row in
`market_consensus_books` so a historical replay can always audit which
books drove the devig.

Operational note:

- `event_game_mappings.jsonl` still records unmatched same-team events for audit
  and debugging
- `prop_line_snapshots.jsonl` now only persists line rows from matched target-date
  events, so downstream scoring does not inherit known-unmatched event IDs
- the CLI ingest summary now reports how many unmatched events were skipped and how
  many matched events returned no pitcher-strikeout markets yet

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
- park strikeout factor is resolved via a static lookup checked into the
  repo at `data/static/park_factors/park_k_factors.csv`; the feature build
  joins by MLB `venue_id` and emits `park_k_factor`,
  `park_k_factor_vs_rhh`, and `park_k_factor_vs_lhh` with
  `park_factor_status = "ok"`, only preserving the
  `missing_park_factor_source` status when the venue id is unknown
  (extend the CSV rather than papering over the miss at read time)

Statcast CSV pulls retry transient failures (HTTP 429/5xx, timeouts, and
connection errors) with bounded exponential backoff and fan out across a
small thread pool (default four workers) so a full slate fetches in parallel
while still writing raw and normalized artifacts in deterministic pull order.

Normalized JSONL artifacts from the MLB metadata, Odds API, and Statcast
feature ingests are written atomically: each `_write_jsonl` call streams to a
sibling `.tmp` file in the same run directory and then `os.replace`s it over
the target path. Readers therefore only ever observe a fully written JSONL or
the prior version, and a crash mid-write leaves the previous artifact intact
rather than producing a truncated normalized file.

## Historical Backfill

`AGE-201` adds the `backfill-historical` CLI subcommand so the three ingest
adapters above can be replayed across an arbitrary date window in one
invocation. The default sweep walks `ingest-mlb-metadata`,
`ingest-odds-api-lines`, and `ingest-statcast-features` for every calendar
date in `[--start-date, --end-date]`, and each per-date source is treated
independently — a transient odds-history gap or a single bad Statcast pull
no longer aborts the rest of the run.

```bash
ODDS_API_KEY=YOUR_KEY uv run python -m mlb_props_stack backfill-historical \
  --start-date 2024-03-28 \
  --end-date 2024-09-29
```

By default, the sweep is **idempotent**: for each date and source the helper
checks whether the latest normalized run directory already contains every
required artifact (`games.jsonl` plus `probable_starters.jsonl` and
`lineup_snapshots.jsonl` for the MLB metadata source, `event_game_mappings.jsonl`
plus `prop_line_snapshots.jsonl` for the odds source, and the five Statcast
feature tables for the Statcast source). When everything is on disk the
source is recorded as `skipped_resume` and no API call is made; pass
`--force` to re-ingest dates that are already complete. Use `--sources` to
restrict the sweep to a subset (e.g. `--sources mlb-metadata,statcast-features`
to skip the Odds API entirely while testing).

Each sweep also writes a manifest under
`data/normalized/backfill/run=<RUN_ID>/backfill_manifest.json` capturing the
per-date outcome (`ingested`, `skipped_resume`, or `failed`), the resulting
ingest `run_id`, and any error type and message recorded for failed sources.
The CLI exits non-zero whenever at least one source recorded `failed` so
overnight runs can be supervised by simple shell wrappers.

Operational notes for full-season runs:

- **Runtime.** Each MLB metadata pull costs ~1 schedule call plus one
  `feed/live` call per scheduled game (~15 games/day in regular season).
  The Statcast feature build issues one CSV pull per starter and per
  opposing-lineup batter (default `--history-days 30`), with the bounded
  retry/backoff and four-worker thread pool from AGE-199 controlling fan-out.
  In practice a single full regular season (~180 dates) takes several hours
  end to end and is best run overnight on a stable connection. The
  resume logic means an interrupted run can be re-invoked with the exact
  same arguments and only the missing dates will be ingested.
- **Disk footprint.** A full regular season produces tens of gigabytes of
  raw Statcast CSVs and hundreds of megabytes of normalized JSONL. The
  raw `data/` tree is intentionally `.gitignore`d; commit the normalized
  outputs (or, for the AGE-201 acceptance run, ship the raw artifacts as
  a release tarball or via git-lfs rather than as plain git history).
- **Odds-history limitation.** The Odds API only exposes live and
  near-future markets on the standard plan, so historical pitcher
  strikeout lines for 2024 and 2025 will frequently come back empty or
  result in `failed` outcomes. The `--sources` flag and best-effort
  failure handling let the MLB-metadata and Statcast-feature backfills
  succeed even when the odds source has nothing useful to return for a
  given date — this is expected behavior, not a regression.

## Starter Strikeout Baseline Training

`AGE-147` adds the first reproducible training loop for expected starter
strikeouts.

This command expects AGE-146 feature runs to already exist under
`data/normalized/statcast_search/...` for the requested date span. It joins
`pitcher_daily_features`, `lineup_daily_features`, and
`game_context_features`, then pulls same-day pitcher Statcast rows to derive the
official starter strikeout target for each feature row.

Train the baseline model on one date range:

```bash
uv run python -m mlb_props_stack train-starter-strikeout-baseline \
  --start-date 2026-04-01 \
  --end-date 2026-04-20 \
  --feature-set expanded
```

By default that writes:

- raw same-day outcome pulls under
  `data/raw/statcast_search_outcomes/date=YYYY-MM-DD/player_id=.../`
- normalized baseline outputs under
  `data/normalized/starter_strikeout_baseline/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../`

The normalized outputs include:

- `training_dataset.jsonl`
  one joined training row per `official_date`, `game_pk`, and starter
- `starter_outcomes.jsonl`
  the exact same-game strikeout labels derived from Statcast rows
- `date_splits.json`
  train, validation, and test dates saved explicitly instead of random row
  splits
- `baseline_model.json`
  the serialized ridge-style linear baseline model, feature schema, and fitted
  global count-distribution dispersion parameter plus the stored probability
  calibrator metadata
- `evaluation.json`
  RMSE, MAE, and Spearman rank correlation for both the naive benchmark and the
  trainable baseline, plus coefficient-based feature importance, held-out
  count-distribution metrics, and honest raw-vs-calibrated probability
  diagnostics
- `ladder_probabilities.jsonl`
  one row per starter-game with the predicted mean, fitted negative-binomial
  dispersion, raw half-strikeout ladder probabilities, calibrated ladder
  probabilities derived from the stored calibrator, and the feature / lineup
  references needed to materialize replayable projections later
- `probability_calibrator.json`
  the production calibrator artifact fit from out-of-fold ladder probabilities
- `raw_vs_calibrated_probabilities.jsonl`
  honest held-out probability rows with raw and calibrated over/under values
- `calibration_summary.json`
  reliability bins and probability diagnostics mirrored into the tracked
  training run
- `evaluation_summary.json`
  a compact machine-readable offline report with held-out benchmark-vs-model
  metrics, feature-set and optional-feature selection diagnostics, top feature
  importance, MLflow run metadata, and same-window previous-run deltas
- `evaluation_summary.md`
  a human-readable markdown version of the same offline report for quick local
  review after each training run
- `reproducibility_notes.md`
  the exact rerun command, MLflow experiment name, MLflow run ID, and local run
  directory for the saved training slice

The training CLI summary also now prints the held-out RMSE and MAE for both the
benchmark and the model, plus the MLflow run metadata, reproducibility-note
path, and previous run ID when the same date window has already been trained
before.

The current baseline fit is intentionally conservative on short windows:

- it always trains on a dense core of pitcher and workload features
- it only adds lineup-derived numeric fields when those columns are populated
  and variable across the training dates
- it drops categorical dummy fields from the early baseline so sparse,
  short-window runs do not overfit to team-side splits alone

Use `--feature-set core` when a run must pin the dense pitcher/workload schema
and exclude every optional family. Use `--feature-set expanded` when a run
should admit optional fields that pass the configured coverage and variance
gates; sparse or constant optional fields remain explicit exclusions in the
saved feature schema instead of being silently ignored.

## Starter Strikeout Model Variant Comparison

`AGE-268` adds a reproducible same-window comparison command for the current
core-only baseline versus the expanded optional-feature candidate.

```bash
uv run python -m mlb_props_stack compare-starter-strikeout-baselines \
  --start-date 2026-04-18 \
  --end-date 2026-04-23 \
  --output-dir /Users/dylanmccavitt/projects/nba-ml/data
```

The command trains two model variants over the same feature/outcome window,
then runs pinned walk-forward backtests for each saved model run:

- `core`
  dense pitcher/workload fields only
- `expanded`
  dense core plus optional features that pass coverage and variance gates

By default it writes:

- `model_comparison.json`
  machine-readable metrics, optional-feature activation/exclusion diagnostics,
  backtest row counts, CLV, ROI, edge-bucket rollups, final-gate approved wager
  counts, timestamp-audit status, and the promotion recommendation
- `model_comparison.md`
  a readable report for dashboard/readiness follow-up
- `reproducibility_notes.md`
  the exact core/expanded training commands and pinned backtest commands

The recommendation stays `keep_core_only` unless the expanded candidate has
active optional features and improves held-out error without worsening
calibration, decision coverage, final-gate approval counts, CLV/ROI metrics, or
timestamp-safety audits.

## Edge Candidate Build

`AGE-150` turns saved calibrated ladder probabilities plus real line snapshots
into replayable pricing decisions.

This command expects:

- a prior AGE-145 odds ingest for the target date under
  `data/normalized/the_odds_api/...`
- a prior AGE-149 baseline run whose `ladder_probabilities.jsonl` contains the
  requested official date under
  `data/normalized/starter_strikeout_baseline/...`

Build edge candidates for one official date:

```bash
uv run python -m mlb_props_stack build-edge-candidates --date 2026-04-20
```

By default that writes:

- normalized edge-candidate outputs under
  `data/normalized/edge_candidates/date=YYYY-MM-DD/run=.../`

The normalized `edge_candidates.jsonl` rows are keyed by:

- `line_snapshot_id`
- `model_version`

Each row preserves:

- the matched line snapshot contract and pricing inputs
- the projection input refs and conservative timestamp ordering
- raw and calibrated model probabilities for the exact posted line
- no-vig market probabilities, EV, fair odds, and capped Kelly sizing
- an evaluation status so below-threshold, skipped, or training-split rows stay
  auditable

## Walk-Forward Backtest

`AGE-151` and `AGE-152` add the first timestamp-safe historical backtest slice
plus the first chart-ready reporting tables.

This command expects:

- prior AGE-145 odds ingests for every evaluated date under
  `data/normalized/the_odds_api/...`, because it replays all saved runs for a
  date to find the latest exact-line snapshot at or before the configured
  cutoff
- a prior starter baseline run whose `training_dataset.jsonl`,
  `raw_vs_calibrated_probabilities.jsonl`, and `starter_outcomes.jsonl` cover
  the requested date window under `data/normalized/starter_strikeout_baseline/...`

Build a walk-forward backtest for one historical window:

```bash
uv run python -m mlb_props_stack build-walk-forward-backtest \
  --start-date 2026-04-19 \
  --end-date 2026-04-20
```

By default that writes:

- normalized backtest outputs under
  `data/normalized/walk_forward_backtest/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../`

The normalized outputs are:

- `backtest_bets.jsonl`
  one row per exact sportsbook line group, including actionable bets,
  below-threshold passes, and skipped rows such as late-only snapshots or
  training-split projections
- `bet_reporting.jsonl`
  a flat per-bet table for Plotly or Streamlit, with paper result, CLV status,
  edge bucket, and model-vs-market scatter fields already broken out
- `backtest_runs.jsonl`
  one summary row for the requested window with placed-bet counts, ROI, CLV,
  edge-bucket rollups, and the associated MLflow run ID
- `join_audit.jsonl`
  one audit row per backtest entry showing cutoff compliance, feature and lineup
  refs, train-window freshness, and outcome traceability
- `clv_summary.jsonl`
  daily and overall CLV rows so the stack can separate paper winners from bets
  that actually beat the market near close
- `roi_summary.jsonl`
  daily and overall realized PnL rows, including cumulative stake, profit, and
  ROI
- `edge_bucket_summary.jsonl`
  one row per configured edge bucket with realized PnL and CLV hit counts
- `reproducibility_notes.md`
  the exact rerun command, pinned source model run directory, and MLflow
  metadata for the backtest slice

The current backtest runner stays honest in two important ways:

- it uses the latest exact-line snapshot at or before the configured pregame
  cutoff, never a later line
- it uses held-out probabilities from
  `raw_vs_calibrated_probabilities.jsonl` for headline backtest rows instead of
  the production calibrator embedded in `ladder_probabilities.jsonl`

## Daily Candidate Workflow

`AGE-153` adds the first repeatable paper-tracking loop for the live slate.

This command expects:

- a prior AGE-146 feature build for the target date under
  `data/normalized/statcast_search/...`
- a prior AGE-145 odds ingest for the target date under
  `data/normalized/the_odds_api/...`
- a prior historical starter baseline run that ends before the target date
  under `data/normalized/starter_strikeout_baseline/...`

Build one target-date sheet and refresh cumulative paper results:

```bash
uv run python -m mlb_props_stack build-daily-candidates --date 2026-04-22
```

If `--date` is omitted, the command defaults to today in the configured stack
timezone.

By default that writes:

- `data/normalized/starter_strikeout_inference/date=YYYY-MM-DD/run=.../`
  with target-date ladder probabilities scored from the latest non-leaky saved
  baseline run
- `data/normalized/daily_candidates/date=YYYY-MM-DD/run=.../daily_candidates.jsonl`
  with the ranked current-slate sheet
- `data/normalized/paper_results/date=YYYY-MM-DD/run=.../paper_results.jsonl`
  with the latest actionable sheet per date resolved to pending or settled paper
  outcomes

`daily_candidates.jsonl` keeps:

- ranked scored props for the selected slate
- the matched line snapshot, selected side, edge, EV, and suggested stake
- the inference and edge run IDs used to produce the sheet
- the final wager-gate status that separates raw actionable edges from
  approved wagers

Print the terminal-first approved wager card from the latest saved daily sheet:

```bash
uv run python -m mlb_props_stack build-wager-card --date 2026-04-22
```

By default the card prints only rows with `wager_approved=true` and writes:

- `data/normalized/wager_card/date=YYYY-MM-DD/run=.../wager_card.jsonl`
  with the exact terminal card rows
- `data/normalized/wager_card/date=YYYY-MM-DD/run=.../wager_card_metadata.json`
  with the source daily-candidate run and approved/blocked counts

Add `--include-rejected` to print blocked candidates in a separate diagnostic
section without mixing them into the approved wager list.

`paper_results.jsonl` keeps:

- only approved paper bets from the latest sheet for each date
- same-line CLV where an exact close snapshot exists
- pending vs settled result status with realized PnL once outcomes are available

Launch the local dashboard with either entrypoint:

```bash
uv run streamlit run src/mlb_props_stack/dashboard/app.py
uv run streamlit run app.py
```

The dashboard now reads:

- `daily_candidates` or `edge_candidates` for the live board
- the latest `walk_forward_backtest` reporting rows as a historical replay board
  when live-slate artifacts are absent
- `paper_results` for recent paper performance
- `evaluation_summary.json` and `calibration_summary.json` for model quality and
  feature inspection
- walk-forward summary artifacts such as `roi_summary.jsonl` and
  `clv_summary.jsonl` for the backtest screen
- the local MLflow store under `file:./artifacts/mlruns` for the registry view

For model refinement on historical data, the intended loop is:

1. train a baseline on a historical window
2. run `build-walk-forward-backtest` on saved historical odds snapshots
3. open the dashboard and inspect replayed historical dates from the board date
   picker

That avoids re-calling The Odds API every iteration once the historical odds
artifacts already exist locally.

## Data Alignment Diagnostic

`AGE-198` adds a fast per-date coverage report across ingest, feature, and
modeling artifacts so the root cause of all-skipped backtest windows is obvious
before anyone spelunks into individual JSONL files.

Check coverage for a historical window:

```bash
uv run python -m mlb_props_stack check-data-alignment \
  --start-date 2026-04-18 \
  --end-date 2026-04-23
```

The command exits non-zero when any date falls below the `--threshold`
coverage ratio (default `0.5`), so it can gate `build-walk-forward-backtest`
in future automation. It reports per-date row counts for `games.jsonl`,
`probable_starters.jsonl`, `lineup_snapshots.jsonl`, `prop_line_snapshots.jsonl`,
the Statcast feature tables, and the latest baseline run's
`training_dataset.jsonl`, `raw_vs_calibrated_probabilities.jsonl`, and
`starter_outcomes.jsonl`, plus the derived `feature_coverage`,
`outcome_coverage`, and `odds_coverage` ratios.

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

Those CI checks are only the baseline repo gate. They do not prove that
Streamlit startup, live CLI entrypoints, or generated artifacts work end to
end. For PR review and handoff on runtime-facing changes, also run the affected
checks in [`docs/review_runtime_checks.md`](docs/review_runtime_checks.md).

## Future Hooks

- `mlb_props_stack.tracking.TrackingConfig` now owns the local MLflow store plus
  the separate training and backtest experiment names.
- `mlb_props_stack.dashboard.app` now hosts the local Strike Ops Streamlit
  workbench for board review, pitcher drill-down, backtests, registry, feature
  inspection, and config controls.
- Training and walk-forward backtest runs now log params, metrics, and local
  artifacts into `file:./artifacts/mlruns`.

## Risk

This project should be treated as a research and decision-support system.
Sportsbooks price efficiently enough that sloppy data handling, leakage, bad
calibration, and untracked line movement can erase a paper edge very quickly.
