# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Last completed issue: `AGE-204` on branch
  `dylan/magical-ptolemy-1d5da2`
- This slice adds a pregame **home-plate umpire** ingest to the pipeline
  so downstream modeling can consume per-game umpire identity plus two
  rolling 30-day strike-zone signals — `ump_called_strike_rate_30d` and
  `ump_k_per_9_delta_vs_league_30d`. The issue suggested
  UmpScorecards/Retrosheet scraping, but the MLB Stats API `feed/live`
  endpoint already returns `liveData.boxscore.officials` including the
  home-plate umpire's id and full name, and those payloads are already
  persisted by the MLB metadata ingest. The umpire adapter mines the
  existing raw feed/live files (falling back to a fresh HTTP fetch
  only when the payload is missing) so the new source takes no
  additional scraping dependency
- Current status: umpire ingest runs standalone
  (`uv run python -m mlb_props_stack.cli ingest-umpire --date …`),
  inside the historical backfill orchestrator as a new `umpire`
  source slot between `weather` and `odds-api` (runs after
  `mlb-metadata` so `games.jsonl` + persisted feed/live exist, runs
  before `statcast-features` so umpire snapshots join
  `game_context_features.jsonl`), and the statcast feature ingest
  joins the latest umpire snapshots into every emitted
  `game_context_features.jsonl` row. `check-data-alignment` surfaces
  umpire coverage in a new `ump_cov` column plus a raw count column
  (`umpire_ok`). Training rows expose seven new fields
  (`umpire_status`, `umpire_source`, `umpire_id`, `umpire_name`,
  `umpire_captured_at`, `ump_called_strike_rate_30d`,
  `ump_k_per_9_delta_vs_league_30d`); the two numerics are gated by
  `OPTIONAL_FEATURE_MIN_COVERAGE` so sparse umpire history will not
  silently regress the model

## What Was Completed In This Slice

- `src/mlb_props_stack/ingest/umpire.py` (new module)
  - `UmpireAssignmentRecord` frozen dataclass holds the raw-ish
    home-plate umpire assignment per scheduled game; `UmpireSnapshotRecord`
    is the normalized snapshot joined with the 30-day rolling metrics
  - `normalize_feed_live_officials_payload` extracts the
    `liveData.boxscore.officials` array from a feed/live payload;
    `_extract_home_plate_umpire` tolerates casing variations and
    returns `(None, None)` when the assignment has not been published
  - `_latest_persisted_feed_live_path` finds the most recent
    persisted feed/live JSON for one game so the adapter can avoid a
    fresh HTTP call; `ingest_umpire_for_date` falls back to
    `MLBStatsAPIClient.fetch_json` when no persisted file exists and
    emits a `missing_umpire_source` sentinel when the fetch raises
  - `_load_prior_umpire_game_pks_by_umpire` walks the prior 30
    calendar days of normalized umpire runs to build
    `umpire_id -> {date -> {game_pks}}`; `_load_prior_pitch_aggregates`
    walks the matching Statcast `pitch_level_base.jsonl` runs to
    tally per-date and per-game-pk pitch/called-strike/PA/strikeout
    counts; `compute_rolling_umpire_metrics` combines the two into
    `(ump_called_strike_rate_30d, ump_k_per_9_delta_vs_league_30d)`
  - K/9 is approximated as `K_rate × 38.25` (9 innings × ~4.25 PAs
    per inning) documented inline on `APPROXIMATE_PA_PER_NINE_INNINGS`
    — the pitch-level base captures plate appearances via the
    final-pitch marker, not innings directly, and the multiplier is
    stable enough for a delta-vs-league feature
  - `ingest_umpire_for_date` writes raw artifacts at
    `data/raw/umpire/date=<iso>/game_pk=<pk>/captured_at=<ts>.json`
    (extracted officials block plus metadata) and normalized snapshots
    at `data/normalized/umpire/date=<iso>/run=<ts>/umpire_snapshots.jsonl`
  - `load_latest_umpire_snapshots_for_date` returns a
    `dict[int, UmpireSnapshotRecord]` keyed on `game_pk` for the
    latest complete run, used by the statcast feature ingest
- `src/mlb_props_stack/ingest/statcast_features.py`
  - `GameContextFeatureRow` gains `umpire_status`, `umpire_source`,
    `umpire_id`, `umpire_name`, `umpire_captured_at`,
    `ump_called_strike_rate_30d`, and
    `ump_k_per_9_delta_vs_league_30d`
  - `_build_game_context_feature_row` now joins via an
    `umpire_lookup: dict[int, UmpireSnapshotRecord]` passed in from
    the top-level ingest call; games without a snapshot land with the
    `missing_umpire_source` sentinel
- `src/mlb_props_stack/ingest/__init__.py`
  - re-exports the umpire dataclasses, constants, and both entry
    points alongside the existing ingest surface
- `src/mlb_props_stack/cli.py`
  - new `ingest-umpire` subcommand (`--date`, `--output-dir`)
  - `render_umpire_ingest_summary` renders run id, snapshot counts,
    history window, and raw + normalized output paths
  - `backfill-historical` help copy now lists the umpire source
- `src/mlb_props_stack/backfill.py`
  - new `SOURCE_UMPIRE = "umpire"` placed between `SOURCE_WEATHER`
    and `SOURCE_ODDS_API` in `ALL_SOURCES`; comments document why
    umpire must run after MLB metadata and before statcast-features
  - `REQUIRED_ARTIFACT_FILES` + `NORMALIZED_ROOT_BY_SOURCE` entries
    for umpire
  - `backfill_historical` accepts an `umpire_runner` kwarg
    (defaulting to `ingest_umpire_for_date`) and routes it through
    the same resume / error-isolation path as the other sources
- `src/mlb_props_stack/modeling.py`
  - `StarterStrikeoutTrainingRow` grows seven new fields for the
    umpire assignment + rolling metrics
  - `ump_called_strike_rate_30d` and
    `ump_k_per_9_delta_vs_league_30d` join `OPTIONAL_NUMERIC_FEATURES`
    so they are gated by `OPTIONAL_FEATURE_MIN_COVERAGE` and will
    not be pinned into the model on sparse history
  - training-row construction uses `.get()` so older fixtures that
    predate the umpire schema still pass
- `src/mlb_props_stack/data_alignment.py`
  - `ArtifactCounts` gains `umpire_assignments` and
    `umpire_ok_snapshots`
  - `DateCoverageRow` gains `umpire_coverage`; it's computed as
    `ok / games` (missing-source rows do not count toward coverage)
  - `render_data_alignment_summary` prints two new columns
    (`umpire_ok`, `ump_cov`)
- `tests/test_umpire_ingest.py` (14 tests)
  - happy path via persisted feed/live, HTTP fallback when no
    persisted payload exists, missing-home-plate sentinel,
    HTTP-error sentinel, leakage guard (`captured_at <= commence_time`),
    negative `history_days` rejection, rolling-metric computation
    from prior umpire + Statcast history, `compute_rolling_umpire_metrics`
    degrading to `None` when no prior data is seeded,
    `normalize_feed_live_officials_payload` happy + missing-block
    paths, `load_latest_umpire_snapshots_for_date` returning the
    latest complete run (or `{}` when nothing exists), and the
    `DEFAULT_UMPIRE_HISTORY_DAYS` spec guard
- `tests/test_statcast_feature_ingest.py`
  - new assertions that all seven umpire fields arrive on the
    emitted `game_context_features.jsonl` rows (populated when a
    snapshot joins, `None` when the stub falls through)
- `tests/test_modeling.py`
  - the training-row fixture now writes the new umpire field names
    so the baseline training happy path still passes with the
    updated schema
- `tests/test_backfill.py`
  - every `ALL_SOURCES` test now constructs a `_RunnerSpy` for
    umpire and threads it through
    `backfill_historical(umpire_runner=…)`; counts updated
    (5 sources ingested instead of 4, 5 skipped instead of 4, etc.);
    the manifest test asserts the new `umpire` entry lands alongside
    mlb-metadata, weather, odds-api, and statcast-features; a new
    `test_backfill_historical_passes_target_date_and_output_dir_to_umpire_runner`
    confirms the orchestrator forwards kwargs to the umpire runner
- `tests/test_data_alignment.py`
  - `_counts` helper gains the two new `ArtifactCounts` fields with
    sensible defaults; new `test_collect_artifact_counts_reads_umpire_statuses`
    seeds a three-row `umpire_snapshots.jsonl` and asserts the
    status counts land correctly; the render-summary test asserts
    the `umpire_ok` and `ump_cov` headers appear

## Files Changed

- `docs/NEXT_SESSION_HANDOFF.md`
- `src/mlb_props_stack/backfill.py`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/data_alignment.py`
- `src/mlb_props_stack/ingest/__init__.py`
- `src/mlb_props_stack/ingest/statcast_features.py`
- `src/mlb_props_stack/ingest/umpire.py` (new)
- `src/mlb_props_stack/modeling.py`
- `tests/test_backfill.py`
- `tests/test_data_alignment.py`
- `tests/test_modeling.py`
- `tests/test_statcast_feature_ingest.py`
- `tests/test_umpire_ingest.py` (new)

## Verification

Commands run successfully before merge:

```bash
uv sync --extra dev
uv run pytest
uv run python -m mlb_props_stack.cli --help
uv run python -m mlb_props_stack.cli ingest-umpire --help
```

Observed results:

- full test suite passed: `177 passed` (up from `161` — sixteen
  additional tests: fourteen new `test_umpire_ingest.py` cases plus a
  new data-alignment umpire counts test and a new backfill umpire
  kwargs test)
- CLI help lists the `ingest-umpire` subcommand and the backfill help
  lists umpire alongside mlb-metadata, weather, odds-api, and
  statcast-features

## Recommended Next Issue

- Backfill 2024 + 2025 slates with the new umpire source to seed
  enough coverage for the gated numeric features
  (`ump_called_strike_rate_30d`, `ump_k_per_9_delta_vs_league_30d`)
  to clear `OPTIONAL_FEATURE_MIN_COVERAGE` and enter the model. Until
  then they're held out automatically. The adapter auto-mines the
  already-persisted feed/live payloads, so the bulk of the rebackfill
  is just `backfill-historical --force --sources umpire` across the
  seasons we already have MLB metadata for
- Evaluate whether K/9 via the `× 38.25` PA approximation materially
  differs from a proper innings-based K/9. If the Statcast
  `events == "strikeout"` marker plus inning metadata is already in
  `pitch_level_base.jsonl`, swap in a real innings denominator
- Consider decomposing `ump_called_strike_rate_30d` into in-zone
  vs. out-of-zone called-strike rates (needs `plate_x` / `plate_z` /
  `zone` columns). "Expanded the zone" and "shrunk the zone" are
  likely more useful signals than a single aggregate rate
- Once enough history exists, compare a "umpire-only" model ablation
  against the current baseline to confirm the two optional numerics
  actually clear calibration noise before they pin into the ladder

## Design Callouts

- Data source: MLB Stats API `feed/live` endpoint. Every scheduled
  game already persists a feed/live JSON under
  `data/raw/mlb_stats_api/date=<iso>/feed_live/game_pk=<pk>/captured_at=<ts>.json`
  as part of the MLB metadata ingest. Reusing it avoids adding an
  external scraping dependency on UmpScorecards or Retrosheet.
  Fresh HTTP fetches only happen when the persisted file is missing
  (e.g. the MLB metadata ingest ran before the umpire was
  announced); those fetches are batched one-per-game sequentially
  through the shared `MLBStatsAPIClient`, matching how every other
  MLB-Stats-API call works in this codebase
- K/9 approximation: the Statcast pitch-level base does not store
  innings; it stores plate appearances via the
  `is_plate_appearance_final_pitch` marker. K/9 is approximated as
  `K_rate × 38.25` (9 innings × ~4.25 PAs per inning). The constant
  is exposed as `APPROXIMATE_PA_PER_NINE_INNINGS` with an inline
  comment so anyone extending this knows to look first for a real
  innings denominator if one is added to the pitch-level base
- Leakage guard: `captured_at` is clamped to
  `min(now(), commence_time)` before emission, mirroring the weather
  ingest contract. The `ingest_umpire_for_date` caller asserts
  `captured_at <= commence_time` on every snapshot before writing to
  normalized so downstream features cannot leak post-pitch data
- Rolling-metric walker: two-pass over the prior 30 days. First pass
  loads `umpire_id -> {date -> {game_pks}}` from prior normalized
  umpire runs; second pass loads prior pitch_level_base per date
  and aggregates per `game_pk`. When a prior date's Statcast ingest
  hasn't run yet, the metric silently degrades to `None` rather than
  half-counting. Similarly, when the umpire has no prior games in
  the window, the metric returns `None` and the sentinel status
  stays as `ok` (the assignment itself is still trustworthy — only
  the history is missing)
- Sentinel semantics: `missing_umpire_source` covers three cases —
  feed/live payload is unreachable (HTTP error / JSON decode /
  filesystem error on the persisted file), feed/live payload has no
  `officialType == "Home Plate"` entry (assignment not published
  yet), or the Home Plate entry has a partial `official` block
  (id present but fullName missing, or vice versa). In all three
  cases `umpire_id` / `umpire_name` are `None`, `umpire_source` is
  `None`, and the rolling metrics are `None`. Coverage checks
  (`ump_cov` in `check-data-alignment`) count only `ok` rows
- Raw artifact shape: each persisted raw JSON stores `game_pk`,
  `captured_at`, `source`, `source_feed_live_path` (the feed/live
  file we read from, if any), the full extracted `officials` array,
  and the resolved `umpire_id` / `umpire_name` / `umpire_status`.
  Storing the full officials array means we can later extend to
  first/second/third base umps without re-scraping
- Coverage ratio: `umpire_coverage = umpire_ok_snapshots / games`,
  not `(ok + missing) / games`. A missing-source row still gets
  written (so coverage checks can tell the difference between
  "umpire ingest hasn't run" and "umpire ingest ran but the
  assignment wasn't published yet"), but it does not count toward
  coverage. This matches the weather ingest convention
- Adapter dispatch order: `mlb-metadata → weather → umpire → odds-api
  → statcast-features`. Umpire must run after MLB metadata (needs
  `games.jsonl` and persisted feed/live) and before statcast-features
  (which joins umpire snapshots into `game_context_features.jsonl`).
  The weather/umpire order is arbitrary — both depend only on MLB
  metadata — but keeping the order stable means resume-aware
  manifests stay diff-friendly

## Constraints And Open Questions

- No secondary umpire data source. If MLB Stats API is unreachable
  and no feed/live payload was ever persisted for a given game, the
  umpire assignment lands as `missing_umpire_source` with no
  fallback. UmpScorecards or Retrosheet could backstop that, but
  they'd require a scraping dependency and introduce diverging id
  spaces. Currently the mitigation is to re-run `ingest-umpire`
  after the MLB metadata ingest catches the published assignment
- Rolling metrics are computed at emit time from the most recent
  normalized runs. This means `ingest-umpire` must be re-run for
  every slate date — there's no incremental update or cache. For
  a full-season backfill the walker re-reads 30 days of
  `pitch_level_base.jsonl` per date, which is linear in
  `history_days × games`. If that becomes a bottleneck, precompute
  a per-umpire rolling aggregate during the Statcast ingest
- The K/9 PA approximation (`× 38.25`) is a known noise source. It
  does not affect `ump_called_strike_rate_30d` (which is pitch-level)
  but does bias `ump_k_per_9_delta_vs_league_30d` compared to a
  proper innings-based K/9. See "Recommended Next Issue" above

## Known Follow-Up Nits (Non-Blocking)

- `ingest-umpire` only persists raw artifacts for games where a
  feed/live payload was available. If the HTTP fallback also fails
  or the persisted file errors out, only the sentinel snapshot is
  written and no raw file lands on disk. This keeps the raw
  directory free of empty placeholders, but a missing raw file means
  a given game's `missing_umpire_source` status can only be
  explained by reading the sentinel row's `error_message`. Adding a
  `umpire_failures.jsonl` side-file would make gap investigations
  easier
- The handoff previously mentioned a potential integration test for
  `ingest-weather` → `ingest-statcast-features` threading; the same
  gap exists for `ingest-umpire` → `ingest-statcast-features`. The
  two halves are unit-tested separately (statcast test seeds an
  umpire snapshot JSONL directly), but an end-to-end CLI smoke test
  would catch path mismatches that unit tests cannot
- `_load_prior_umpire_game_pks_by_umpire` and
  `_load_prior_pitch_aggregates` both walk run directories
  independently. For a full-season backfill they could share a
  single walk. Low-priority optimization; correctness is
  straightforward with the current split
