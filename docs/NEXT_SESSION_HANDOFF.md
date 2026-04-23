# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Last completed issue: `AGE-205` on branch
  `dylan/determined-bartik-9ed6da`
- This slice replaces the `missing_weather_source` stub on
  `game_context_features.jsonl` with a real pregame weather snapshot
  per game, sourced from Open-Meteo's free Archive API, anchored to
  `commence_time - 60 minutes` at the venue's lat/lon. Closed-roof
  stadiums emit a `roof_closed` sentinel row instead of hitting the
  weather API, retractable-roof parks are ingested as outdoor, and
  games whose venue is missing from the static metadata table land
  with the existing `missing_venue_metadata` status so downstream
  callers keep running
- Current status: weather ingest runs standalone
  (`uv run python -m mlb_props_stack.cli ingest-weather --date …`),
  inside the historical backfill orchestrator (new `weather` source
  slot between `mlb-metadata` and `odds-api`), and the statcast
  feature ingest now joins the latest weather snapshots into every
  emitted `game_context_features.jsonl` row. `check-data-alignment`
  surfaces weather coverage in a new `wx_cov` column plus
  per-status breakdowns (`weather_ok`, `weather_roof`). Training rows
  expose four new numeric fields (`weather_temperature_f`,
  `weather_wind_speed_mph`, `weather_humidity_pct`) and three
  traceability fields (`weather_source`, `weather_captured_at`,
  `roof_type`); the numerics are gated by `OPTIONAL_FEATURE_MIN_COVERAGE`
  so sparse weather history will not silently regress the model

## What Was Completed In This Slice

- `src/mlb_props_stack/ingest/weather.py` (new module)
  - `VenueMetadata` + `load_venue_metadata` + `lookup_venue_metadata`
    read the curated CSV at `data/static/venues/venue_metadata.csv`,
    validate `roof_type` against `{"open", "retractable", "fixed"}`,
    and return a frozen dataclass keyed on `venue_id`
  - `OpenMeteoClient` + `build_open_meteo_archive_url` call the
    Archive API with UTC timestamps and fahrenheit/mph units; the
    client is injected so tests never hit the network
  - `normalize_open_meteo_payload` picks the hourly row whose
    timestamp is `<= commence_time - offset_minutes` so the
    leakage guard is enforced at the source boundary; it raises
    `ValueError` on malformed payloads and returns `None` when no
    pre-commence row is available
  - `ingest_weather_for_date` walks the latest `games.jsonl`, emits
    one snapshot per game (outdoor call, closed-roof sentinel,
    missing-venue-metadata fallback, or missing-weather-source
    fallback when the client raises), and writes raw + normalized
    JSONL under `data/{raw,normalized}/weather/date=<iso>/run=<ts>/`
  - `load_latest_weather_snapshots_for_date` returns a
    `dict[int, WeatherSnapshotRecord]` keyed on `game_pk` for the
    latest complete run, used by the statcast feature ingest
- `src/mlb_props_stack/ingest/statcast_features.py`
  - `GameContextFeatureRow` gains `weather_source`,
    `weather_temperature_f`, `weather_wind_speed_mph`,
    `weather_wind_direction_deg`, `weather_humidity_pct`,
    `weather_captured_at`, and `roof_type`; the two old stub fields
    (`weather_wind_mph`, `weather_conditions`) are gone
  - `_build_game_context_feature_row` now joins via a
    `weather_lookup: dict[int, WeatherSnapshotRecord]` passed in
    from the top-level ingest call
- `src/mlb_props_stack/ingest/__init__.py`
  - re-exports the weather dataclasses, client, ROOF_TYPE_* /
    WEATHER_STATUS_* constants, and both weather entry points
- `src/mlb_props_stack/cli.py`
  - new `ingest-weather` subcommand (`--date`, `--output-dir`)
  - `backfill-historical` help copy now lists the weather source
- `src/mlb_props_stack/backfill.py`
  - new `SOURCE_WEATHER = "weather"` placed between MLB metadata and
    odds-api in `ALL_SOURCES` so weather runs after `games.jsonl`
    exists (its dependency) and before statcast-features joins it
  - `REQUIRED_ARTIFACT_FILES` + `NORMALIZED_ROOT_BY_SOURCE` entries
    for weather
  - `backfill_historical` accepts a `weather_runner` kwarg
    (defaulting to `ingest_weather_for_date`) and routes it through
    the same resume / error-isolation path as the other sources
- `src/mlb_props_stack/modeling.py`
  - `StarterStrikeoutTrainingRow` grows seven new fields for the
    weather snapshot + roof type
  - `weather_temperature_f`, `weather_wind_speed_mph`, and
    `weather_humidity_pct` join `OPTIONAL_NUMERIC_FEATURES` so they
    are gated by `OPTIONAL_FEATURE_MIN_COVERAGE` and will not be
    pinned into the model on sparse history
  - training-row construction uses `.get()` everywhere so tests that
    stub the old `game_context_features.jsonl` schema keep working
- `src/mlb_props_stack/data_alignment.py`
  - `ArtifactCounts` gains `weather_snapshots`, `weather_ok_snapshots`,
    and `weather_roof_closed_snapshots`
  - `DateCoverageRow` gains `weather_coverage`; it's computed as
    `(ok + roof_closed) / games` so closed-roof dates do not drag
    coverage even though they never hit the weather API
  - `render_data_alignment_summary` prints three new columns
    (`weather_ok`, `weather_roof`, `wx_cov`)
- `data/static/venues/venue_metadata.csv` + `README.md`
  - curated 30-team stadium table with `venue_id`, lat/lon, and
    `roof_type`; README documents the Open-Meteo choice, leakage
    guard, and roof semantics (closed -> sentinel, retractable ->
    outdoor)
- `tests/test_weather_ingest.py` (15 tests)
  - happy path outdoor snapshot, fixed-roof sentinel,
    retractable-roof treated as outdoor, missing-venue-metadata
    fallback, missing-weather-source fallback when the client
    raises, leakage guard never returns a row past `commence_time`,
    negative `offset_minutes` rejected, URL builder parameters,
    payload normalize happy + error paths, venue CSV loader
    happy + unknown-roof rejection, and
    `load_latest_weather_snapshots_for_date` returning the latest
    complete run (or `{}` when nothing exists)
- `tests/test_statcast_feature_ingest.py`
  - new assertions that all seven weather fields arrive on the
    emitted `game_context_features.jsonl` rows (populated when a
    weather snapshot joins, `None` when the stub falls through)
- `tests/test_modeling.py`
  - the training-row fixture now writes the new field names so the
    baseline training happy path still passes with the updated
    schema
- `tests/test_backfill.py`
  - every `ALL_SOURCES` test now constructs a `_RunnerSpy` for
    weather and threads it through
    `backfill_historical(weather_runner=…)`; counts updated
    (4 ingested instead of 3, 4 skipped instead of 3, etc.); the
    manifest test asserts the new `weather` entry lands alongside
    mlb-metadata, odds-api, and statcast-features
- `tests/test_data_alignment.py`
  - `_counts` helper gains the three new `ArtifactCounts` fields
    with sensible defaults; new `test_collect_artifact_counts_reads_weather_statuses`
    seeds a three-row `weather_snapshots.jsonl` and asserts the
    status counts land correctly; the render-summary test asserts
    the `weather_ok`, `weather_roof`, and `wx_cov` headers appear

## Files Changed

- `docs/NEXT_SESSION_HANDOFF.md`
- `data/static/venues/README.md`
- `data/static/venues/venue_metadata.csv`
- `src/mlb_props_stack/backfill.py`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/data_alignment.py`
- `src/mlb_props_stack/ingest/__init__.py`
- `src/mlb_props_stack/ingest/statcast_features.py`
- `src/mlb_props_stack/ingest/weather.py`
- `src/mlb_props_stack/modeling.py`
- `tests/test_backfill.py`
- `tests/test_data_alignment.py`
- `tests/test_modeling.py`
- `tests/test_statcast_feature_ingest.py`
- `tests/test_weather_ingest.py`

## Verification

Commands run successfully before merge:

```bash
uv sync --extra dev
uv run pytest
uv run python -m mlb_props_stack
uv run python -m mlb_props_stack.cli --help
```

Observed results:

- full test suite passed: `161 passed` (up from `145` on the
  previous slice; sixteen additional tests — fifteen new
  `test_weather_ingest.py` cases plus a new data-alignment weather
  counts test)
- CLI help now lists the `ingest-weather` subcommand and the
  backfill help lists weather between `mlb-metadata` and `odds-api`
- `uv run python -m mlb_props_stack` still renders the stack
  defaults cleanly (nothing weather-specific in the runtime
  summary — intentional, since weather feeds the model, not the
  scoring config)

## Recommended Next Issue

- Backfill 2024 + 2025 slates with the new weather source to seed
  enough coverage for the gated numeric features
  (`weather_temperature_f`, `weather_wind_speed_mph`,
  `weather_humidity_pct`) to clear `OPTIONAL_FEATURE_MIN_COVERAGE`
  and enter the model. Until then they're held out automatically
- Consider splitting wind into parallel / perpendicular components
  against the batting orientation (needs stadium bearing metadata),
  or deriving a simple wind-helps-hitters / wind-helps-pitchers
  signal. Raw `wind_speed_mph` is likely less useful than a
  directional feature
- Once we have enough history, evaluate whether humidity and
  temperature actually improve starter-strikeout calibration — the
  literature disagrees and it is easy to add noise here

## Constraints And Open Questions

- Open-Meteo Archive serves pregame weather for the full
  historical window we care about, but the API has a ~1000-point
  hourly cap per request; the current client fetches one day at a
  time, so long backfills cost one HTTP call per game per day. If
  we need to amortize, batch by venue over a date window
- The venue CSV hard-codes lat/lon and roof type. If a team moves
  (temporary relocations, spring training, London / Seoul / Tokyo
  series), the row must be updated by hand — there is no runtime
  fallback that infers coordinates from `venue_name`
- The leakage guard is "hourly row whose timestamp `<= commence_time
  - offset_minutes`". A game that starts on the hour (e.g. 19:00Z)
  pulls the 18:00Z row, so the recorded `captured_at` is typically
  between 61 and 119 minutes before first pitch — safe, but the
  recorded value is the hourly row, not the exact 60-minute offset
- `roof_closed` is emitted for every `fixed`-roof stadium regardless
  of the actual weather that day (Tropicana Field). Retractable-roof
  parks are always treated as outdoor; the model does not yet know
  whether the roof was actually closed for a given game, because
  the MLB Stats API does not expose a reliable historical roof-state
  field. This is a likely source of noise in Toronto, Arizona,
  Miami, Seattle, Minneapolis, Houston, Milwaukee, and Texas games
- Weather numerics are gated by `OPTIONAL_FEATURE_MIN_COVERAGE`; if
  the next training run's weather coverage is below that threshold
  across the training window, these features will silently drop
  out of the model. That is the intended safety rail, but worth
  watching after the first backfill

## Known Follow-Up Nits (Non-Blocking)

- The venue metadata CSV is checked into `data/static/venues/`.
  It's tiny (30 rows) but the directory layout implies a larger
  static catalog — consider moving it to `docs/static/` or
  `src/mlb_props_stack/static/` if we never grow past a couple of
  tables
- `load_latest_weather_snapshots_for_date` returns `{}` when no
  complete run exists; the statcast feature ingest treats that as
  "stub everything as `missing_weather_source`". If we ever want to
  block statcast-feature ingest on weather availability we'll need
  a stricter mode, but that is out of scope for this slice
- There is no explicit integration test that exercises
  `ingest-weather` -> `ingest-statcast-features` end to end; the
  two sides are covered by unit-level fixtures and the statcast
  ingest test seeds a weather JSONL directly. A full pipeline smoke
  test that threads both ingests through the CLI would catch path
  mismatches that unit tests cannot
