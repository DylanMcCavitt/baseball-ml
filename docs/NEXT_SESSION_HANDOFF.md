# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Last completed issue: `AGE-203` on branch
  `dylan/eloquent-mclaren-c1bbe8`
- This slice replaces the `missing_park_factor_source` placeholder in the
  game-context feature build with a static FanGraphs-anchored park
  strikeout factor joined by MLB `venue_id`, and emits `park_k_factor`,
  `park_k_factor_vs_rhh`, and `park_k_factor_vs_lhh` on every slate row
- Current status: `game_context_features.jsonl` now carries non-null park
  factors for every venue present in the static lookup (17 parks covered
  with low venue ids plus 13 high-id stadiums). Rows for unknown venue
  ids fall back to `park_factor_status = "missing_park_factor_source"`
  and null factor values, so the placeholder is preserved only in the
  case the issue requires

## What Was Completed In This Slice

- `data/static/park_factors/park_k_factors.csv` (new static lookup)
  - one row per `(season, venue_mlb_id)` with columns `season`,
    `venue_mlb_id`, `venue_name`, `park_k_factor`,
    `park_k_factor_vs_rhh`, `park_k_factor_vs_lhh`
  - values are three-year rolling FanGraphs Guts! park factors for
    starter strikeouts, converted from the FanGraphs 100-scale to a
    ratio centered on `1.00`
  - 2026 rows carry 2025 values forward until the season-end FanGraphs
    refresh lands
- `data/static/park_factors/README.md`
  - documents the schema, source, vintage, and how to extend the table
    with new MLB venue ids
- `src/mlb_props_stack/ingest/park_factors.py` (new module)
  - `ParkKFactorRecord` frozen dataclass
  - `load_park_k_factors(path=DEFAULT_PARK_FACTORS_PATH)` CSV loader
    that skips rows whose `season` or `venue_mlb_id` fail to parse so a
    partially-edited file still loads the valid rows
  - `lookup_park_k_factor(season, venue_mlb_id, table=None)` helper
    that returns the record for the requested season and falls back to
    the prior season when the requested one is missing (so an
    end-of-season slate keeps working before the next Guts! publication)
  - `PARK_FACTOR_STATUS_OK` / `PARK_FACTOR_STATUS_MISSING_SOURCE`
    constants shared with the feature builder
- `src/mlb_props_stack/ingest/statcast_features.py`
  - `GameContextFeatureRow` gains `park_k_factor`,
    `park_k_factor_vs_rhh`, `park_k_factor_vs_lhh` (the legacy
    always-null `park_factor` field is renamed to `park_k_factor`)
  - `_build_game_context_feature_row` now accepts a `park_k_factor_table`
    keyword and populates the three new fields; `park_factor_status` is
    set to `"ok"` on a successful join and only retains the
    `missing_park_factor_source` placeholder when the venue id is not in
    the lookup
  - `ingest_statcast_features_for_date` loads the default table once at
    the top of the feature pass and threads it through every row build
- `src/mlb_props_stack/modeling.py`
  - `StarterStrikeoutTrainingRow` gains the three new park-factor fields
  - `OPTIONAL_NUMERIC_FEATURES` now includes `park_k_factor`,
    `park_k_factor_vs_rhh`, and `park_k_factor_vs_lhh` so the
    vectorizer picks them up for the starter strikeout baseline
- `tests/test_park_factors.py` (new file, 6 tests)
  - loader round-trip for the committed CSV
  - prior-season fallback
  - unknown-venue and missing `venue_mlb_id` returning `None`
  - skip rows with blank keys
  - status-constant stability
- `tests/test_statcast_feature_ingest.py`
  - the existing happy-path test now asserts `park_factor_status == "ok"`
    and the three emitted factor values for `venue_id=5`
  - new test `test_ingest_statcast_features_preserves_missing_park_factor_source_for_unknown_venue`
    rewrites the seeded game's `venue_id` to `999999` and verifies the
    feature row falls back to `missing_park_factor_source` with all
    three factor fields null
- `tests/test_modeling.py`
  - fixture now seeds the three new park-factor fields with
    `park_factor_status = "ok"` so the training-row loader reads the
    populated values
- `README.md`
  - "missing inputs" block updated to describe the static lookup join
    and the new emitted fields

## Files Changed

- `README.md`
- `data/static/park_factors/README.md`
- `data/static/park_factors/park_k_factors.csv`
- `docs/NEXT_SESSION_HANDOFF.md`
- `src/mlb_props_stack/ingest/park_factors.py`
- `src/mlb_props_stack/ingest/statcast_features.py`
- `src/mlb_props_stack/modeling.py`
- `tests/test_modeling.py`
- `tests/test_park_factors.py`
- `tests/test_statcast_feature_ingest.py`

## Verification

Commands run successfully before merge:

```bash
uv sync --extra dev
uv run pytest
uv run python -m mlb_props_stack
```

Observed results:

- full test suite passed: `130 passed` (up from `121` on the previous
  slice; the nine additional tests are the six in
  `tests/test_park_factors.py`, the new unknown-venue test in
  `tests/test_statcast_feature_ingest.py`, and two existing assertions
  updated to read the new fields)
- `uv run python -m mlb_props_stack` still renders the runtime summary

## Recommended Next Issue

- Run the overnight 2024 + 2025 regular-season backfill that was queued
  up by AGE-201 (out of scope here). With park factors now flowing
  through the feature build, a fresh
  `train-starter-strikeout-baseline` over the backfill window should
  exercise the three new `park_k_factor*` features and produce a
  non-zero coefficient for at least one of the handedness splits — that
  is the remaining verification step called out on the AGE-203 issue
- Use that same training run to sanity-check the park factor sign and
  magnitude: Coors Field (venue 22) should land near `0.94` on the K
  factor and the trained model coefficient should reward starters
  projected there for fewer strikeouts, all else equal
- If new MLB venues appear in 2027+ schedules, extend
  `data/static/park_factors/park_k_factors.csv` before the first slate
  that uses the new `venue_mlb_id`; otherwise the feature row will fall
  back to `missing_park_factor_source` and the trained model will see a
  masked feature for those games

## Constraints And Open Questions

- Values in the static CSV are anchored to FanGraphs Guts! three-year
  averaged park factors, not the single-season refresh. That matches
  how most props shops treat park factors (they're noisy year-over-year
  and the rolling average is what's actually actionable), but it does
  mean the 2025 and 2026 rows are identical in this first cut — rotate
  them separately once the FanGraphs 2026 pull is available
- The default lookup is loaded lazily per feature run via
  `load_park_k_factors()` rather than cached at module import. That
  keeps the CSV editable during a running daemon / dashboard session
  but means each ingest call pays a one-shot CSV parse (still cheap —
  60 rows in the current file, runs in microseconds)
- `lookup_park_k_factor` deliberately does not fall back to the
  overall-league average when a venue is missing. The issue wants the
  `missing_park_factor_source` placeholder preserved for unknown venue
  ids so the modeling layer can notice the masked row, and so a silent
  league-average substitution can't mask an extension gap in the CSV
- The legacy `park_factor` field (which was always `None`) is renamed
  to `park_k_factor`. Any downstream consumer that reads the raw JSONL
  and expects the old key will need to be updated — the repo's own
  modeling and paper-tracking paths have already been migrated, but
  external notebooks operating on older artifacts will continue to see
  the pre-AGE-203 shape
