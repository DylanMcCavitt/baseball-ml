# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Last completed issue: `AGE-211` on branch
  `dylan/blissful-zhukovsky-ae89e4`
- This slice is a pure refactor of
  `src/mlb_props_stack/ingest/statcast_features.py`. The previous
  ~1,600-line file conflated raw CSV ingest, pitcher-level feature
  derivation, lineup-level aggregation, slate-level game-context
  derivation, and orchestration. It has been split into four focused
  modules under `src/mlb_props_stack/ingest/` while the orchestrator
  module keeps its name and re-exports every symbol its callers (and
  tests) already depended on, so the public surface is unchanged and
  existing tests pass without edits
- Current status: `uv run pytest` is green at 178 tests (same count
  and outcome as before the refactor), `uv run python -m
  mlb_props_stack.cli --help` still lists the full subcommand surface
  including `ingest-statcast-features`, and `uv run python -m
  mlb_props_stack` continues to print the runtime configuration
  banner. No data alignment, modeling, or CLI behavior changed

## What Was Completed In This Slice

- `src/mlb_props_stack/ingest/statcast_ingest.py` (new module)
  - Ingest foundation shared by every feature module: the Baseball
    Savant endpoint / request headers, default fetch-attempt and
    backoff tuning constants (`DEFAULT_MAX_FETCH_ATTEMPTS`, ...,
    `DEFAULT_MAX_FETCH_WORKERS`), the strikeout / whiff / called-strike
    / contact / swing classification sets, and `STRIKEOUT_EVENTS` plus
    the description sets that `normalize_statcast_csv_text` uses
  - Data contracts: `StatcastPullRecord` (one raw CSV pull manifest
    entry) and `StatcastPitchRecord` (one normalized pitch-level base
    row)
  - HTTP client: `StatcastSearchClient` plus `_is_retriable_http_error`
    and `_fetch_csv_texts_concurrently`. The client deliberately
    dispatches through a small `_urlopen` shim that does a deferred
    `from . import statcast_features` and calls
    `statcast_features.urlopen(...)` at call time so the existing
    `monkeypatch.setattr("mlb_props_stack.ingest.statcast_features.urlopen", ...)`
    test hooks keep working after the move (documented inline)
  - URL builder: `build_statcast_search_csv_url`
  - Normalizer: `normalize_statcast_csv_text` plus helpers
    (`_optional_text`, `_coerce_optional_int`, `_coerce_optional_float`,
    `_pitch_record_id`, `_batting_team_abbreviation`, `_is_out_of_zone`)
  - Shared pitch-record utilities needed by every feature module
    because they operate on `StatcastPitchRecord`: `_sorted_rows`,
    `_pitch_rows_for_player`, `_batter_rows`, `_plate_appearance_key`,
    `_count_plate_appearances`, `_last_game_date`, `_pitch_type_usage`,
    `_rows_in_recent_window`, `_rows_grouped_by_start`,
    `_history_cutoff`, `_safe_rate`, `_round_optional`, `_mean`
  - Team/lineup helpers shared across feature modules:
    `_opponent_team_side`, `_opponent_team`,
    `_select_pregame_lineup_snapshot`
- `src/mlb_props_stack/ingest/pitcher_features.py` (new module)
  - `PitcherDailyFeatureRow` dataclass
  - `_build_pitcher_daily_feature_row` plus the pitcher-only helpers
    `_pitcher_hand_split_rates`, `_pitcher_hand`, and `_expected_leash`
    (the last is shared with `game_context`, see below)
- `src/mlb_props_stack/ingest/lineup_aggregation.py` (new module)
  - `LineupDailyFeatureRow` dataclass
  - `_BatterMetricBundle`, `_batter_metric_bundle`,
    `_batter_k_rate_vs_p_throws`, `_batting_order_weight`,
    `_weighted_mean`, `_latest_prior_team_lineup_player_ids`, and
    `_build_lineup_daily_feature_row`
- `src/mlb_props_stack/ingest/game_context.py` (new module)
  - `GameContextFeatureRow` dataclass
  - `_build_game_context_feature_row` with the park-factor / weather /
    umpire joins and the rest-days + expected-leash calculations. The
    module imports `_expected_leash` from `pitcher_features` rather
    than duplicating the helper, which establishes the only
    intra-feature dependency (`game_context → pitcher_features`)
- `src/mlb_props_stack/ingest/statcast_features.py` (slimmed to the
  orchestrator)
  - Owns only: `DEFAULT_HISTORY_DAYS`,
    `StatcastFeatureIngestResult`, `_LoadedMLBMetadata`, the MLB
    metadata loaders (`_latest_complete_run_dir`,
    `_latest_pregame_valid_run_dir`, `_run_is_pregame_valid`,
    `_load_latest_mlb_metadata_for_date`, `_load_jsonl_rows`), the
    filesystem writer helpers (`_path_timestamp`, `_json_ready`,
    `_write_text`, `_write_json`, `_write_jsonl`), and
    `ingest_statcast_features_for_date`
  - Keeps `from urllib.request import Request, urlopen` at module
    scope so tests can patch `statcast_features.urlopen`; the
    `StatcastSearchClient.fetch_csv` call in `statcast_ingest.py`
    dispatches through this symbol at call time via the `_urlopen`
    shim described above
  - Re-exports every symbol its callers (`ingest/__init__.py`,
    `modeling.py`, `tests/test_statcast_feature_ingest.py`,
    `tests/test_ingest_atomic_writes.py`) previously imported from
    this file. That includes the public data contracts
    (`GameContextFeatureRow`, `LineupDailyFeatureRow`,
    `PitcherDailyFeatureRow`, `StatcastFeatureIngestResult`,
    `StatcastPitchRecord`, `StatcastPullRecord`,
    `StatcastSearchClient`, `DEFAULT_HISTORY_DAYS`,
    `DEFAULT_MAX_FETCH_WORKERS`, `build_statcast_search_csv_url`,
    `ingest_statcast_features_for_date`, `normalize_statcast_csv_text`),
    the test-referenced private helpers (`_pitcher_hand_split_rates`,
    `_write_jsonl`), and every remaining underscore-prefixed helper
    that lived on the original module so attribute lookups against
    `statcast_features.<helper>` keep working
- `src/mlb_props_stack/ingest/__init__.py`
  - Unchanged. Still imports from `.statcast_features`, which now
    re-exports the moved symbols from the submodules, so the ingest
    package's public surface is byte-identical

## Files Changed

- `docs/NEXT_SESSION_HANDOFF.md`
- `src/mlb_props_stack/ingest/game_context.py` (new)
- `src/mlb_props_stack/ingest/lineup_aggregation.py` (new)
- `src/mlb_props_stack/ingest/pitcher_features.py` (new)
- `src/mlb_props_stack/ingest/statcast_features.py` (slimmed)
- `src/mlb_props_stack/ingest/statcast_ingest.py` (new)

## Verification

Commands run successfully during this issue:

```bash
uv sync --extra dev
uv run pytest
uv run python -m mlb_props_stack.cli --help
uv run python -m mlb_props_stack.cli ingest-statcast-features --help
uv run python -m mlb_props_stack
```

Observed results:

- full test suite passed: `178 passed` (same count as before the
  refactor; no test files were modified, matching the issue
  acceptance criteria)
- `ingest-statcast-features` subcommand still appears in the top-level
  help output alongside every other ingest subcommand
- `uv run python -m mlb_props_stack` prints the same runtime
  configuration banner it printed before the refactor

Not run this slice (see Constraints):

- The issue packet's `uv run python -m mlb_props_stack
  ingest-statcast-features --date 2026-04-21` byte-identical reference
  comparison was not executed because the local repo has no persisted
  `data/normalized/mlb_stats_api/date=2026-04-21/` run to drive the
  CLI, and that CLI call requires network access to Baseball Savant
  for the Statcast CSV pulls. The 178-test suite covers the ingest
  end-to-end against seeded metadata + CSV fixtures, including the
  `urlopen` monkeypatch, pull ordering, parallel fetch, dedupe, park
  factor / weather / umpire joins, and each of the three feature
  builders, so the behavioral contract is exercised even without a
  real slate. Anyone with a pregame slate handy should still run the
  byte-identical comparison before merging downstream refactors
  against the same file

## Recommended Next Issue

- Before cascading more feature families into the new submodules,
  validate the byte-identical artifact check from the AGE-211 issue
  packet on a real slate: capture
  `data/normalized/statcast_search/date=<iso>/run=<ts>/` JSONL files
  from `main` and this branch for the same `--date`, diff them
  modulo run-id paths, and record the result. That closes the last
  verification gap this refactor could not exercise offline
- The previous handoff's follow-ups still stand: backfill 2024 + 2025
  slates with the new umpire source so the gated numeric umpire
  features clear `OPTIONAL_FEATURE_MIN_COVERAGE`, evaluate whether the
  `× 38.25` PA approximation in `ump_k_per_9_delta_vs_league_30d`
  materially differs from an innings-based K/9, and consider
  decomposing `ump_called_strike_rate_30d` into in-zone vs.
  out-of-zone rates using `plate_x`/`plate_z`/`zone` columns
- With the new seams in place, the next feature family (park-of-pitch
  batter splits, bullpen carryover, etc.) can land in its own module
  under `src/mlb_props_stack/ingest/` without growing any of the four
  that already exist. Any additional shared helpers should continue
  to live in `statcast_ingest.py` so the feature modules stay leaf
  consumers

## Design Callouts

- Four-module split matches the AGE-211 packet exactly:
  `statcast_ingest.py` (CSV fetch + parsing + normalization +
  foundation helpers), `pitcher_features.py` (pitcher derivation),
  `lineup_aggregation.py` (lineup derivation), `game_context.py`
  (game-context derivation). The original `statcast_features.py`
  stays as the orchestrator that loads inputs and writes artifacts,
  per the packet's "keep `statcast_features.py` as the orchestrator"
  bullet
- Shared helpers (e.g. `_sorted_rows`, `_safe_rate`, `_mean`) live in
  `statcast_ingest.py` rather than a fifth `_helpers.py` module so
  the packet's "four focused modules" acceptance is respected. They
  operate on `StatcastPitchRecord` (defined in the same file), so
  their home is semantically coherent
- `_expected_leash` is defined in `pitcher_features.py` but imported
  by `game_context.py` because leash modeling is a pitcher-centric
  statistic and `game_context` already conceptually depends on the
  pitcher slot. This is the only cross-feature import; the other
  three submodules are leaf consumers of `statcast_ingest.py`
- Monkeypatch compatibility: `tests/test_statcast_feature_ingest.py`
  patches `mlb_props_stack.ingest.statcast_features.urlopen` at three
  sites. The orchestrator keeps its `from urllib.request import
  Request, urlopen` line so that attribute exists, and
  `statcast_ingest.StatcastSearchClient.fetch_csv` calls
  `_urlopen(...)` which does a deferred `from . import
  statcast_features; statcast_features.urlopen(...)` lookup at call
  time. This means the monkeypatch keeps taking effect even though
  the HTTP client moved. The reasoning is documented inline on both
  the shim and the orchestrator import
- `_write_jsonl` compatibility:
  `tests/test_ingest_atomic_writes.py` treats
  `mlb_props_stack.ingest.statcast_features._write_jsonl` as a
  module-level attribute and parameterizes across the three ingest
  modules. `_write_jsonl` lives in the orchestrator (it is the
  write-side of the orchestrator, not an ingest-layer concern), so
  the test's lookup resolves directly; no re-export indirection is
  needed
- Re-export lists in `statcast_features.py` are intentionally broad.
  Because the original file's underscore-prefixed helpers were
  module-level attributes, any downstream code or test that did
  `statcast_features._foo` needs `_foo` to remain on that module
  after the move. The orchestrator therefore imports every moved
  helper back by name so attribute lookups against the orchestrator
  keep resolving. This is the lightest path to "existing tests pass
  unchanged" without forcing test edits
- `__init__.py` did not need changes. Its `from .statcast_features
  import (...)` line still works because the orchestrator continues
  to expose the same names, re-exported from the new submodules
- Module sizes after the split (for reference when the next feature
  family lands): orchestrator 610 lines (of which roughly half is
  re-export wiring; the orchestrator body is ~360 lines),
  `statcast_ingest.py` 555 lines, `lineup_aggregation.py` 260 lines,
  `pitcher_features.py` 238 lines, `game_context.py` 191 lines.
  Every feature-derivation module is well under the 47 KB / 1,600-line
  bar the original file had crossed

## Constraints And Open Questions

- Byte-identical artifact comparison from the issue packet was not
  run locally (see "Verification / Not run this slice" above). If
  reviewers want it run before merge, they should pick a slate date
  with persisted `data/normalized/mlb_stats_api/date=<iso>/` metadata
  and diff the `statcast_search` run outputs between `main` and this
  branch
- The re-export block on `statcast_features.py` lists every moved
  helper by name. This is by design (tests/grep-accessible
  attributes) but it means adding or renaming a helper in a
  submodule in future issues will require touching the orchestrator's
  import list too. Prefer doing renames as their own isolated slices
  so the diffs stay auditable
- The `_urlopen` lazy-dispatch shim in `statcast_ingest.py` exists
  solely to keep the existing test monkeypatch targeting
  `statcast_features.urlopen` functional. A future cleanup could move
  the monkeypatch target to `statcast_ingest.urlopen` (or to an
  injectable `urlopen=` kwarg on `StatcastSearchClient`) and drop the
  shim, but that is out of scope for a no-behavior-change refactor
- No `from urllib.request import urlopen` call site in production
  code uses the local name after the move — the orchestrator's
  `urlopen` import is load-bearing only for the monkeypatch contract.
  The `# noqa: F401` annotation documents this explicitly

## Known Follow-Up Nits (Non-Blocking)

- Import list in the orchestrator is long (~50 names). A future
  hygiene pass could drop any underscore-prefixed names that turn
  out to be unused outside the submodules once tests explicitly
  target the submodule attribute instead. That would need coordinated
  test edits, which AGE-211 intentionally avoided
- `_expected_leash` cross-module import establishes `game_context
  → pitcher_features`. If a future issue decomposes pitcher feature
  derivation further, consider promoting `_expected_leash` to
  `statcast_ingest.py` (it only reads `StatcastPitchRecord`) so
  `game_context` can depend on the foundation module directly
- The orchestrator still owns `_LoadedMLBMetadata` and the MLB
  metadata loader helpers. These are arguably orchestration
  responsibilities, but they could also fit into a dedicated
  `mlb_metadata_loader.py` submodule if future work adds more
  reading-side helpers for MLB metadata. Not worth a dedicated issue
  by itself
- `DEFAULT_HISTORY_DAYS` lives in the orchestrator while
  `DEFAULT_MAX_FETCH_WORKERS` lives in `statcast_ingest.py`. They are
  both re-exported via the orchestrator for `ingest/__init__.py`
  compatibility. The split is intentional (`HISTORY_DAYS` is an
  orchestrator default, `MAX_FETCH_WORKERS` is a fetch-client
  default) but the asymmetry is worth flagging for future readers
