# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Last completed issue: `AGE-201` on branch
  `feat/age-201-backfill-historical`
- This slice adds the `backfill-historical` CLI subcommand and a new
  `mlb_props_stack.backfill` module that walks an inclusive date window
  and replays `ingest-mlb-metadata`, `ingest-odds-api-lines`, and
  `ingest-statcast-features` for each calendar date with idempotent
  resume and best-effort per-source failure handling
- Current status: a season-long backfill can be re-invoked safely after a
  crash or interrupt — every source for every date is checked against
  the latest normalized run on disk first, so only missing artifacts are
  re-fetched. Source-level exceptions are captured per-date so a sparse
  Odds API response cannot abort the rest of the sweep, and a manifest
  records the per-date outcome under `data/normalized/backfill/run=.../`

## What Was Completed In This Slice

- `src/mlb_props_stack/backfill.py` (new module)
  - `iter_backfill_dates(start, end)`: inclusive calendar-day iterator
    that rejects inverted windows
  - `is_source_complete(output_dir, source, target_date)`: returns True
    when the latest `run=...` directory under
    `data/normalized/<source-root>/date=<iso>/` contains every required
    artifact file (mirrors the "latest run wins" rule used by
    `data_alignment._latest_run_dir_for_date`)
  - `normalize_sources(seq)`: validates the requested source list,
    rejects unknown values and empty input, and dedupes while preserving
    declared order
  - `backfill_historical(...)`: core orchestration helper. For each date
    and each source it skips when the artifacts are already complete and
    `force=False`; otherwise it invokes the matching ingest runner and
    records its `run_id`. Source-level `Exception`s are captured per
    date so the sweep continues; `BaseException` (KeyboardInterrupt,
    SystemExit) still propagates immediately.
  - manifest writes go through `<name>.tmp` + `os.replace`, matching the
    atomicity guarantee from AGE-200
  - exposes `BackfillResult`, `BackfillDateOutcome`,
    `BackfillSourceOutcome`, plus `STATUS_INGESTED`,
    `STATUS_SKIPPED_RESUME`, `STATUS_FAILED`, `ALL_SOURCES`, and the
    per-source required-artifact list
- `src/mlb_props_stack/__init__.py`
  - adds `"backfill"` to `__all__`
- `src/mlb_props_stack/cli.py`
  - imports the new backfill helpers and adds
    `render_backfill_historical_summary`
  - adds the `backfill-historical` subparser with `--start-date`,
    `--end-date`, `--output-dir`, `--sources` (comma-separated, defaults
    to all three), `--force`, `--history-days` (default
    `DEFAULT_HISTORY_DAYS` from the Statcast ingest), and `--api-key`
  - dispatches to `backfill_historical`, prints the summary, and exits
    non-zero whenever any source recorded `failed`
- `tests/test_backfill.py` (new file, 18 tests)
  - covers `iter_backfill_dates` window validation
  - covers `normalize_sources` dedup, ordering, unknown-source rejection,
    and empty-list rejection
  - covers `is_source_complete` requiring every artifact and falling
    back to an older complete run when a newer run only has a partial
    write (the AGE-200 guarantee in action)
  - covers `backfill_historical` invoking each runner per date,
    skipping complete dates, re-ingesting under `--force`, only
    re-ingesting missing sources for partially-complete dates,
    continuing past a single failed source, restricting to a subset of
    sources, writing the manifest with per-date outcomes, and
    forwarding `--history-days` and `--api-key` to the right runners
  - exercises the CLI render path for both the success and the failure
    exit codes
- `README.md`
  - new "Historical Backfill" section documenting the CLI invocation,
    idempotent resume behavior, manifest layout, expected runtime, disk
    footprint, and the Odds API history limitation

## Files Changed

- `README.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `src/mlb_props_stack/__init__.py`
- `src/mlb_props_stack/backfill.py`
- `src/mlb_props_stack/cli.py`
- `tests/test_backfill.py`

## Verification

Commands run successfully before merge:

```bash
uv sync --extra dev
uv run pytest
uv run python -m mlb_props_stack
uv run python -m mlb_props_stack backfill-historical --help
```

Observed results:

- full test suite passed: `121 passed` (up from `103` on the previous
  slice; the 18 additional tests are the parametrized backfill coverage
  in `tests/test_backfill.py`)
- `uv run python -m mlb_props_stack` rendered the runtime summary as
  before
- `backfill-historical --help` lists the documented arguments and
  references the three valid `--sources` values

## Recommended Next Issue

- Run the actual overnight 2024 + 2025 regular-season backfill against
  real APIs (out of scope for the code-only slice committed here):
  `uv run python -m mlb_props_stack backfill-historical --start-date 2024-03-28 --end-date 2024-09-29`
  followed by the same invocation across the 2025 regular season. The
  resume logic added in this slice means an interrupted run can be
  re-invoked verbatim and only the missing dates will be ingested.
- After both sweeps finish, run
  `uv run python -m mlb_props_stack check-data-alignment --start-date 2024-03-28 --end-date 2024-09-29 --threshold 0.95`
  (and the same for 2025) to confirm the issue's >=95% per-date feature
  and outcome coverage criterion.
- Then trigger a fresh
  `uv run python -m mlb_props_stack train-starter-strikeout-baseline --start-date 2024-03-28 --end-date 2024-09-29`
  to confirm `held_out_rows >= 100`, and a 14-day
  `uv run python -m mlb_props_stack build-walk-forward-backtest`
  inside the backfill window to confirm at least one scoreable row.
- Raw artifacts from those runs should ship via git-lfs or a release
  tarball rather than plain git, per the issue requirement.

Why this should go next:

- The CLI, resume logic, manifest, and tests for the backfill are now
  all in place — only the actual long-running ingest sweep is pending,
  and it is the precondition the rest of the stage-gate metrics
  (`held_out_rows >= 100`, `scoreable_backtest_rows >= 100`, settled
  paper-bet counts) depend on
- Until the sweep finishes, every downstream evaluation is still
  dominated by the four-date 2026-04-18 → 2026-04-21 window from the
  previous slice

## Constraints And Open Questions

- "Officialness" of dates is not enforced. `iter_backfill_dates` walks
  every calendar date in `[start, end]`. The MLB Stats API schedule
  endpoint returns an empty payload for off-days (no games), so the
  resulting `games.jsonl` will be a zero-row file and the date is still
  recorded as `ingested`. The Odds and Statcast sources will then fall
  through with no work to do for those dates. If a future issue wants
  to skip non-game days entirely, drive the iterator from the MLB
  schedule rather than the calendar.
- Resume completeness is judged on file existence, not row count. Empty
  off-day artifacts are therefore considered "complete" and resume will
  skip them on rerun. This matches what `check-data-alignment` reads,
  so the diagnostic stays consistent end-to-end. If a future feature
  requires nonzero-row guarantees, lift the row-count check from
  `data_alignment.collect_artifact_counts_for_date` into
  `is_source_complete`.
- Source-level failure handling captures `Exception` only.
  `KeyboardInterrupt` and `SystemExit` still propagate, so a Ctrl-C
  during an overnight run aborts immediately rather than silently
  retrying. The previous date's manifest is *not* written when a
  `BaseException` propagates mid-sweep — only completed sweeps emit a
  manifest. If supervisors need partial manifests, wrap the loop in a
  `try/finally` that persists progress on `BaseException` too.
- The Odds API ingest still expects a prior MLB metadata run for the
  same date to exist before it starts. The default source order in
  `ALL_SOURCES` (`mlb-metadata`, `odds-api`, `statcast-features`) keeps
  that invariant satisfied because each date is processed top-to-bottom
  before moving to the next date. If a user passes a custom `--sources`
  with `odds-api` before `mlb-metadata`, the odds runner will still
  fail with `FileNotFoundError` from `_load_latest_mlb_metadata_for_date`
  — that failure is captured per-date as `failed` rather than raised, so
  the sweep continues, but the user should keep the default order for
  full season runs.
- `BackfillResult.manifest_path` is the only on-disk artifact this
  slice adds. The actual raw and normalized data still flows through
  the existing ingest helpers, so all downstream consumers
  (`check-data-alignment`, `train-starter-strikeout-baseline`,
  `build-walk-forward-backtest`) need no changes to read backfilled
  dates.
