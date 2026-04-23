# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Last completed issue: `AGE-200` on branch
  `dylan/condescending-gagarin-7a53f8`
- This slice hardens normalized JSONL writes across every ingest adapter:
  `ingest/mlb_stats_api.py`, `ingest/odds_api.py`, and
  `ingest/statcast_features.py` now stream every `_write_jsonl` through a
  sibling `<name>.tmp` file and only `os.replace` it over the target path
  once the full record stream has been serialized successfully
- Current status: crashes or exceptions mid-write can no longer leave a
  truncated normalized JSONL on disk; the prior artifact (if any) is
  preserved unchanged, and the temp file is unlinked before the exception
  propagates

## What Was Completed In This Slice

- `src/mlb_props_stack/ingest/mlb_stats_api.py`
  - adds `import os`
  - rewrites `_write_jsonl` to write into `path.with_name(f"{path.name}.tmp")`,
    `os.replace` the temp file over the target path on success, and unlink
    the temp file on any `BaseException` (including KeyboardInterrupt) before
    re-raising
- `src/mlb_props_stack/ingest/odds_api.py`
  - rewrites `_write_jsonl` with the same atomic `.tmp` + `os.replace`
    pattern (the module already imported `os`)
- `src/mlb_props_stack/ingest/statcast_features.py`
  - adds `import os`
  - rewrites `_write_jsonl` with the same atomic `.tmp` + `os.replace`
    pattern so pull manifests, pitch-level base rows, and the three daily
    feature tables all publish atomically
- `tests/test_ingest_atomic_writes.py`
  - new parametrized test file that exercises `_write_jsonl` in all three
    ingest modules: happy path writes the target and removes the temp file,
    nested parents are created, and a `TypeError` mid-write leaves any
    prior file intact and removes the temp file (covering both the
    fresh-target and overwrite-existing cases)
- `README.md`
  - documents the atomic JSONL write guarantee for the three ingest
    adapters alongside the existing Statcast retry/backoff note

## Files Changed

- `README.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `src/mlb_props_stack/ingest/mlb_stats_api.py`
- `src/mlb_props_stack/ingest/odds_api.py`
- `src/mlb_props_stack/ingest/statcast_features.py`
- `tests/test_ingest_atomic_writes.py`

## Verification

Commands run successfully before merge:

```bash
uv sync --extra dev
uv run pytest
uv run python -m mlb_props_stack
```

Observed results:

- full test suite passed: `103 passed` (up from `91` on the previous slice;
  the 12 additional tests are the parametrized atomic-write coverage across
  all three ingest modules)
- `uv run python -m mlb_props_stack` rendered the runtime summary as before
- new tests cover happy-path writes, parent-directory creation, failure
  mid-write preserving a prior file, and first-write failures not leaving a
  target or a temp file behind

## Recommended Next Issue

- Backfill the saved historical ingest, feature, and odds snapshot set under
  `data/normalized/mlb_stats_api/`, `data/normalized/statcast_search/`, and
  `data/normalized/the_odds_api/` for `2026-04-18` through `2026-04-23`, then
  rerun `check-data-alignment` and `build-walk-forward-backtest` to confirm
  the diagnostic flips to green and the backtest window produces non-zero
  actionable rows

Why this should go next:

- `check-data-alignment` still flags the missing historical ingest/feature/
  odds artifacts for that window, and AGE-199 + AGE-200 only hardened how
  those artifacts are fetched and written — they did not actually pull any
  new snapshots
- the threaded fetch pool from AGE-199 and the atomic JSONL writes from
  AGE-200 together make a multi-date backfill safe to re-run: partial
  failures no longer corrupt a run directory, and retries stay bounded
- once the coverage report passes for that window, the natural follow-up is
  to gate `build-walk-forward-backtest` on `check-data-alignment --threshold`
  so all-skipped windows surface as a precondition failure instead of an
  opaque skip-rate

## Constraints And Open Questions

- Atomicity guarantee is scoped to the normalized JSONL artifacts inside
  the three ingest modules. Raw JSON snapshots (`_write_json`) and raw
  Statcast CSV pulls (`_write_text` in `statcast_features.py`) still rename
  in-place; if a future issue needs the same guarantee for those, copy the
  same `.tmp` + `os.replace` pattern. The same is true for the write sites
  in `backtest.py`, `modeling.py`, `edge.py`, and `paper_tracking.py`, which
  were intentionally out of scope for AGE-200
- `os.replace` is atomic on POSIX and same-volume on Windows. The temp file
  is always created as a sibling of the final path (`path.with_name(...)`),
  so the replace is guaranteed to stay on the same filesystem and remain
  atomic regardless of where the user points `output_dir`
- The helpers catch `BaseException` rather than `Exception` on purpose: a
  KeyboardInterrupt or SystemExit during serialization also needs to clean
  up the sibling `.tmp` file, and the exception is re-raised unchanged. If
  the process is `SIGKILL`ed between the final write and the `os.replace`,
  the target remains the prior version (which is the point of the change)
  but a stray `.tmp` can linger until the next successful write overwrites
  it. A follow-up could add a startup sweep if lingering temp files ever
  become a practical problem
- Three near-identical `_write_jsonl` helpers now live in the ingest
  package. Extracting a shared utility was deliberately avoided in this
  slice to keep the diff narrow and reviewable; consolidate only if a
  future ingest adapter lands and the duplication becomes a real
  maintenance burden
