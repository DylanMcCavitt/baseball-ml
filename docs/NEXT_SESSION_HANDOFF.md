# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Last completed issue: `AGE-199` on branch
  `dylan/unruffled-albattani-af303f`
- This slice hardens Statcast CSV ingestion: `StatcastSearchClient.fetch_csv`
  now retries transient failures with bounded exponential backoff, and
  `ingest_statcast_features_for_date` fans Statcast pulls out across a small
  thread pool instead of serialising one HTTP request per player
- Current status: the client retries 429/5xx/URLError/TimeoutError while
  passing 4xx through immediately, and the ingest reliably completes all
  pulls in pull-order-deterministic parallel fetches

## What Was Completed In This Slice

- `src/mlb_props_stack/ingest/statcast_features.py`
  - adds `DEFAULT_MAX_FETCH_ATTEMPTS`, `DEFAULT_INITIAL_BACKOFF_SECONDS`,
    `DEFAULT_BACKOFF_MULTIPLIER`, `DEFAULT_MAX_BACKOFF_SECONDS`, and
    `DEFAULT_MAX_FETCH_WORKERS` module constants so retry/parallelism
    defaults are explicit and tunable
  - rewrites `StatcastSearchClient` with retry + exponential backoff around
    `urlopen`; accepts `max_attempts`, `initial_backoff_seconds`,
    `backoff_multiplier`, `max_backoff_seconds`, and an injectable `sleep`
    callable; only retries 429/5xx `HTTPError` plus `URLError` / `TimeoutError`
  - adds `_is_retriable_http_error` and `_fetch_csv_texts_concurrently`
    helpers; ingest pre-computes pull specs serially (keeps the `now()` test
    seam deterministic) and then dispatches CSV fetches through a bounded
    `ThreadPoolExecutor` while writes, dedup, and record ordering remain
    serial
  - `ingest_statcast_features_for_date` gains `max_fetch_workers`
    (defaulting to `DEFAULT_MAX_FETCH_WORKERS = 4`) and validates it
- `src/mlb_props_stack/ingest/__init__.py`
  - re-exports `DEFAULT_MAX_FETCH_WORKERS` alongside the existing ingest
    surface
- `README.md`
  - notes the retry-with-backoff and threaded fetch pool behaviour of the
    Statcast feature ingest
- `tests/test_statcast_feature_ingest.py`
  - adds retry-success, retry-exhausted, 4xx-no-retry, and
    config-validation coverage for `StatcastSearchClient`
  - adds a `_ConcurrencyProbeClient` that blocks until at least two fetches
    overlap to prove the thread pool actually parallelises work
  - adds a pull-ordering assertion that confirms manifest rows stay in
    pitcher-then-batter sorted order and captured-at timestamps stay
    monotonic even under threaded fetches
  - asserts that `max_fetch_workers=0` is rejected

## Files Changed

- `README.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `src/mlb_props_stack/ingest/__init__.py`
- `src/mlb_props_stack/ingest/statcast_features.py`
- `tests/test_statcast_feature_ingest.py`

## Verification

Commands run successfully before merge:

```bash
uv sync --extra dev
uv run pytest
uv run python -m mlb_props_stack
```

Observed results:

- full test suite passed: `91 passed` (up from `84` on the previous slice)
- `uv run python -m mlb_props_stack` rendered the runtime summary as before
- new tests cover both the retry loop and the parallel fetch pool behaviour

## Recommended Next Issue

- Backfill the saved historical ingest, feature, and odds snapshot set under
  `data/normalized/mlb_stats_api/`, `data/normalized/statcast_search/`, and
  `data/normalized/the_odds_api/` for `2026-04-18` through `2026-04-23`, then
  rerun `check-data-alignment` and `build-walk-forward-backtest` to confirm
  the diagnostic flips to green and the backtest window produces non-zero
  actionable rows

Why this should go next:

- `check-data-alignment` still flags the missing historical ingest/feature/
  odds artifacts for that window, and this slice only improved how the
  ingest fetches data — it did not actually pull any new snapshots
- the threaded fetch pool is what unblocks doing the backfill in a
  reasonable wall-clock budget, but the backfill itself is still pending
- once the coverage report passes for that window, the natural follow-up is
  to gate `build-walk-forward-backtest` on `check-data-alignment --threshold`
  so all-skipped windows surface as a precondition failure instead of an
  opaque skip-rate

## Constraints And Open Questions

- `StatcastSearchClient` retries only on 429, 5xx, `URLError`, and
  `TimeoutError`. Other `HTTPError` codes (e.g. 400, 401, 403, 404)
  short-circuit immediately so bad requests are not amplified into retry
  storms. If a future Baseball Savant behavioural change requires retrying
  other status codes, revisit `_is_retriable_http_error`
- The thread pool is CPU-light — each worker spends almost all of its time
  blocked on I/O — so `DEFAULT_MAX_FETCH_WORKERS=4` is deliberately modest
  to keep us well under any Baseball Savant soft rate limits. Raise
  `max_fetch_workers` only after confirming the endpoint tolerates it, and
  consider wiring the override through the CLI if the ingest ever needs to
  burst harder in a specific slice
- Pull timestamps (`captured_at`) are still generated serially in the main
  thread before dispatch so the existing `now()` / deterministic test
  seam keeps working. If the real-world gap between dispatch and completion
  ever matters (e.g. to compute latency per pull), add a second completion
  timestamp rather than threading the `now()` closure into workers
- `modeling._fetch_starter_outcome` still calls `client.fetch_csv` one row
  at a time. It already benefits from retry/backoff via the shared client,
  but it has not been parallelised yet. Add a threaded fan-out there only if
  a future training slice actually needs the throughput
