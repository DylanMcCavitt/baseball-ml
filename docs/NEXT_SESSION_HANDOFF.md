# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current issue branch: none; `AGE-260` has been merged to `main`
- Last completed issue: `AGE-260` - add an approved wager card CLI/report for
  the live slate
- Status: implementation, local verification, saved-slate runtime checks,
  Codex browser dashboard checks, PR creation, PR merge, and canonical local
  `main` sync are complete
- Merged PR: <https://github.com/DylanMcCavitt/baseball-ml/pull/35>
- Merge commit on `main`: `4f0eb76`
- Streamlit preview for this branch is running at `http://localhost:8502` with
  `MLB_PROPS_STACK_DATA_DIR=/Users/dylanmccavitt/projects/nba-ml/data`

## What Changed In This Slice

- Added `src/mlb_props_stack/wager_card.py`
  - loads the latest `daily_candidates` artifact for a target date, or the
    latest saved date when `--date` is omitted
  - builds a terminal-first approved wager card from AGE-259 final
    `wager_approved` fields
  - writes auditable artifacts under
    `data/normalized/wager_card/date=<iso>/run=<ts>/`
  - writes `wager_card.jsonl` plus `wager_card_metadata.json`
  - supports blocked-candidate diagnostics through `include_rejected`
- Added the CLI command:

```bash
uv run python -m mlb_props_stack build-wager-card --date 2026-04-23 --output-dir /Users/dylanmccavitt/projects/nba-ml/data
```

- Added `--include-rejected` for diagnostics:

```bash
uv run python -m mlb_props_stack build-wager-card --date 2026-04-23 --output-dir /Users/dylanmccavitt/projects/nba-ml/data --include-rejected
```

- Added `tests/test_wager_card.py`
  - proves blocked rows are excluded from the default card
  - proves approved rows are included
  - proves blocked diagnostics are available with `--include-rejected`
  - proves the command exits cleanly and prints an honest empty card when no
    wagers are approved
  - proves the wager-card approved count matches the dashboard board count for
    the seeded fixture
- Updated docs:
  - `README.md` now documents `build-wager-card` and the new artifact paths
  - `docs/review_runtime_checks.md` now lists `build-wager-card` and requires
    count parity with the dashboard `plays cleared` metric
- Fixed the remaining dashboard navigation bug found in the Codex browser:
  - raw HTML pitcher/back anchors in the Streamlit markdown surface did not
    navigate when clicked
  - board pitcher names are now dense display text
  - board-to-pitcher navigation now uses a Streamlit-native `Pitcher detail`
    selector plus `Open pitcher` button
  - pitcher-to-board navigation now uses a Streamlit-native `← board` button
  - verified in the Codex browser that header nav, board-to-pitcher, and
    pitcher-to-board navigation update query params correctly

## Files Changed

- `README.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/review_runtime_checks.md`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/dashboard/lib/navigation.py`
- `src/mlb_props_stack/dashboard/screens/board.py`
- `src/mlb_props_stack/dashboard/screens/pitcher.py`
- `src/mlb_props_stack/wager_card.py`
- `tests/test_wager_card.py`

## Verification

Commands run successfully:

```bash
uv sync --extra dev
uv run pytest tests/test_wager_card.py tests/test_dashboard_app.py tests/test_dashboard_data.py tests/test_cli.py
uv run pytest tests/test_paper_tracking.py tests/test_dashboard_data.py tests/test_cli.py tests/test_wager_card.py
uv run pytest tests/test_runtime_smokes.py
uv run pytest
python3 -m compileall src tests
uv run python -m mlb_props_stack
uv run python -m mlb_props_stack build-daily-candidates --date 2026-04-23 --output-dir /Users/dylanmccavitt/projects/nba-ml/data
uv run python -m mlb_props_stack build-wager-card --date 2026-04-23 --output-dir /Users/dylanmccavitt/projects/nba-ml/data
uv run python -m mlb_props_stack build-wager-card --date 2026-04-23 --output-dir /Users/dylanmccavitt/projects/nba-ml/data --include-rejected
```

Observed results:

- focused wager-card/dashboard/CLI suite passed: `18 passed`
- focused paper/dashboard/CLI/wager-card suite passed: `19 passed`
- runtime smoke suite passed: `3 passed`, with the existing third-party MLflow /
  Pydantic deprecation warnings
- full test suite passed: `187 passed`, with the existing third-party MLflow /
  Pydantic deprecation warnings
- `python -m mlb_props_stack` printed the runtime configuration banner
- `python3 -m compileall src tests` completed successfully
- live-artifact daily workflow wrote run `20260423T210236Z` against the
  canonical local data directory:
  - `scored_candidates=75`
  - `actionable_candidates=53`
  - `approved_wagers=0`
  - `settled_paper_results=0`
  - `pending_paper_results=0`
- approved-only wager card wrote run `20260423T210402Z`:
  - `total_candidates=75`
  - `approved_wagers=0`
  - `blocked_candidates=75`
  - `included_rows=0`
  - `wager_card.jsonl` contained 0 rows
  - `wager_card_metadata.json` recorded the source daily-candidate run
    `20260423T210236Z`
- diagnostic wager card with `--include-rejected` wrote run
  `20260423T210408Z`:
  - `included_rows=75`
  - blocked section showed explicit notes such as `hold above max` and
    `correlated same-slate play`
- daily candidate gate counts for run `20260423T210236Z`:
  - `wager_approved=0`
  - `bet_placed=0`
  - blocked reasons: `hold above max` 54, `below edge threshold` 14,
    `below confidence floor` 7
- Browser preview at `http://localhost:8502`:
  - board selected `2026-04-23`
  - `75 props`, `plays cleared=0`, `total stake=0.00u`
  - header nav buttons opened pitcher, backtest, MLflow, features, config, and
    board with query params updating correctly
  - raw HTML board/pitcher links were confirmed buggy before the patch
  - native `Open pitcher` and `← board` controls were verified after the patch
  - dashboard `plays cleared=0` matches the latest approved wager card count
- Follow-up dashboard/model investigation after merge:
  - the repeated Jacob deGrom and Davis Martin rows are mostly not true
    duplicate artifacts; the board is line/book-level and currently hides
    sportsbook provenance
  - with `Show rejected` enabled, the top rows are dominated by multiple
    sportsbooks and alternate lines for the same pitchers
  - active inference run
    `data/normalized/starter_strikeout_inference/date=2026-04-23/run=20260423T210236Z`
    uses source model run `20260422T202712Z`
  - the active model uses only 11 core encoded features:
    `pitch_sample_size`, `plate_appearance_sample_size`, `pitcher_k_rate`,
    `swinging_strike_rate`, `csw_rate`, `recent_batters_faced`,
    `recent_pitch_count`, `rest_days`, `last_start_batters_faced`,
    `expected_leash_batters_faced`, and `lineup_is_confirmed`
  - optional split, lineup, park, weather, and umpire features are supported in
    code but are not active in the current model because the source training
    rows had zero populated optional-feature coverage
  - target-date weather and umpire artifacts were missing for `2026-04-23`;
    `check-data-alignment` showed `wx_cov=0.0%` and `ump_cov=0.0%`

## Recommended Issue Order

Work the next issues in this order so the dashboard and model evidence remain
coherent:

1. `AGE-261` - add a stage-gate evaluator CLI for live-use readiness
   - Independent and useful immediately.
   - Produces the executable readiness report that later evidence issues should
     consume.
2. `AGE-265` - add sportsbook provenance and grouped pitcher view to the
   dashboard board
   - Fixes the deGrom/Davis Martin duplicate-looking board rows.
   - Makes line/book-level candidates clear without deleting auditable rows.
3. `AGE-266` - show active and excluded optional feature diagnostics in the
   dashboard
   - Makes the Feature Inspection screen answer whether optional features are
     active, missing from artifacts, or excluded by coverage/variance gates.
4. `AGE-267` - regenerate historical optional-feature artifacts with
   timestamp-valid coverage
   - Backfills/regenerates weather, umpire, park, handedness split, and lineup
     aggregate artifacts so optional columns can actually reach training rows.
5. `AGE-268` - train and compare expanded-feature baseline against core-only
   model
   - Runs only after `AGE-267` proves optional feature coverage exists.
   - Compares the expanded model against the core model with walk-forward,
     calibration, CLV, ROI, edge-bucket, and approval-gate evidence.
6. `AGE-262` - run the first approved-wager evidence refresh and readiness
   report
   - Now blocked behind `AGE-268` so it does not refresh readiness against a
     known core-only/empty-optional-feature model.
7. `AGE-263` - close the sample-size gap for live-use stage gates
   - Broader evidence collection after the first coherent refresh.

Related but not on the critical path:

- `AGE-209` - add per-pitcher and per-game exposure caps to Kelly sizing
  - Keep queued unless grouped board work shows exposure/correlation sizing is
    still materially distorting the wager card before `AGE-262`.

## Constraints And Notes

- Use the canonical ignored data directory for live verification:
  `/Users/dylanmccavitt/projects/nba-ml/data`
- The saved 2026-04-23 slate still has no approved wagers. That is expected
  because final wager gates block every scored candidate.
- The wager card is a report/audit surface only. It does not place bets,
  loosen stage gates, retrain models, or change odds ingest behavior.
- If a new thread needs a quick confidence check, run:

```bash
uv run python -m mlb_props_stack build-wager-card --date 2026-04-23 --output-dir /Users/dylanmccavitt/projects/nba-ml/data
```
