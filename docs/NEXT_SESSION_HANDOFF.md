# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current issue branch: `feat/age-265-dashboard-board-grouping`
- Last completed issue in this worktree: `AGE-265` - add sportsbook provenance
  and grouped pitcher view to the dashboard board
- Status: implementation and local verification are complete; PR publication is
  the next closeout step
- Base branch at issue start: `main` / `origin/main` at `348aba6`

## What Changed In This Slice

- Dashboard board rows now expose sportsbook provenance from joined
  `prop_line_snapshots.jsonl` metadata:
  - sportsbook title and key
  - source event ID
  - compact line snapshot label
  - market last-update and line capture timestamps
  - a combined provenance string for audit surfaces
- Added `group_board_by_pitcher()` to collapse visible board rows to the current
  top-ranked row per pitcher while preserving:
  - total line-row count
  - hidden row count
  - distinct sportsbook count
  - compact book/side/line summary for hidden rows
- The board screen now has a row-mode control:
  - `All line rows` remains the default and shows every sportsbook/alternate
    line row.
  - `Grouped by pitcher` shows one row per pitcher plus hidden book/line
    counts and summaries.
- The board table now includes `SOURCE` and `GROUP` columns so repeated pitchers
  read as sportsbook/line-level rows rather than accidental duplicates.
- The pitcher detail selector continues to use the visible row set and keeps
  query-param navigation intact.
- Updated dashboard/runtime tests for the new row-mode control, sportsbook
  provenance rendering, and grouped-pitcher summaries.
- Updated `docs/review_runtime_checks.md` so future board changes require
  browser verification of provenance, grouped mode, hidden-row counts, and
  detail navigation.

## Files Changed

- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/review_runtime_checks.md`
- `src/mlb_props_stack/dashboard/lib/data.py`
- `src/mlb_props_stack/dashboard/screens/board.py`
- `tests/test_dashboard_app.py`
- `tests/test_dashboard_data.py`
- `tests/test_paper_tracking.py`
- `tests/test_runtime_smokes.py`

## Verification

Commands run successfully:

```bash
uv sync --extra dev
uv run pytest tests/test_dashboard_data.py tests/test_dashboard_app.py
uv run pytest tests/test_wager_card.py tests/test_cli.py
uv run pytest tests/test_runtime_smokes.py
uv run pytest tests/test_dashboard_data.py tests/test_dashboard_app.py tests/test_runtime_smokes.py tests/test_paper_tracking.py
uv run pytest
python3 -m compileall src tests
uv run python -m mlb_props_stack
MLB_PROPS_STACK_DATA_DIR=/Users/dylanmccavitt/projects/nba-ml/data uv run streamlit run src/mlb_props_stack/dashboard/app.py --server.port 8502 --server.headless true
```

Observed results:

- focused dashboard suite passed: `9 passed`
- required wager-card/CLI suite passed: `13 passed`
- runtime smoke suite passed: `4 passed`
- broader dashboard/runtime/paper suite passed: `16 passed`
- full test suite passed: `194 passed`
- `python3 -m compileall src tests` completed successfully
- `python -m mlb_props_stack` printed the runtime configuration banner
- A direct canonical-data helper check loaded:
  - source: `daily_candidates`
  - line rows: `75`
  - unique pitchers: `9`
  - grouped rows: `9`
  - Jacob deGrom: `1 of 9`, `+8 hidden`, `7 books`
  - Davis Martin: `1 of 8`, `+7 hidden`, `6 books`
  - approved wagers remained `0`
- Browser verification against `http://localhost:8502/?screen=board&board_date=2026-04-23`:
  - `Show rejected` surfaces line-level rows with sportsbook/event/snapshot
    provenance.
  - `Grouped by pitcher` shows `9` visible rows from `75` book/line rows.
  - Jacob deGrom and Davis Martin collapse to one visible row each with hidden
    row counts and book/line summaries.
  - `Open pitcher` navigates to
    `?screen=pitcher&board_date=2026-04-23&pitcher_id=mlb-pitcher%3A594798`.
  - A layout issue where row-mode labels wrapped vertically was fixed by giving
    the mode control its own wider row area and moving search below it.

The test runs still show the existing third-party MLflow/Pydantic deprecation
warnings.

## Recommended Issue Order

Work the next issues in this order so the dashboard and model evidence remain
coherent:

1. `AGE-266` - show active and excluded optional feature diagnostics in the
   dashboard
   - Makes the Feature Inspection screen answer whether optional features are
     active, missing from artifacts, or excluded by coverage/variance gates.
2. `AGE-267` - regenerate historical optional-feature artifacts with
   timestamp-valid coverage
   - Backfills/regenerates weather, umpire, park, handedness split, and lineup
     aggregate artifacts so optional columns can actually reach training rows.
3. `AGE-268` - train and compare expanded-feature baseline against core-only
   model
   - Runs only after `AGE-267` proves optional feature coverage exists.
   - Compares the expanded model against the core model with walk-forward,
     calibration, CLV, ROI, edge-bucket, and approval-gate evidence.
4. `AGE-262` - run the first approved-wager evidence refresh and readiness
   report
   - Use `evaluate-stage-gates` as the readiness report command.
   - Still blocked behind `AGE-268` so it does not refresh readiness against a
     known core-only/empty-optional-feature model.
5. `AGE-263` - close the sample-size gap for live-use stage gates
   - Broader evidence collection after the first coherent refresh.

Related but not on the critical path:

- `AGE-209` - add per-pitcher and per-game exposure caps to Kelly sizing
  - Keep queued unless grouped board work shows exposure/correlation sizing is
    still materially distorting the wager card before `AGE-262`.

## Constraints And Notes

- Use the canonical ignored data directory for live verification:
  `/Users/dylanmccavitt/projects/nba-ml/data`
- `evaluate-stage-gates` is a report/audit surface only. It does not lower
  thresholds, place bets, retrain models, add features, or change wagering
  decisions.
- The latest coherent canonical artifact set is still `research_only`.
- The current blockers are evidence depth, not command plumbing:
  `held_out_rows=48`, `scoreable_backtest_rows=0`, and `settled_paper_bets=0`.
- If a workflow needs failure semantics, add `--fail-on-research-only`; otherwise
  the command exits `0` even when the honest status remains `research_only`.
- This slice did not change wager approval gates, pricing, devig, Kelly sizing,
  paper tracking, candidate generation, or stage-gate thresholds.
- Keep using the 2026-04-23 canonical board artifact when manually checking the
  duplicate-looking row problem:
  `data/normalized/daily_candidates/date=2026-04-23/run=20260423T210236Z`.
