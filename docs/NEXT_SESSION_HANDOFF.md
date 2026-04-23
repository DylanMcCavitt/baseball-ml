# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current issue branch: `feat/age-259-dashboard-wager-gates`
- Current issue: `AGE-259` - unify daily candidate approval with the
  dashboard wager gates
- Status: implementation and local verification are complete. A Streamlit
  preview is running at `http://localhost:8501` with
  `MLB_PROPS_STACK_DATA_DIR=/Users/dylanmccavitt/projects/nba-ml/data`

## What Changed In This Slice

- Added `src/mlb_props_stack/wager_approval.py` as the shared final wager-gate
  evaluator for daily sheets and dashboard rows
  - evaluates edge, hold, confidence, stake cap, pitcher status, model age,
    same-pitcher correlation, and daily exposure
  - annotates every scored row with explicit `wager_gate_details`,
    `wager_gate_notes`, `wager_gate_status`, `wager_blocked_reason`, and
    `wager_approved`
  - keeps the AGE-209 per-pitcher/per-game Kelly sizing work separate; this
    slice only unifies the final board approval gate so paper tracking cannot
    diverge from the dashboard
- Updated `src/mlb_props_stack/paper_tracking.py`
  - raw `evaluation_status=actionable` rows keep `actionable_rank`
  - `bet_placed=true` is now set only when `wager_approved=true`
  - approved rows get `approved_rank`
  - `paper_results.jsonl` is built only from final approved wagers while
    rejected scored rows remain auditable in `daily_candidates.jsonl`
- Updated `src/mlb_props_stack/dashboard/lib/data.py`
  - dashboard rows now use the same shared evaluator for `cleared` and gate
    notes instead of a second copy of the gate logic
  - pitcher PMF lookup now resolves `starter_strikeout_inference` run ids, so
    daily-candidate drilldowns no longer fall back to a baseline folder that
    may lack `ladder_probabilities.jsonl`
- Updated dashboard chart calls in `screens/backtest.py`, `screens/features.py`,
  and `screens/pitcher.py` from deprecated `use_container_width=True` to
  `width="stretch"`
- Updated CLI summary output to include `approved_wagers=...`
- Updated docs to distinguish raw actionable candidates from final approved
  wagers

## Files Changed

- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/architecture.md`
- `docs/modeling.md`
- `docs/review_runtime_checks.md`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/dashboard/lib/data.py`
- `src/mlb_props_stack/dashboard/screens/backtest.py`
- `src/mlb_props_stack/dashboard/screens/features.py`
- `src/mlb_props_stack/dashboard/screens/pitcher.py`
- `src/mlb_props_stack/paper_tracking.py`
- `src/mlb_props_stack/wager_approval.py`
- `tests/test_cli.py`
- `tests/test_dashboard_data.py`
- `tests/test_paper_tracking.py`

## Verification

Commands run successfully:

```bash
uv sync --extra dev
uv run pytest tests/test_dashboard_data.py tests/test_paper_tracking.py tests/test_edge.py tests/test_cli.py
uv run pytest tests/test_runtime_smokes.py
uv run pytest
python3 -m compileall src tests
uv run python -m mlb_props_stack
uv run python -m mlb_props_stack build-daily-candidates --date 2026-04-23 --output-dir /Users/dylanmccavitt/projects/nba-ml/data
```

Observed results:

- focused dashboard/paper/edge/CLI suite passed: `27 passed`
- runtime smoke suite passed
- full test suite passed: `181 passed`, with the existing third-party MLflow /
  Pydantic deprecation warnings
- `python -m mlb_props_stack` printed the runtime configuration banner
- live-artifact daily workflow wrote run `20260423T202309Z` against the
  canonical local data directory:
  - CLI summary: `scored_candidates=75`, `actionable_candidates=53`,
    `approved_wagers=0`, `settled_paper_results=0`, `pending_paper_results=0`
  - `edge_candidates.jsonl`: 97 rows; 53 raw actionable, 22 below-threshold, 8
    missing line probability, 11 missing projection, 3 invalid projection
  - `daily_candidates.jsonl`: 75 scored rows; 53 raw actionable, 22
    below-threshold, `wager_approved=0`, `bet_placed=0`
  - `paper_results.jsonl`: 0 rows for 2026-04-23
  - block reasons were `hold above max` for 54 rows, `below confidence floor`
    for 7 rows, and `below edge threshold` for 14 rows
- Browser preview at `http://localhost:8501`:
  - board selected `2026-04-23`
  - `75 props`, `plays cleared=0`, `total stake=0.00u`
  - rejected board rows show explicit notes such as `hold above max` and
    `correlated same-slate play`
  - pitcher, backtest, features, and config screens opened without Streamlit
    tracebacks after the PMF lookup fix
  - chart deprecation warnings disappeared after the `width="stretch"` cleanup

Command run with expected non-zero outcome:

```bash
uv run python -m mlb_props_stack check-data-alignment --start-date 2026-04-23 --end-date 2026-04-23 --output-dir /Users/dylanmccavitt/projects/nba-ml/data
```

Observed result:

- exited `1` because `2026-04-23` has no settled outcomes yet (`out_cov=n/a`);
  the report still found the saved slate artifacts: 9 games, 97 prop lines, 18
  pitcher feature rows, 18 lineup feature rows, 18 context feature rows, and
  72.2% odds coverage

## Recommended Next Issue

- Build the dedicated approved-wager card command/report from the AGE-259 issue
  packet, so terminal output shows the exact same final surface the dashboard
  uses before any sportsbook action
- Keep AGE-209 as the separate Kelly sizing allocator issue for per-pitcher and
  per-game exposure caps; do not fold that into the dashboard-only approval gate

## Constraints And Notes

- The issue worktree does not contain the ignored 2026-04-23 live artifacts, so
  live verification used `--output-dir /Users/dylanmccavitt/projects/nba-ml/data`
  and the dashboard preview uses `MLB_PROPS_STACK_DATA_DIR` pointed at that same
  canonical local data directory
- The Streamlit preview process is intentionally left running for Codex browser
  use on `http://localhost:8501`
