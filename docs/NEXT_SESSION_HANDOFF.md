# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current issue branch: `feat/age-261-stage-gate-evaluator`
- Current issue: `AGE-261` - add a stage-gate evaluator CLI for live-use
  readiness
- Status: implementation and local verification are complete; open/merge the PR
  from this branch after review and green CI
- Last completed mainline issue before this branch: `AGE-260` - add an approved
  wager card CLI/report for the live slate

## What Changed In This Slice

- Added `src/mlb_props_stack/stage_gates.py`
  - discovers the latest coherent training, backtest, CLV, ROI, and paper
    result artifacts
  - resolves status as `research_only`, `eligible_for_live_discussion`, or
    `eligible_for_next_market_expansion`
  - computes the metric definitions from `docs/stage_gates.md`, including:
    `scoreable_backtest_rows`, `backtest_skip_rate`,
    `paper_same_line_clv_sample`, paper ROI, and backtest ROI
  - writes `stage_gate_report.json` and `stage_gate_report.md` under
    `data/normalized/stage_gates/run=<timestamp>/`
  - treats missing or empty artifact evidence as a failing gate while preserving
    warnings and artifact paths for audit
- Added the CLI command:

```bash
uv run python -m mlb_props_stack evaluate-stage-gates
```

- Added `--fail-on-research-only` so automation can opt into a nonzero exit
  when the evaluated status is still `research_only`; default runs remain
  informational.
- Added fixture-backed tests for both:
  - a current-like failing `research_only` artifact set
  - a synthetic full-pass set that reaches
    `eligible_for_next_market_expansion`
- Updated `docs/stage_gates.md` to point at the executable evaluator and report
  paths.
- Updated `docs/review_runtime_checks.md` so readiness review uses
  `evaluate-stage-gates`.

## Files Changed

- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/review_runtime_checks.md`
- `docs/stage_gates.md`
- `src/mlb_props_stack/__init__.py`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/stage_gates.py`
- `tests/stage_gate_fixtures.py`
- `tests/test_cli.py`
- `tests/test_runtime_smokes.py`
- `tests/test_stage_gates.py`

## Verification

Commands run successfully:

```bash
uv sync --extra dev
uv run pytest tests/test_stage_gates.py tests/test_cli.py tests/test_runtime_smokes.py
uv run pytest tests/test_cli.py tests/test_runtime_smokes.py
uv run pytest
python3 -m compileall src tests
uv run python -m mlb_props_stack
uv run python -m mlb_props_stack evaluate-stage-gates
uv run python -m mlb_props_stack evaluate-stage-gates --output-dir /Users/dylanmccavitt/projects/nba-ml/data
```

Observed results:

- focused stage-gate/CLI/runtime suite passed: `15 passed`
- required CLI/runtime suite passed: `13 passed`
- full test suite passed: `191 passed`
- `python3 -m compileall src tests` completed successfully
- `python -m mlb_props_stack` printed the runtime configuration banner
- The worktree-local `evaluate-stage-gates` run resolved to `research_only`:
  - `held_out_rows=48`
  - `scoreable_backtest_rows=0`
  - `backtest_skip_rate=1`
  - `settled_paper_bets=0`
  - `paper_same_line_clv_sample=0`
  - `paper_roi=n/a`
  - `backtest_roi=n/a`
  - the worktree-local paper artifact was missing, so the report included a
    `missing paper_results.jsonl artifact` warning
- The canonical data-dir run also resolved to `research_only` and found the
  latest paper artifact:
  - report:
    `/Users/dylanmccavitt/projects/nba-ml/data/normalized/stage_gates/run=20260423T215618Z/stage_gate_report.json`
  - markdown report:
    `/Users/dylanmccavitt/projects/nba-ml/data/normalized/stage_gates/run=20260423T215618Z/stage_gate_report.md`
  - training run: `20260422T205727Z`
  - backtest run: `20260422T205734Z`
  - paper results run: `20260423T210236Z`
  - `held_out_rows=48`
  - `scoreable_backtest_rows=0`
  - `backtest_placed_bets=0`
  - `settled_paper_bets=0`

The test runs still show the existing third-party MLflow/Pydantic deprecation
warnings.

## Recommended Issue Order

Work the next issues in this order so the dashboard and model evidence remain
coherent:

1. `AGE-265` - add sportsbook provenance and grouped pitcher view to the
   dashboard board
   - Fixes the deGrom/Davis Martin duplicate-looking board rows.
   - Makes line/book-level candidates clear without deleting auditable rows.
2. `AGE-266` - show active and excluded optional feature diagnostics in the
   dashboard
   - Makes the Feature Inspection screen answer whether optional features are
     active, missing from artifacts, or excluded by coverage/variance gates.
3. `AGE-267` - regenerate historical optional-feature artifacts with
   timestamp-valid coverage
   - Backfills/regenerates weather, umpire, park, handedness split, and lineup
     aggregate artifacts so optional columns can actually reach training rows.
4. `AGE-268` - train and compare expanded-feature baseline against core-only
   model
   - Runs only after `AGE-267` proves optional feature coverage exists.
   - Compares the expanded model against the core model with walk-forward,
     calibration, CLV, ROI, edge-bucket, and approval-gate evidence.
5. `AGE-262` - run the first approved-wager evidence refresh and readiness
   report
   - Use `evaluate-stage-gates` as the readiness report command.
   - Still blocked behind `AGE-268` so it does not refresh readiness against a
     known core-only/empty-optional-feature model.
6. `AGE-263` - close the sample-size gap for live-use stage gates
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
