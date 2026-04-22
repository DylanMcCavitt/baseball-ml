# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Active issue branch: `feat/age-155-live-and-expansion-gates`
- This slice defines the numeric go or no-go gates for live-usage discussion
  and next-market expansion
- Current status under the new gate doc is still `research_only`

## What Was Completed In This Slice

- `docs/stage_gates.md`
  - adds the new source-of-truth checklist for:
    - live-use discussion gates
    - next-market expansion gates
    - status resolution between `research_only`,
      `eligible_for_live_discussion`, and
      `eligible_for_next_market_expansion`
  - defines exact artifact-to-metric mappings for held-out rows, backtest
    coverage, paper sample size, CLV, and ROI
  - applies the checklist to one real saved artifact set and records the
    current repo status as `research_only`
- `docs/modeling.md`
  - replaces the older qualitative promotion language with a pointer to the new
    numeric stage-gates document
- `docs/architecture.md`
  - makes `docs/stage_gates.md` the explicit promotion-status source of truth
- `README.md`
  - adds `docs/stage_gates.md` to the repo docs map

## Files Changed

- `README.md`
- `docs/architecture.md`
- `docs/modeling.md`
- `docs/stage_gates.md`
- `docs/NEXT_SESSION_HANDOFF.md`

## Verification

Commands run successfully on April 22, 2026:

```bash
uv sync --extra dev
uv run pytest
uv run python -m mlb_props_stack
python3 - <<'PY'
from __future__ import annotations
import json
from pathlib import Path

repo = Path('/Users/dylanmccavitt/projects/nba-ml')
eval_path = repo / 'data/normalized/starter_strikeout_baseline/start=2026-04-18_end=2026-04-23/run=20260422T205727Z/evaluation_summary.json'
backtest_path = repo / 'data/normalized/walk_forward_backtest/start=2026-04-23_end=2026-04-23/run=20260422T205734Z/backtest_runs.jsonl'
paper_path = repo / 'data/normalized/paper_results/date=2026-04-23/run=20260422T190633Z/paper_results.jsonl'

summary = json.loads(eval_path.read_text())
backtest = json.loads(backtest_path.read_text().splitlines()[-1])
paper_rows = [json.loads(line) for line in paper_path.read_text().splitlines() if line.strip()]

print('held_out_rows=', summary['row_counts']['held_out'])
print('calibrated_ece=', summary['held_out_probability_calibration']['calibrated']['expected_calibration_error'])
print('scoreable_backtest_rows=', backtest['row_counts']['actionable'] + backtest['row_counts']['below_threshold'])
print('backtest_placed_bets=', backtest['bet_outcomes']['placed_bets'])
print('backtest_skip_rate=', backtest['row_counts']['skipped'] / backtest['row_counts']['snapshot_groups'])
print('settled_paper_bets=', len([row for row in paper_rows if row.get('settlement_status') in {'win', 'loss', 'push'}]))
PY
```

Observed results:

- `uv run pytest`
  - `68 passed`
- `uv run python -m mlb_props_stack`
  - printed the runtime summary successfully
- artifact verification script
  - `held_out_rows=48`
  - `calibrated_ece=0.02285`
  - `scoreable_backtest_rows=0`
  - `backtest_placed_bets=0`
  - `backtest_skip_rate=1.0`
  - `settled_paper_bets=0`

Interpretation:

- the current saved artifact set is unambiguously `research_only`
- held-out model metrics are not enough because the latest saved backtest has
  zero scoreable bets and the latest saved paper-tracking run has zero settled
  bets

## Recommended Next Issue

- Fix the current-market operational blocker so the repo can produce non-empty
  `daily_candidates`, non-empty `paper_results`, and a walk-forward backtest
  with scoreable exact-line rows

Why this should go next:

- the new stage-gate doc makes the blocker explicit: live-use and expansion
  gates cannot pass until the pitcher strikeout market produces real paper and
  backtest samples
- the latest saved backtest still reports `139` skipped rows and `0` scoreable
  rows, so another readiness discussion before the ingest or join path is fixed
  would just repeat the same `research_only` outcome

## Constraints And Open Questions

- The worked example in `docs/stage_gates.md` was verified against the
  canonical local checkout under `/Users/dylanmccavitt/projects/nba-ml/data/`
  because this issue worktree does not carry the full generated artifact tree
- The latest saved paper-results run `20260422T190633Z` is empty, and the
  earlier `20260422T173038Z` paper-results run is also empty
- The latest saved walk-forward backtest run `20260422T205734Z` still has
  `snapshot_groups=139`, `actionable_bets=0`, and `skipped=139`
- There is still no automated CLI for the stage-gate evaluation; the thresholds
  are documented clearly now, but applying them is still a manual artifact
  review step
