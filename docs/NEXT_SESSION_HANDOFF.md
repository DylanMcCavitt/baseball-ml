# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Active issue branch: `feat/offline-eval-report`
- `main` already includes the merged `AGE-189` runtime-smoke coverage work
- This branch contains the offline evaluation-report slice and is not merged yet

## What Was Completed In This Slice

- `src/mlb_props_stack/modeling.py`
  - adds `evaluation_summary.json` and `evaluation_summary.md` to each
    `train-starter-strikeout-baseline` run
  - summarizes held-out benchmark-vs-model metrics, held-out calibration
    metrics, top feature importance, and same-window previous-run deltas
  - carries the new summary paths plus headline held-out metrics in
    `StarterStrikeoutBaselineTrainingResult`
- `src/mlb_props_stack/cli.py`
  - prints the held-out model and benchmark RMSE / MAE directly after training
  - prints the previous run ID when a prior run exists for the same date window
  - prints the new evaluation summary artifact paths
- `tests/test_modeling.py`
  - verifies the summary JSON and markdown files are written
  - verifies previous-run comparison is populated and marks improved held-out
    RMSE / MAE deltas correctly
- `tests/test_cli.py`
  - locks the new training-summary output lines
- `README.md`
  - documents the new summary artifacts in the training section
- `docs/modeling.md`
  - documents that training now emits a readable offline report plus previous
    run comparison
- `docs/review_runtime_checks.md`
  - adds `evaluation_summary.json` and `evaluation_summary.md` to the required
    artifact inspection list for training work

## Files Changed

- `README.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/modeling.md`
- `docs/review_runtime_checks.md`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/modeling.py`
- `tests/test_cli.py`
- `tests/test_modeling.py`

## Verification

Commands run successfully on April 22, 2026:

```bash
uv sync --extra dev
uv run pytest tests/test_cli.py tests/test_modeling.py
uv run pytest
uv run python -m compileall src tests
uv run python -m mlb_props_stack
```

Observed results:

- focused pytest run:
  - `12 passed`
- full pytest run:
  - `65 passed`
- `uv run python -m compileall src tests`
  - compiled all source and test modules successfully
- `uv run python -m mlb_props_stack`
  - printed the runtime summary successfully

## Recommended Next Issue

- Improve the starter strikeout baseline against the naive benchmark using the
  new readable offline report as the acceptance gate

Why this should go next:

- the repo now produces interpretable offline summaries, so model iteration can
  be judged quickly without hand-parsing `evaluation.json`
- the latest saved held-out run still underperforms the naive benchmark, so the
  clearest product value is improving the actual model rather than more report
  plumbing

Suggested acceptance criteria:

- keep a fixed historical window for before / after comparison
- record held-out RMSE and MAE for both benchmark and model in the PR
- do not call the change an improvement unless held-out metrics actually beat
  the prior run

## Constraints And Open Questions

- The new summaries improve readability, but the current training loop still
  fetches same-day Statcast outcome rows when labels are rebuilt
- If fast offline retraining becomes the next real pain point, the follow-up
  slice should cache or reuse historical outcome labels so local eval does not
  depend on live network access each time
- Keep future model-improvement reviews honest:
  - compare against the naive benchmark
  - compare against the previous run on the same date window
  - preserve the timestamp-valid feature rules already encoded in the training
    data
