# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current issue branch:
  `dylanmccavitt2015/age-149-add-calibration-and-probability-diagnostics`
- `main` already includes the merged `AGE-143` docs work, `AGE-144` MLB
  metadata ingest, `AGE-145` sportsbook ingest, `AGE-146` Statcast feature
  tables, `AGE-147` starter strikeout baseline mean training, and `AGE-148`
  count-distribution ladder probabilities
- This branch adds `AGE-149`: an explicit probability-calibration layer and
  reliability diagnostics on top of AGE-148’s raw ladder outputs

## What Was Completed In AGE-149

- `src/mlb_props_stack/modeling.py`
  - keeps AGE-147 as the only source of the baseline mean expectation
  - generates expanding-window out-of-fold ladder-event probabilities so each
    predicted date only uses prior dates in the base fit
  - fits and stores an isotonic probability calibrator from those out-of-fold
    ladder events
  - writes new durable artifacts under:
    - `data/normalized/starter_strikeout_baseline/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../`
    - `probability_calibrator.json`
    - `raw_vs_calibrated_probabilities.jsonl`
    - `calibration_summary.json`
  - extends `baseline_model.json` with the stored probability-calibration
    metadata
  - extends `evaluation.json` with honest held-out raw-vs-calibrated
    probability diagnostics:
    - mean Brier score
    - mean log loss
    - expected calibration error
    - reliability bins
  - extends `ladder_probabilities.jsonl` so each row now carries:
    - the raw ladder probabilities
    - the calibrated ladder probabilities
    - calibration metadata for downstream consumption
- `src/mlb_props_stack/cli.py`
  - extends the training summary with:
    - `probability_calibrator.json`
    - `raw_vs_calibrated_probabilities.jsonl`
    - `calibration_summary.json`
- `tests/test_modeling.py`
  - adds deterministic coverage for:
    - stored calibrator artifacts
    - honest prior-only calibration windows on held-out rows
    - calibrated ladder monotonicity and complement behavior
- `tests/test_cli.py`
  - adds CLI coverage for the new calibration artifact paths
- `README.md`
  - documents the new calibration artifacts and calibrated ladder outputs
- `docs/architecture.md`
  - documents the AGE-149 calibration layer in the modeling flow
- `docs/modeling.md`
  - documents the out-of-fold calibration path and probability diagnostics

## Verification Run

These commands were run successfully during AGE-149:

```bash
uv sync --extra dev
python3 -m compileall src tests
uv run pytest
uv run python -m mlb_props_stack
```

Local results:

- `uv run pytest`
  - `40 passed`
- `uv run python -m mlb_props_stack`
  - still prints the runtime summary cleanly

Not run during AGE-149:

```bash
uv run python -m mlb_props_stack train-starter-strikeout-baseline \
  --start-date 2026-04-01 \
  --end-date 2026-04-20
```

Reason:

- the repo still does not commit live AGE-146 feature runs, so the training
  command remains verified through deterministic end-to-end tests rather than a
  networked local dataset build against real feature artifacts

## Recommended Next Issue

- Wire `AGE-145` line snapshots plus the AGE-149 calibrated ladder outputs into
  actual `PropProjection` records and edge evaluation

Why this should go next:

- the repo now has calibrated over/under probabilities for each half-strikeout
  ladder line, not just raw count-distribution outputs
- the next missing seam is turning those calibrated probabilities into real
  `PropProjection` records keyed to sportsbook lines so `edge.py` can rank
  candidate props

## Constraints For The Next Worktree

- Start the next issue worktree from the current `main` after AGE-149 is
  merged.
- Keep AGE-147 as the only source of the baseline mean expectation.
- Reuse the AGE-149 calibrated ladder artifact instead of recalibrating
  probabilities inside pricing or edge code.
- Preserve the current leakage rules:
  - no IDs, target columns, or postgame timestamps in the training matrix
  - no same-day pitch rows in the feature inputs themselves
- Treat `baseline_model.json`, `evaluation.json`, `ladder_probabilities.jsonl`,
  `probability_calibrator.json`, `raw_vs_calibrated_probabilities.jsonl`, and
  `calibration_summary.json` as durable debug artifacts, not throwaway local
  output.

## Open Questions

- A real local smoke run of `train-starter-strikeout-baseline` still needs a
  non-test date span with AGE-146 feature runs already materialized under
  `data/normalized/statcast_search/...`.
- The current calibrator is global across ladder events. A future issue may
  decide to compare that against line-bucket or richer walk-forward
  calibration once real prop pricing and backtests exist.
- Same-day Statcast outcome pulls still hit Baseball Savant directly for the
  training label. If that path becomes rate-limited in practice, the repo may
  need a cached label-build slice later rather than changing the model contract
  here.
