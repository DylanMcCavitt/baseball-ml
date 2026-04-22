# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Active issue branch: `feat/improve-strikeout-baseline-window`
- `main` already includes the merged offline evaluation-report work
- This branch contains the starter-baseline improvement slice plus one fixed-window
  artifact bundle under
  `data/normalized/starter_strikeout_baseline/start=2026-04-18_end=2026-04-21/`
  and is not merged yet

## What Was Completed In This Slice

- `src/mlb_props_stack/modeling.py`
  - raises the ridge penalty from `1.0` to `10.0` for the early short-window
    baseline
  - changes feature selection so the mean model always trains on a dense
    pitcher/workload core and only adds lineup-derived numeric fields when the
    train window has enough populated, non-constant lineup data
  - removes categorical dummy fields from the baseline vectorizer so short
    windows do not overfit to team-side splits with thin history
- `tests/test_modeling.py`
  - adds coverage for sparse optional lineup fields being excluded from the
    trained schema when they are absent in the train window
- `README.md`
  - documents the more conservative short-window baseline behavior
- `docs/modeling.md`
  - documents the dense-core plus conditional-lineup feature policy
- `data/normalized/starter_strikeout_baseline/start=2026-04-18_end=2026-04-21/run=20260422T202712Z/`
  - saves the new fixed-window training artifacts, including
    `evaluation.json`, `evaluation_summary.json`, `evaluation_summary.md`,
    `calibration_summary.json`, `raw_vs_calibrated_probabilities.jsonl`, and
    `ladder_probabilities.jsonl`
- `data/normalized/starter_strikeout_baseline/start=2026-04-18_end=2026-04-21/run=20260422T194922Z/evaluation.json`
  - copied in so the new local run can compute same-window previous-run deltas

## Files Changed

- `README.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/modeling.md`
- `src/mlb_props_stack/modeling.py`
- `tests/test_modeling.py`
- `data/normalized/starter_strikeout_baseline/start=2026-04-18_end=2026-04-21/run=20260422T194922Z/evaluation.json`
- `data/normalized/starter_strikeout_baseline/start=2026-04-18_end=2026-04-21/run=20260422T202712Z/`

## Verification

Commands run successfully on April 22, 2026:

```bash
uv sync --extra dev
uv run pytest tests/test_modeling.py
uv run pytest tests/test_runtime_smokes.py
uv run pytest
uv run python -m mlb_props_stack
uv run python -m mlb_props_stack train-starter-strikeout-baseline \
  --start-date 2026-04-18 \
  --end-date 2026-04-21 \
  --output-dir data
```

Observed results:

- `uv run pytest tests/test_modeling.py`
  - `5 passed`
- `uv run pytest tests/test_runtime_smokes.py`
  - `3 passed`
- `uv run pytest`
  - `66 passed`
- `uv run python -m mlb_props_stack`
  - printed the runtime summary successfully
- fixed-window training run `20260422T202712Z`
  - held-out model RMSE: `2.322574`
  - held-out benchmark RMSE: `2.518693`
  - held-out model MAE: `1.960881`
  - held-out benchmark MAE: `2.047866`
  - `held_out_status=beating_benchmark`
  - previous run for the same window: `20260422T194922Z`
- same-window deltas versus `20260422T194922Z`
  - held-out RMSE delta: `-0.899748`
  - held-out MAE delta: `-0.676436`
  - held-out Spearman delta: `+0.244693`

## Recommended Next Issue

- Tighten short-window probability-calibration selection so the calibrator only
  ships when it does not regress held-out log loss, or falls back to identity
  when the fitted ladder calibration is directionally mixed

Why this should go next:

- the mean model now beats both the naive benchmark and the previous run on the
  fixed historical window, so the clearest remaining modeling regression is in
  calibration selection rather than the mean baseline itself
- the new run improved held-out Brier score and expected calibration error, but
  held-out calibrated log loss worsened from `0.286245` to `0.346381`

## Constraints And Open Questions

- The fixed-window mean model improvement is real on RMSE and MAE, but do not
  oversell the calibration layer yet:
  - `calibration_summary.json` and `evaluation_summary.md` show better held-out
    Brier score and ECE
  - the same artifacts also show worse held-out calibrated log loss
- Optional lineup features can still disappear from the trained schema when the
  train window has missing or late lineup data. Check
  `baseline_model.json -> encoded_feature_names` before interpreting the
  coefficient table.
- Training still rebuilds same-day Statcast outcome labels by refetching
  historical outcome rows during each run. If repeated offline retraining
  becomes painful again, caching or reusing those labels is still an open
  follow-up.
