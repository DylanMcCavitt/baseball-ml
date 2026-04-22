# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Active issue branch: `feat/age-154-experiment-tracking`
- `main` already includes the merged held-out baseline improvement work from
  `feat/improve-strikeout-baseline-window`
- This branch adds MLflow-backed experiment tracking plus one new tracked
  training artifact bundle under
  `data/normalized/starter_strikeout_baseline/start=2026-04-18_end=2026-04-23/`
  and one new tracked backtest artifact bundle under
  `data/normalized/walk_forward_backtest/start=2026-04-23_end=2026-04-23/`

## What Was Completed In This Slice

- `src/mlb_props_stack/tracking.py`
  - turns the placeholder tracking seam into a small MLflow helper layer with
    lazy import behavior, a local file-backed store, and separate experiment
    names for training vs backtest runs
- `src/mlb_props_stack/modeling.py`
  - starts an MLflow run for each baseline training execution before writing
    local artifacts
  - logs params, held-out metrics, and the full local run directory into MLflow
  - writes `reproducibility_notes.md` and stores MLflow metadata inside
    `evaluation.json`, `evaluation_summary.json`, `evaluation_summary.md`, and
    `baseline_model.json`
- `src/mlb_props_stack/backtest.py`
  - starts an MLflow run for each walk-forward backtest execution
  - logs summary metrics and the full backtest output directory into MLflow
  - writes `reproducibility_notes.md` and stores MLflow metadata inside
    `backtest_runs.jsonl`
- `src/mlb_props_stack/cli.py`
  - surfaces the training/backtest MLflow run ID, experiment name, and
    reproducibility-note path in the CLI summaries
- `tests/test_tracking.py`
  - adds focused coverage for the MLflow helper behavior without depending on a
    live server
- `tests/test_modeling.py`, `tests/test_backtest.py`, `tests/test_cli.py`,
  `tests/test_runtime_smokes.py`
  - updates direct training/backtest tests to use temp tracking roots and
    asserts the new MLflow/reproducibility contract
- `README.md`
  - documents the new MLflow-backed training/backtest artifacts and the local
    tracking store
- `docs/modeling.md`
  - documents the AGE-154 tracking contract
- `docs/architecture.md`
  - notes that `backtest_runs.jsonl` now carries MLflow traceability fields
- `data/normalized/starter_strikeout_baseline/start=2026-04-18_end=2026-04-23/run=20260422T205727Z/`
  - saves the first tracked local training run, including
    `evaluation_summary.json`, `evaluation_summary.md`, and
    `reproducibility_notes.md`
- `data/normalized/walk_forward_backtest/start=2026-04-23_end=2026-04-23/run=20260422T205734Z/`
  - saves the first tracked local backtest run, including `backtest_runs.jsonl`
    and `reproducibility_notes.md`

## Files Changed

- `.gitignore`
- `README.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/architecture.md`
- `docs/modeling.md`
- `pyproject.toml`
- `src/mlb_props_stack/backtest.py`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/modeling.py`
- `src/mlb_props_stack/tracking.py`
- `tests/test_backtest.py`
- `tests/test_cli.py`
- `tests/test_modeling.py`
- `tests/test_runtime_smokes.py`
- `tests/test_tracking.py`
- `uv.lock`
- `data/normalized/starter_strikeout_baseline/start=2026-04-18_end=2026-04-23/run=20260422T205727Z/`
- `data/normalized/walk_forward_backtest/start=2026-04-23_end=2026-04-23/run=20260422T205734Z/`

## Verification

Commands run successfully on April 22, 2026:

```bash
uv sync --extra dev
uv run python -m mlb_props_stack
uv run pytest tests/test_tracking.py tests/test_modeling.py tests/test_backtest.py \
  tests/test_cli.py tests/test_runtime_smokes.py
uv run pytest
uv run python -m mlb_props_stack train-starter-strikeout-baseline \
  --start-date 2026-04-18 \
  --end-date 2026-04-23 \
  --output-dir data
uv run python -m mlb_props_stack build-walk-forward-backtest \
  --start-date 2026-04-23 \
  --end-date 2026-04-23 \
  --output-dir data \
  --model-run-dir data/normalized/starter_strikeout_baseline/start=2026-04-18_end=2026-04-23/run=20260422T205727Z
```

Observed results:

- `uv run python -m mlb_props_stack`
  - printed the runtime summary successfully
- `uv run pytest tests/test_tracking.py tests/test_modeling.py tests/test_backtest.py tests/test_cli.py tests/test_runtime_smokes.py`
  - `21 passed`
- `uv run pytest`
  - run after AGE-154 changes; final passing count should be re-recorded if more
    tests land before merge
- training run `20260422T205727Z`
  - MLflow run ID: `5fbc851c3c7643daa4add2fb6706eee5`
  - experiment: `mlb-props-stack-starter-strikeout-training`
  - held-out model RMSE: `2.322574`
  - held-out benchmark RMSE: `2.518693`
  - held-out model MAE: `1.960881`
  - held-out benchmark MAE: `2.047866`
  - `held_out_status=beating_benchmark`
- backtest run `20260422T205734Z`
  - MLflow run ID: `f29d23b7a6274d48b64b01b46ea3d8d3`
  - experiment: `mlb-props-stack-walk-forward-backtest`
  - `snapshot_groups=139`
  - `actionable_bets=0`
  - `skipped=139`
  - `bet_outcomes.placed_bets=0`

## Recommended Next Issue

- Tighten calibration/backtest handoff so the tracked walk-forward run can
  distinguish genuinely below-threshold lines from rows that were skipped
  because no honest held-out probability existed for that date or line

Why this should go next:

- AGE-154 now makes training and backtest runs traceable, but the first real
  `2026-04-23` backtest window logged `139` skipped rows and `0` placed bets
- the next issue should make that skip mix easier to diagnose from the saved
  reporting artifacts before anyone overreads a zero-bet ROI row

## Constraints And Open Questions

- `mlflow-skinny==2.22.4` brings in a wider transitive set than the pre-AGE-154
  bootstrap baseline, but it kept the implementation small and the local file
  store simple
- The first real backtest verification window (`2026-04-23`) produced no
  actionable bets. That does not block AGE-154, because the issue only required
  traceable runs and summary artifacts, but it means the next issue should
  inspect why every row was skipped before interpreting the zero-bet summary
- Training still refetches same-day Statcast outcome rows during each run, so
  repeated retraining continues to pay the network cost unless a later caching
  issue tackles it explicitly
