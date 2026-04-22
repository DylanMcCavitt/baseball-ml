# Training Reproducibility Notes

- Local run ID: `20260422T205727Z`
- MLflow run ID: `5fbc851c3c7643daa4add2fb6706eee5`
- MLflow experiment: `mlb-props-stack-starter-strikeout-training`
- Tracking URI: `file:./artifacts/mlruns`
- Local run directory: `data/normalized/starter_strikeout_baseline/start=2026-04-18_end=2026-04-23/run=20260422T205727Z`
- Date window: `2026-04-18` -> `2026-04-23`
- CLI rerun command: `uv run python -m mlb_props_stack train-starter-strikeout-baseline --start-date 2026-04-18 --end-date 2026-04-23 --output-dir data`
- Inputs: saved AGE-146 Statcast feature rows already written under `data/normalized/statcast_search/date=...` inside the requested window.
- Honest rules: date splits stay chronological and same-game outcome rows are fetched without using post-game features.
