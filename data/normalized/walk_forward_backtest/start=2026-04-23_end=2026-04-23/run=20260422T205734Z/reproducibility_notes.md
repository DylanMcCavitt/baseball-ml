# Walk-Forward Backtest Reproducibility Notes

- Local backtest run ID: `20260422T205734Z`
- MLflow run ID: `f29d23b7a6274d48b64b01b46ea3d8d3`
- MLflow experiment: `mlb-props-stack-walk-forward-backtest`
- Tracking URI: `file:./artifacts/mlruns`
- Local run directory: `data/normalized/walk_forward_backtest/start=2026-04-23_end=2026-04-23/run=20260422T205734Z`
- Date window: `2026-04-23` -> `2026-04-23`
- Source model run ID: `20260422T205727Z`
- Source model run directory: `data/normalized/starter_strikeout_baseline/start=2026-04-18_end=2026-04-23/run=20260422T205727Z`
- Cutoff minutes before first pitch: `30`
- CLI rerun command: `uv run python -m mlb_props_stack build-walk-forward-backtest --start-date 2026-04-23 --end-date 2026-04-23 --output-dir data --model-run-dir data/normalized/starter_strikeout_baseline/start=2026-04-18_end=2026-04-23/run=20260422T205727Z --cutoff-minutes-before-first-pitch 30`
- Honest rules: the rerun pins the exact model run directory so the same saved historical probabilities and timestamps are replayed.
