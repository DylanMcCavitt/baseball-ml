# Runtime Review Checks

## Why This Exists

Baseline repo checks can stay green while real runtime entrypoints still fail.
That happened on April 22, 2026 after `AGE-153` merged:

- `uv run pytest` and `uv run python -m mlb_props_stack` passed
- `uv run streamlit run src/mlb_props_stack/dashboard/app.py` failed on a
  relative import
- historical `ingest-statcast-features` failed even though the code path was
  supposed to support past-date backfills
- live workflow output still produced empty `daily_candidates` because
  unmatched same-team Odds API events were still being normalized into
  `prop_line_snapshots`, while some true target-date matched events had no
  pitcher-strikeout markets yet
- before `AGE-189`, CI had no dedicated fixture-backed smoke layer for the
  dashboard file path, historical metadata backfill coverage, or training
  entrypoint coverage

Use this document during implementation review, PR review, and final handoff
any time a change touches runtime entrypoints or generated artifacts.

## Baseline Checks

Run these on every issue unless the repo is already blocked before setup:

```bash
uv sync --extra dev
uv run pytest tests/test_runtime_smokes.py
uv run pytest
python3 -m compileall src tests
uv run python -m mlb_props_stack
```

These are necessary. They are not sufficient for dashboard, ingest, training,
or slate-output changes.

The dedicated fixture-backed smoke suite lives in:

```bash
uv run pytest tests/test_runtime_smokes.py
```

That suite now covers:

- the dashboard file-entry path at `src/mlb_props_stack/dashboard/app.py`
- the historical metadata backfill selection used by
  `ingest-statcast-features`
- the seeded `train-starter-strikeout-baseline` runtime path

## Required Runtime Checks By Change Area

### Dashboard Or Streamlit Changes

If a PR touches `src/mlb_props_stack/dashboard/app.py`, dashboard loaders, or
the artifacts it reads:

```bash
uv run pytest tests/test_runtime_smokes.py
uv run streamlit run src/mlb_props_stack/dashboard/app.py
```

Review requirement:

- confirm the app boots instead of throwing an import or startup exception
- confirm the page loads in the browser
- if the page is expected to read artifacts, confirm whether it shows real rows
  or an honest empty-state message
- for board row or table changes, confirm the all-line-row view and any grouped
  pitcher view preserve sportsbook provenance, hidden-row counts, and pitcher
  detail navigation

### CLI, Ingest, Modeling, Backtest, Or Paper-Tracking Changes

Run the actual changed command, not just the module entrypoint:

- `uv run pytest tests/test_runtime_smokes.py`
- `ingest-mlb-metadata`
- `ingest-statcast-features`
- `ingest-odds-api-lines`
- `build-starter-strikeout-dataset`
- `build-pitcher-skill-features`
- `build-lineup-matchup-features`
- `build-workload-leash-features`
- `train-candidate-strikeout-models`
- `validate-model-only-strikeouts`
- `train-starter-strikeout-baseline`
- `compare-starter-strikeout-baselines`
- `build-edge-candidates`
- `build-walk-forward-backtest`
- `build-daily-candidates`
- `build-wager-card`
- `evaluate-stage-gates`

Review requirement:

- use a representative date or seeded fixture path for the changed code path
- if timestamp-valid historical behavior changed, run at least one historical
  date that should succeed without pregame requirements
- if live-slate behavior changed, run a future or clearly pregame slate date
  instead of a date that has already locked

### Artifact-Producing Changes

Do not stop at process exit code. Inspect the outputs.

For training and calibration work, open:

- `starter_game_training_dataset.jsonl`
- `coverage_report.json`
- `coverage_report.md`
- `missing_targets.jsonl`
- `source_manifest.jsonl`
- `schema_drift_report.json`
- `timestamp_policy.md`
- `evaluation.json`
- `evaluation_summary.json`
- `evaluation_summary.md`
- `calibration_summary.json`
- `raw_vs_calibrated_probabilities.jsonl`
- `ladder_probabilities.jsonl`

For starter-game dataset builds, confirm:

- row counts by season include the intended multi-season window
- `source_chunks.cap_warning_count` is `0` or every warning is explained
- `missing_targets.jsonl` contains only known starter edge cases
- `timestamp_policy.status` is `ok`

For projection-rebuild feature layers, inspect:

- `pitcher_skill_features.jsonl`
- `lineup_matchup_features.jsonl`
- `batter_matchup_features.jsonl`
- `workload_leash_features.jsonl`
- `feature_report.json`
- `feature_report.md`
- `reproducibility_notes.md`

For lineup matchup builds, confirm the report distinguishes no confirmed
lineup, no projection, and incomplete batter history; confirm
`leakage_policy.status` is `ok`; and spot-check that same-game batter IDs do
not appear in the projected lineup when no pregame lineup snapshot exists.

For workload/leash builds, confirm `rest_policy.raw_rest_days_primary_driver`
is `false`, long-layoff rows are counted separately from standard rest, and
`leakage_policy.status` is `ok`. Spot-check that expected pitch count and
expected batters faced come from prior starts/team context, while IL, rehab,
and role-change flags remain unknown/false unless a timestamp-valid source
explicitly backs them.

For model-variant comparison work, open:

- `model_comparison.json`
- `model_comparison.md`
- the core and expanded `evaluation_summary.json` files linked from the report
- the core and expanded `backtest_runs.jsonl`, `clv_summary.jsonl`,
  `roi_summary.jsonl`, and `edge_bucket_summary.jsonl` files linked from the
  report

Confirm that both variants use the same date window and cutoff, that optional
features are listed as active or explicitly excluded, and that final-gate
approved wager counts are reported separately from edge-rule placed bets.

For candidate strikeout model-family work, open:

- `model_comparison.json`
- `model_comparison.md`
- `selected_model.json`
- `model_outputs.jsonl`
- `reproducibility_notes.md`

Confirm that every trained family used the same date split, the selected
candidate was chosen by validation evidence, the report includes MAE/RMSE,
common-line log-loss and Brier scores, calibration curves, distribution
diagnostics, and feature-group contribution summaries. Spot-check
`model_outputs.jsonl` for a full count distribution, arbitrary line
over/under probability support from the count distribution, and uncertainty
intervals. Confirm the command did not emit edge candidates, wager approval
rows, or betting decisions.

For model-only walk-forward validation work, open:

- `validation_report.json`
- `validation_report.md`
- `validation_predictions.jsonl`
- `reproducibility_notes.md`

Confirm headline metrics use rolling walk-forward season splits rather than
random row splits. Check that MAE/RMSE, count-distribution log loss, common-line
log loss and Brier, calibration by line bucket and confidence bucket, bias by
pitcher tier, handedness, workload, rest/layoff, season, and rule environment,
recency sensitivity, observed calibration-derived threshold proposals, and the
go/no-go recommendation are present. Confirm the report did not emit wagering,
CLV, ROI, edge-candidate, approval, or stake-sizing metrics.

For odds, edge, or daily candidate work, inspect:

- CLI summary counts such as `matched_events`, `unmatched_events`,
  `scored_candidates`, `actionable_candidates`, `approved_wagers`, and
  `blocked_candidates`
- `event_game_mappings.jsonl`
- `prop_line_snapshots.jsonl`
- `edge_candidates.jsonl`
- `daily_candidates.jsonl`
- `wager_card.jsonl`
- `wager_card_metadata.json`
- `paper_results.jsonl`

For daily candidate approval changes, count `wager_approved=true` and
`bet_placed=true` in `daily_candidates.jsonl`, then confirm the same count
lands in `paper_results.jsonl` for that date. Rows can remain
`evaluation_status=actionable` while still being blocked by final wager gates
such as hold, confidence, model age, same-pitcher correlation, or daily
exposure. For wager-card changes, confirm `build-wager-card --date ...` reports
the same approved count as the dashboard board's `plays cleared` metric.

For readiness changes, run `evaluate-stage-gates`, inspect
`stage_gate_report.json`, and confirm the printed status matches the saved
report. The default command is informational; use `--fail-on-research-only`
only when the calling workflow should fail on a research-only status.

If a workflow is expected to produce scored rows but writes only skipped or
empty outputs, call that out explicitly in the PR and handoff.

## PR Review And Handoff Requirements

Every PR that touches runtime entrypoints or artifact contracts must record:

- whether `uv run pytest tests/test_runtime_smokes.py` passed
- the exact commands that were run
- the exact dates or fixture inputs used
- whether the outputs were non-empty, partially matched, or intentionally empty
- any external-source caveats such as upstream `502` or `503` responses

Do not write "all checks passed" if only the baseline repo checks ran.

## Current Known Gaps

- future-slate runs can still honestly produce zero scored candidates when the
  matched target-date Odds API events return `bookmakers: []` or otherwise lack
  pitcher-strikeout markets at capture time; treat
  `matched_events_without_props` as an availability signal before assuming a
  join regression
- `AGE-189`
  Missing fixture-backed runtime smoke coverage for dashboard boot and
  representative pipeline entrypoints
