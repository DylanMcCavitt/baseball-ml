# Next Session Handoff

## Status

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Active issue: `AGE-311` - vertical slice: starter strikeout ML report
- Current issue branch:
  `dylanmccavitt2015/age-311-vertical-slice-starter-strikeout-ml-report`
- Base state: branch created from `origin/main` after `git fetch origin --prune`.
- PR: https://github.com/DylanMcCavitt/baseball-ml/pull/55
- Implementation state: code, docs, focused tests, full tests, compile check,
  module smoke, exact CLI smoke window, exact CLI historical window, and output
  artifact inspection are complete.

## What Changed In AGE-311

- Added `build-starter-strikeout-ml-report`, a research-only vertical report
  command over the starter strikeout ML path.
- Added `src/mlb_props_stack/starter_ml_report.py`, which reuses the existing
  candidate model families and feature-join helpers. It does not add model
  families, feature families, sportsbook pricing, CLV, ROI, edge approvals, or
  stake sizing.
- The report reads an existing `starter_strikeout_training_dataset` artifact and
  any existing pitcher-skill, lineup-matchup, and workload/leash feature runs
  that can be joined by `training_row_id`.
- The report writes:
  - `starter_strikeout_ml_report.json`
  - `starter_strikeout_ml_report.md`
  - `starter_strikeout_ml_predictions.jsonl`
  - `reproducibility_notes.md`
- The report includes row counts, date-ordered train/validation/test windows,
  selected feature columns, missing or excluded feature families,
  leakage/timestamp status, held-out RMSE/MAE/Spearman rank correlation, bias
  slices, best/worst prediction examples, and count/common-line probability
  quality.
- Updated `docs/architecture.md`, `docs/modeling.md`, and
  `docs/review_runtime_checks.md` with the new command and review expectations.

## Runtime Evidence

### Tiny Smoke Window

Command:

```bash
rm -rf /tmp/age311-ml-report-smoke && /opt/homebrew/bin/uv run python -m mlb_props_stack build-starter-strikeout-ml-report --start-date 2026-04-01 --end-date 2026-04-08 --output-dir /tmp/age311-ml-report-smoke --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
```

Result:

- `rows=212`
- `selected_candidate=boosted_stump_tree_ensemble`
- `held_out_rmse=2.369923`
- `held_out_mae=2.041447`
- Report path:
  `/tmp/age311-ml-report-smoke/normalized/starter_strikeout_ml_report/start=2026-04-01_end=2026-04-08/run=20260428T202921Z/starter_strikeout_ml_report.json`
- Markdown path:
  `/tmp/age311-ml-report-smoke/normalized/starter_strikeout_ml_report/start=2026-04-01_end=2026-04-08/run=20260428T202921Z/starter_strikeout_ml_report.md`
- Predictions path:
  `/tmp/age311-ml-report-smoke/normalized/starter_strikeout_ml_report/start=2026-04-01_end=2026-04-08/run=20260428T202921Z/starter_strikeout_ml_predictions.jsonl`

### Meaningful Historical Window

Command:

```bash
rm -rf /tmp/age311-ml-report-historical && /opt/homebrew/bin/uv run python -m mlb_props_stack build-starter-strikeout-ml-report --start-date 2019-03-20 --end-date 2026-04-24 --output-dir /tmp/age311-ml-report-historical --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
```

Result:

- `rows=31729`
- `selected_candidate=validation_top_two_mean_blend`
- `held_out_rmse=2.480789`
- `held_out_mae=1.995703`
- Held-out Spearman rank correlation: `0.065863`
- Held-out rows: `12582`
- Common-line probability events: `100656`
- Common-line log loss: `0.482648`
- Common-line Brier: `0.158335`
- Count-distribution negative log likelihood: `2.309371`
- Report path:
  `/tmp/age311-ml-report-historical/normalized/starter_strikeout_ml_report/start=2019-03-20_end=2026-04-24/run=20260428T202955Z/starter_strikeout_ml_report.json`
- Markdown path:
  `/tmp/age311-ml-report-historical/normalized/starter_strikeout_ml_report/start=2019-03-20_end=2026-04-24/run=20260428T202955Z/starter_strikeout_ml_report.md`
- Predictions path:
  `/tmp/age311-ml-report-historical/normalized/starter_strikeout_ml_report/start=2019-03-20_end=2026-04-24/run=20260428T202955Z/starter_strikeout_ml_predictions.jsonl`

Artifact inspection confirmed:

- `starter_strikeout_ml_predictions.jsonl` has `31,729` rows.
- The report records the canonical feature-layer bottleneck explicitly:
  `pitcher_skill`, `matchup`, and `workload` were
  `missing_artifact_or_no_joined_rows`.
- No feature builders were run; the report fell back to dense starter-game
  context columns: `context_home`, `context_pitch_clock_era`, and
  `context_pitcher_right_handed`.
- Leakage/timestamp status is `ok`; date splits are ordered, random splits are
  not used, and same-game starter strikeouts are target-only.

## Files Changed

- `src/mlb_props_stack/starter_ml_report.py`
- `src/mlb_props_stack/cli.py`
- `tests/test_starter_ml_report.py`
- `tests/test_cli.py`
- `docs/architecture.md`
- `docs/modeling.md`
- `docs/review_runtime_checks.md`
- `docs/NEXT_SESSION_HANDOFF.md`

## Verification

Commands run:

```bash
/opt/homebrew/bin/uv sync --extra dev
/opt/homebrew/bin/uv run pytest tests/test_starter_ml_report.py tests/test_cli.py -q
/opt/homebrew/bin/uv run pytest tests/test_runtime_smokes.py -q
/opt/homebrew/bin/uv run pytest
PYTHONPYCACHEPREFIX=/tmp/age311-pycache python3 -m compileall src tests
/opt/homebrew/bin/uv run python -m mlb_props_stack
git diff --check
```

Observed results:

- `uv sync --extra dev`: completed successfully.
- Focused report/CLI tests: `19 passed`.
- Runtime smoke tests: `5 passed` with existing third-party
  MLflow/Pydantic warnings.
- Full suite: `229 passed` with existing third-party MLflow/Pydantic warnings.
- `compileall`: completed successfully.
- `python -m mlb_props_stack`: printed the runtime configuration banner.
- `git diff --check`: no whitespace errors.

## Recommended Next Issue

Next vertical issue: restore or materialize the full historical pitcher-skill,
lineup-matchup, and workload/leash feature artifacts for the canonical
2019-03-20 through 2026-04-24 starter dataset, then rerun
`build-starter-strikeout-ml-report` with those feature run directories and
compare the report against this dense-context fallback.

## Constraints And Risks

- Do not treat the AGE-311 metrics as betting readiness. This report is
  projection-only research evidence.
- The meaningful historical report currently uses dense starter-game context
  only because the expected feature-layer artifacts were not present under the
  canonical data directory available to this worktree.
- Do not start a full feature-builder rebuild casually in the next thread.
  Prove bounded runtime on a small window first, then expand deliberately.
- Keep same-game Statcast outcomes target-only; do not introduce them as
  features.
