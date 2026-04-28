# Next Session Handoff

## Status

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Active issue: `AGE-304` - provide human-readable results from model output
- Current issue branch:
  `dylanmccavitt2015/age-304-provide-human-readable-results-from-model-output`
- Base state: branch started from `origin/main` at
  `84a29ab3543899112f810892d6eeb5d852656c19`.
- Implementation state: code, focused/runtime verification, baseline
  verification, and this handoff are complete. Commit, push, PR creation, and
  Linear `In Review` transition are the remaining closeout steps for this
  session.

## What Changed In AGE-304

- Added a human-readable `model_outputs.md` artifact to
  `train-candidate-strikeout-models`.
- The new Markdown report is generated from the same rows as
  `model_outputs.jsonl` and includes:
  - split-level average projection, actual strikeouts, absolute error, and
    predictive standard deviation
  - common-line average over probabilities for all rows, validation, and test
    splits
  - compact example pitcher projection rows with actual K, projected K, central
    80% interval, and over probabilities at 4.5, 5.5, and 6.5 strikeouts
- Surfaced `model_outputs_markdown_path` in:
  - `CandidateStrikeoutModelTrainingResult`
  - `selected_model.json`
  - CLI output summary
- Updated runtime review docs and README so reviewers know to open
  `model_outputs.md` alongside `model_outputs.jsonl`.
- Kept the output explicitly projection-only: no sportsbook pricing, edge
  ranking, wager approval, or stake sizing is emitted or implied.

## Runtime Evidence

- Fixture-backed CLI smoke output directory:
  `/tmp/age304-candidate-runtime`
- Inspected readable report:
  `/tmp/age304-candidate-runtime/normalized/candidate_strikeout_models/start=2026-04-01_end=2026-04-06/run=20260428T012657Z/model_outputs.md`
- The report contained the expected projection-only disclaimer, split summary,
  common-line probability table, and example pitcher output table.

## Files Changed

- `src/mlb_props_stack/candidate_models.py`
- `src/mlb_props_stack/cli.py`
- `tests/test_candidate_models.py`
- `tests/test_cli.py`
- `tests/test_runtime_smokes.py`
- `README.md`
- `docs/review_runtime_checks.md`
- `docs/NEXT_SESSION_HANDOFF.md`

## Verification

Commands run:

```bash
/opt/homebrew/bin/uv sync --extra dev
/opt/homebrew/bin/uv run pytest tests/test_candidate_models.py tests/test_cli.py::test_candidate_strikeout_models_cli_renders_output_summary tests/test_runtime_smokes.py::test_candidate_model_cli_smoke_writes_distribution_artifacts -q
rm -rf /tmp/age304-candidate-runtime && /opt/homebrew/bin/uv run python - <<'PY'
from pathlib import Path
from tests.test_candidate_models import _seed_candidate_training_window

root = Path('/tmp/age304-candidate-runtime')
start_date, end_date, dataset, pitcher, lineup, workload = _seed_candidate_training_window(root)
print(start_date.isoformat())
print(end_date.isoformat())
print(dataset)
print(pitcher)
print(lineup)
print(workload)
PY
/opt/homebrew/bin/uv run python -m mlb_props_stack train-candidate-strikeout-models --start-date 2026-04-01 --end-date 2026-04-06 --output-dir /tmp/age304-candidate-runtime --dataset-run-dir /tmp/age304-candidate-runtime/normalized/starter_strikeout_training_dataset/start=2026-04-01_end=2026-04-06/run=20260426T120000Z --pitcher-skill-run-dir /tmp/age304-candidate-runtime/normalized/pitcher_skill_features/start=2026-04-01_end=2026-04-06/run=20260426T120100Z --lineup-matchup-run-dir /tmp/age304-candidate-runtime/normalized/lineup_matchup_features/start=2026-04-01_end=2026-04-06/run=20260426T120200Z --workload-leash-run-dir /tmp/age304-candidate-runtime/normalized/workload_leash_features/start=2026-04-01_end=2026-04-06/run=20260426T120300Z
sed -n '1,80p' /tmp/age304-candidate-runtime/normalized/candidate_strikeout_models/start=2026-04-01_end=2026-04-06/run=20260428T012657Z/model_outputs.md
/opt/homebrew/bin/uv run pytest
/opt/homebrew/bin/uv run python -m mlb_props_stack
PYTHONPYCACHEPREFIX=/tmp/age304-pycache python3 -m compileall src tests
```

Observed results:

- `uv sync --extra dev`: completed successfully.
- Focused candidate/CLI/runtime tests: `4 passed`.
- Fixture-backed `train-candidate-strikeout-models` CLI run: completed
  successfully, wrote `model_outputs_markdown_path=.../model_outputs.md`.
- Manual artifact inspection: readable Markdown contained the expected
  projection-only summary and example outputs.
- Full test suite: `224 passed` with the existing MLflow/Pydantic deprecation
  warnings.
- `python -m mlb_props_stack`: printed the runtime configuration banner.
- `compileall`: completed successfully.

## Recommended Next Issue

Run the candidate model training command against the canonical full rebuild
artifacts when those data artifacts are available, then inspect the generated
`model_outputs.md` to decide what belongs in the dashboard/readiness view.

## Constraints And Risks

- Do not treat `model_outputs.md` as betting evidence. It is a readable view of
  projection-only model outputs.
- Do not add sportsbook pricing, edge ranking, approval gates, paper tracking,
  or dashboard behavior to this slice.
- Preserve `model_outputs.jsonl` as the machine-readable source of truth; the
  Markdown report should remain a derived reading layer.
