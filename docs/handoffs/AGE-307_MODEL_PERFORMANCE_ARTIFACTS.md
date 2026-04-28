# AGE-307 Model Performance Artifact Manifest

## Status

- Issue: `AGE-307` - materialize 2019-2026 pitcher strikeout model performance artifacts.
- Workspace: `/Users/dylanmccavitt/.codex/worktrees/symphony-nba-ml/AGE-307`
- Branch: `dylanmccavitt2015/age-307-materialize-2019-2026-pitcher-strikeout-model-performance`
- Window: `2019-03-20` through `2026-04-24`
- Promotion status: `research_only`
- Betting-layer status: `blocked`
- Validation recommendation: `no_go_betting_layer_still_blocked`

Generated artifacts are intentionally ignored by git under `data/normalized/`.
This manifest records the local artifact set and the exact commands used to
produce it.

## Restored Starter Dataset

Path:

`data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z/`

Rows and source counts:

- `starter_game_training_dataset.jsonl`: `31,729`
- `source_manifest.jsonl`: `587`
- `missing_targets.jsonl`: `3`
- `coverage_report.json`: present
- `coverage_report.md`: present
- `schema_drift_report.json`: present
- `timestamp_policy.md`: present
- `reproducibility_notes.md`: present

Restoration command:

```bash
mkdir -p data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24
rsync -a /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/
```

Note: `source_manifest.jsonl` references the canonical raw Statcast CSV cache
under `/Users/dylanmccavitt/projects/nba-ml/data/raw/...`. This issue session
read that cache but did not edit the canonical checkout.

## Feature Artifacts

### Pitcher Skill

Path:

`data/normalized/pitcher_skill_features/start=2019-03-20_end=2026-04-24/run=20260428T184411Z/`

Command:

```bash
/opt/homebrew/bin/uv run python -m mlb_props_stack build-pitcher-skill-features --start-date 2019-03-20 --end-date 2026-04-24 --output-dir data --dataset-run-dir data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
```

Counts:

- dataset rows: `31,729`
- feature rows: `31,729`
- pitch rows read: `4,684,022`
- pitchers: `1,018`

### Workload And Leash

Path:

`data/normalized/workload_leash_features/start=2019-03-20_end=2026-04-24/run=20260428T201400Z/`

Command:

```bash
/opt/homebrew/bin/uv run python -m mlb_props_stack build-workload-leash-features --start-date 2019-03-20 --end-date 2026-04-24 --output-dir data --dataset-run-dir data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
```

Counts:

- dataset rows: `31,729`
- feature rows: `31,729`
- pitch rows read: `4,684,022`
- pitchers: `1,018`

### Lineup Matchup Blocker

The full-window lineup matchup feature artifact was not produced.

Attempted command:

```bash
/opt/homebrew/bin/uv run python -m mlb_props_stack build-lineup-matchup-features --start-date 2019-03-20 --end-date 2026-04-24 --output-dir data --dataset-run-dir data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
```

Observed result:

- The first full run was terminated after roughly 30 minutes with exit code `143`.
- A targeted optimization was added in `src/mlb_props_stack/lineup_matchup_features.py`
  to avoid repeated prior-history sorting and temporary count lists.
- A one-month smoke rerun after the optimization completed successfully:
  `2019-03-20` through `2019-04-20`, `618` feature rows, `5,292` batter rows,
  `97,388` pitch rows.
- The retried full-window run was again terminated with exit code `143` before
  writing a lineup artifact.

Exact missing artifact family:

`data/normalized/lineup_matchup_features/start=2019-03-20_end=2026-04-24/run=.../`

The next worker should either further optimize `build-lineup-matchup-features`
for full-window history or run it in an environment with a longer process
window, then rerun the candidate and validation commands with
`--lineup-matchup-run-dir`.

## Candidate Model Artifacts

Path:

`data/normalized/candidate_strikeout_models/start=2019-03-20_end=2026-04-24/run=20260428T201438Z/`

Command:

```bash
/opt/homebrew/bin/uv run python -m mlb_props_stack train-candidate-strikeout-models --start-date 2019-03-20 --end-date 2026-04-24 --output-dir data --dataset-run-dir data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z --pitcher-skill-run-dir data/normalized/pitcher_skill_features/start=2019-03-20_end=2026-04-24/run=20260428T184411Z --workload-leash-run-dir data/normalized/workload_leash_features/start=2019-03-20_end=2026-04-24/run=20260428T201400Z
```

Produced files:

- `model_comparison.md`
- `model_comparison.json`
- `selected_model.json`
- `model_outputs.jsonl`
- `reproducibility_notes.md`

Counts and coverage from `model_comparison.json`:

- joined rows: `31,729`
- pitcher skill matches: `31,729`
- workload/leash matches: `31,729`
- lineup matchup matches: `0`
- selected candidate: `validation_top_two_mean_blend`
- selected validation common-line log loss: `0.448992`
- selected validation RMSE: `2.319311`
- held-out common-line log loss: `0.447132`
- held-out RMSE: `2.304100`
- skipped candidate: `neural_sequence_challenger`, because current artifacts
  are tabular aggregate rows rather than sequence-shaped inputs.

`model_outputs.jsonl` has `31,729` rows. Rows include:

- `point_projection`
- full `probability_distribution`
- `over_under_probabilities`
- `line_probability_contract` with arbitrary-line support
- `confidence.central_80_interval`
- `model_input_refs`

## Model-Only Validation Artifacts

Path:

`data/normalized/model_only_walk_forward_validation/start=2019-03-20_end=2026-04-24/run=20260428T201620Z/`

Command:

```bash
/opt/homebrew/bin/uv run python -m mlb_props_stack validate-model-only-strikeouts --start-date 2019-03-20 --end-date 2026-04-24 --output-dir data --dataset-run-dir data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z --pitcher-skill-run-dir data/normalized/pitcher_skill_features/start=2019-03-20_end=2026-04-24/run=20260428T184411Z --workload-leash-run-dir data/normalized/workload_leash_features/start=2019-03-20_end=2026-04-24/run=20260428T201400Z
```

Produced files:

- `validation_report.md`
- `validation_report.json`
- `validation_predictions.jsonl`
- `reproducibility_notes.md`

Counts and headline metrics:

- walk-forward splits: `4`
- held-out predictions: `15,358`
- MAE: `1.837943`
- RMSE: `2.300842`
- mean bias: `-0.057394`
- count log loss: `2.231362`
- common-line log loss: `0.445958`
- common-line Brier: `0.145316`
- recommendation: `no_go_betting_layer_still_blocked`
- blocking missing feature layer: `lineup_matchup_matches`

The validation report observed calibration-derived threshold proposals, but the
model remains `research_only` and is not ready for betting-layer validation
until lineup matchup coverage is produced and the model-only validation is
rerun with that artifact.

