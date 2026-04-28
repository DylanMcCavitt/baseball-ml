# Next Session Handoff

## Status

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Active issue: `AGE-307` - materialize 2019-2026 pitcher strikeout model
  performance artifacts.
- Current branch:
  `dylanmccavitt2015/age-307-materialize-2019-2026-pitcher-strikeout-model-performance`
- Workspace:
  `/Users/dylanmccavitt/.codex/worktrees/symphony-nba-ml/AGE-307`
- Artifact window: `2019-03-20` through `2026-04-24`
- PR: pending
- Implementation state: full-window starter dataset restored locally;
  full-window pitcher skill, workload/leash, candidate-model, and model-only
  validation artifacts generated; full-window lineup matchup artifact remains
  blocked by runtime termination.
- Durable artifact manifest:
  `docs/handoffs/AGE-307_MODEL_PERFORMANCE_ARTIFACTS.md`

## What Changed In AGE-307

- Restored the full 2019-2026 starter strikeout training dataset into this
  worktree's ignored `data/normalized/` tree.
- Generated full-window pitcher skill features:
  `data/normalized/pitcher_skill_features/start=2019-03-20_end=2026-04-24/run=20260428T184411Z/`
- Generated full-window workload/leash features:
  `data/normalized/workload_leash_features/start=2019-03-20_end=2026-04-24/run=20260428T201400Z/`
- Added a targeted optimization in
  `src/mlb_props_stack/lineup_matchup_features.py` so lineup matchup builds
  index prior batter/pitcher rows by date and avoid repeated temporary count
  lists.
- Ran full-window candidate model training without lineup matchup coverage:
  `data/normalized/candidate_strikeout_models/start=2019-03-20_end=2026-04-24/run=20260428T201438Z/`
- Ran full-window model-only walk-forward validation without lineup matchup
  coverage:
  `data/normalized/model_only_walk_forward_validation/start=2019-03-20_end=2026-04-24/run=20260428T201620Z/`
- Added `docs/handoffs/AGE-307_MODEL_PERFORMANCE_ARTIFACTS.md` with exact
  artifact paths, commands, row counts, feature match counts, and blocker
  details.

## Artifact Summary

- Starter dataset rows: `31,729`
- Pitcher skill feature rows: `31,729`
- Workload/leash feature rows: `31,729`
- Lineup matchup feature rows: `0` for the full window
- Candidate model output rows: `31,729`
- Validation prediction rows: `15,358`
- Candidate selected: `validation_top_two_mean_blend`
- Candidate selected validation common-line log loss: `0.448992`
- Candidate selected validation RMSE: `2.319311`
- Validation MAE: `1.837943`
- Validation RMSE: `2.300842`
- Validation common-line log loss: `0.445958`
- Validation common-line Brier: `0.145316`
- Validation recommendation: `no_go_betting_layer_still_blocked`
- Promotion state: `research_only`

## Blocker

The full-window lineup matchup artifact did not materialize.

Attempted command:

```bash
/opt/homebrew/bin/uv run python -m mlb_props_stack build-lineup-matchup-features --start-date 2019-03-20 --end-date 2026-04-24 --output-dir data --dataset-run-dir data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
```

Observed result:

- The full-window command was terminated with exit code `143` before writing
  artifacts.
- After the optimization, a smoke run for `2019-03-20` through `2019-04-20`
  completed successfully with `618` feature rows, `5,292` batter rows, and
  `97,388` pitch rows.
- A retried full-window run was still terminated with exit code `143`.

The missing artifact family is:

`data/normalized/lineup_matchup_features/start=2019-03-20_end=2026-04-24/run=.../`

Candidate training and validation therefore report:

- `pitcher_skill_matches=31729`
- `workload_leash_matches=31729`
- `lineup_matchup_matches=0`

## Files Changed

- `src/mlb_props_stack/lineup_matchup_features.py`
- `docs/handoffs/AGE-307_MODEL_PERFORMANCE_ARTIFACTS.md`
- `docs/NEXT_SESSION_HANDOFF.md`

Generated but ignored local artifacts under `data/normalized/` are not intended
to be committed.

## Verification

Commands run:

```bash
/opt/homebrew/bin/uv sync --extra dev
PYTHONPYCACHEPREFIX=/tmp/age307-pycache python3 -m py_compile src/mlb_props_stack/lineup_matchup_features.py
/opt/homebrew/bin/uv run pytest tests/test_cli.py -q
/opt/homebrew/bin/uv run python -m mlb_props_stack build-lineup-matchup-features --start-date 2019-03-20 --end-date 2019-04-20 --output-dir /tmp/age307-profile2 --dataset-run-dir data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
/opt/homebrew/bin/uv run python -m mlb_props_stack build-pitcher-skill-features --start-date 2019-03-20 --end-date 2026-04-24 --output-dir data --dataset-run-dir data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
/opt/homebrew/bin/uv run python -m mlb_props_stack build-workload-leash-features --start-date 2019-03-20 --end-date 2026-04-24 --output-dir data --dataset-run-dir data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
/opt/homebrew/bin/uv run python -m mlb_props_stack train-candidate-strikeout-models --start-date 2019-03-20 --end-date 2026-04-24 --output-dir data --dataset-run-dir data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z --pitcher-skill-run-dir data/normalized/pitcher_skill_features/start=2019-03-20_end=2026-04-24/run=20260428T184411Z --workload-leash-run-dir data/normalized/workload_leash_features/start=2019-03-20_end=2026-04-24/run=20260428T201400Z
/opt/homebrew/bin/uv run python -m mlb_props_stack validate-model-only-strikeouts --start-date 2019-03-20 --end-date 2026-04-24 --output-dir data --dataset-run-dir data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z --pitcher-skill-run-dir data/normalized/pitcher_skill_features/start=2019-03-20_end=2026-04-24/run=20260428T184411Z --workload-leash-run-dir data/normalized/workload_leash_features/start=2019-03-20_end=2026-04-24/run=20260428T201400Z
/opt/homebrew/bin/uv run pytest
PYTHONPYCACHEPREFIX=/tmp/age307-pycache python3 -m compileall src tests
/opt/homebrew/bin/uv run python -m mlb_props_stack
git diff --check
```

Observed results:

- `uv sync --extra dev`: completed successfully.
- `py_compile`: completed successfully.
- `tests/test_cli.py`: `16 passed`.
- Lineup smoke build: completed successfully for `2019-03-20` through
  `2019-04-20`.
- Full pitcher skill build: completed successfully.
- Full workload/leash build: completed successfully.
- Candidate model training: completed successfully.
- Model-only validation: completed successfully with recommendation
  `no_go_betting_layer_still_blocked`.
- Full test suite: `226 passed`, with existing third-party
  MLflow/Pydantic warnings.
- `compileall`: completed successfully.
- `python -m mlb_props_stack`: printed the runtime configuration banner.
- `git diff --check`: no whitespace errors.

## Recommended Next Issue

1. Finish full-window lineup matchup feature materialization for
   `2019-03-20` through `2026-04-24`.
2. Rerun `train-candidate-strikeout-models` with explicit
   `--lineup-matchup-run-dir`.
3. Rerun `validate-model-only-strikeouts` with explicit lineup matchup coverage.
4. Only after model-only validation clears, create or resume the betting-layer
   issue to restore historical strikeout odds artifacts and evaluate
   `build-edge-candidates`, CLV, ROI, and wager gates against real market data.

## Constraints And Risks

- Do not treat the current candidate or validation artifacts as betting-ready.
- Do not approve real wagers or loosen gates to force output.
- The current artifacts are useful model-performance evidence, but the missing
  lineup matchup feature layer keeps the model `research_only`.
- Generated artifacts are ignored and local to this worktree.
