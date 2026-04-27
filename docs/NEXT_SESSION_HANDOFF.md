# Next Session Handoff

## Status

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Active issue: `AGE-291` - train candidate strikeout model families with
  distribution outputs
- Current issue branch:
  `dylanmccavitt2015/age-291-train-candidate-strikeout-model-families-with-distribution`
- Base state: `56d1425` (`origin/main`, AGE-290 merged)
- Linear status was moved from `Todo` to `In Progress` at issue start.
- Implementation state: AGE-291 code, docs, focused tests, full tests, and real
  artifact smoke checks are complete in this worktree. The branch is ready for
  PR review after the latest verification pass, commit, and push.

## What Changed In AGE-291

- Added `src/mlb_props_stack/candidate_models.py`, a projection-only candidate
  model-family trainer over the AGE-287 starter-game dataset and optional
  AGE-288/289/290 feature layers.
- Added the CLI command:

```bash
uv run python -m mlb_props_stack train-candidate-strikeout-models \
  --start-date 2019-03-20 \
  --end-date 2026-04-24 \
  --output-dir data
```

- Candidate families now trained or explicitly represented in one comparable
  report:
  - Poisson count GLM baseline
  - negative-binomial count GLM baseline
  - plate-appearance logistic K-rate model by expected batters faced
  - repo-approved tree ensemble equivalent using deterministic boosted decision
    stumps
  - validation-selected top-two mean blend
  - neural/sequence challenger recorded as skipped until sequence-shaped inputs
    and dependency/architecture approval exist
- Added distribution-output artifacts:
  - `model_comparison.json`
  - `model_comparison.md`
  - `selected_model.json`
  - `model_outputs.jsonl`
  - `reproducibility_notes.md`
- `model_outputs.jsonl` writes one selected-model row per starter-game with:
  - point strikeout projection
  - full strikeout-count probability distribution
  - over/under probabilities for common half-strikeout lines
  - an explicit arbitrary-line probability contract backed by the full count
    distribution
  - predictive standard deviation and central 80% interval
- Selection is validation-evidence based:
  - primary metric is validation common-line mean log loss
  - validation distribution diagnostics and RMSE are tie-breakers
  - test and held-out metrics are reported for audit only
- Feature group contribution summaries are grouped as:
  - pitcher skill
  - matchup
  - workload
  - context
- No betting decisions, edge candidates, wager approval rows, dashboard changes,
  or pricing-layer behavior were added.

## Local AGE-291 Data Evidence

Real starter-dataset-only smoke:

```bash
rm -rf /tmp/age291-candidate-smoke && UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run python -m mlb_props_stack train-candidate-strikeout-models --start-date 2019-03-20 --end-date 2019-03-28 --output-dir /tmp/age291-candidate-smoke --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
```

Result:

- run: `/tmp/age291-candidate-smoke/normalized/candidate_strikeout_models/start=2019-03-20_end=2019-03-28/run=20260427T221839Z/`
- rows: `34`
- selected candidate: `negative_binomial_glm_count_baseline`
- inspected `model_comparison.json`, `model_comparison.md`, and
  `model_outputs.jsonl`
- first output row had a non-empty count distribution, common line probabilities
  starting at `2.5`, arbitrary-line probability support, and confidence
  interval `[1, 6]`

Real canonical pitcher-skill join smoke:

```bash
rm -rf /tmp/age291-candidate-skill-smoke && UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run python -m mlb_props_stack train-candidate-strikeout-models --start-date 2024-04-01 --end-date 2024-04-03 --output-dir /tmp/age291-candidate-skill-smoke --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z --pitcher-skill-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/pitcher_skill_features/start=2024-04-01_end=2024-04-03/run=20260425T180426Z
```

Result:

- run: `/tmp/age291-candidate-skill-smoke/normalized/candidate_strikeout_models/start=2024-04-01_end=2024-04-03/run=20260427T221847Z/`
- rows: `80`
- pitcher-skill feature matches: `80`
- selected candidate: `boosted_stump_tree_ensemble`
- selected model contribution share: `pitcher_skill=1.0`
- inspected `model_comparison.json` and `model_outputs.jsonl`; line
  probabilities and count distributions were non-empty

## Files Changed

- `README.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `docs/architecture.md`
- `docs/modeling.md`
- `docs/review_runtime_checks.md`
- `src/mlb_props_stack/candidate_models.py`
- `src/mlb_props_stack/cli.py`
- `tests/test_candidate_models.py`
- `tests/test_cli.py`
- `tests/test_runtime_smokes.py`

## Verification

Commands run:

```bash
UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run pytest tests/test_candidate_models.py tests/test_cli.py tests/test_runtime_smokes.py -q
PYTHONPYCACHEPREFIX=/tmp/age291-pycache python3 -m compileall src tests
UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv sync --extra dev
UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run pytest
UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run python -m mlb_props_stack
rm -rf /tmp/age291-candidate-smoke && UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run python -m mlb_props_stack train-candidate-strikeout-models --start-date 2019-03-20 --end-date 2019-03-28 --output-dir /tmp/age291-candidate-smoke --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z
rm -rf /tmp/age291-candidate-skill-smoke && UV_CACHE_DIR=/tmp/uv-cache /opt/homebrew/bin/uv run python -m mlb_props_stack train-candidate-strikeout-models --start-date 2024-04-01 --end-date 2024-04-03 --output-dir /tmp/age291-candidate-skill-smoke --dataset-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/starter_strikeout_training_dataset/start=2019-03-20_end=2026-04-24/run=20260425T145813Z --pitcher-skill-run-dir /Users/dylanmccavitt/projects/nba-ml/data/normalized/pitcher_skill_features/start=2024-04-01_end=2024-04-03/run=20260425T180426Z
```

Observed results:

- Candidate/CLI/runtime smoke tests: `22 passed` with existing third-party
  MLflow/Pydantic warnings.
- `python3 -m compileall src tests` completed successfully with
  `PYTHONPYCACHEPREFIX=/tmp/age291-pycache`.
- `uv sync --extra dev` completed successfully with `UV_CACHE_DIR=/tmp/uv-cache`.
- Full suite: `220 passed` with existing third-party MLflow/Pydantic warnings.
- `python -m mlb_props_stack` printed the runtime configuration banner.
- Starter-dataset-only artifact smoke succeeded, selected
  `negative_binomial_glm_count_baseline`, and verified the model comparison,
  distribution output, arbitrary-line contract, and confidence interval.
- Pitcher-skill join artifact smoke succeeded, selected
  `boosted_stump_tree_ensemble`, matched `80` pitcher-skill rows, and verified
  selected feature-group contributions.

## Closeout State

- Git metadata writes are working in this resumed Symphony worktree.
- The AGE-291 branch was created from `origin/main` at `56d1425`.
- No existing PR was found for the branch before closeout.
- Remaining closeout steps for this session: commit, push, open the PR, link it
  in Linear, and move `AGE-291` to `In Review`.

## Recommended Next Issue After AGE-291

1. `AGE-292` - model-only walk-forward validation.
2. Then continue through:
   - `AGE-293` - scoreable historical market joins
   - `AGE-294` - rebuilt betting layer
   - `AGE-295` - dashboard and approved-wager UX reconnection

## Constraints And Risks

- The current candidate command is projection-only. Do not treat its output as
  betting-ready.
- Do not resume live-readiness, approved-wager evidence refresh, or dashboard
  reconnection work before projection validation gates pass.
- The tree ensemble is a standard-library boosted-stump equivalent, not
  LightGBM/XGBoost/CatBoost; adding those dependencies remains a separate repo
  approval decision.
- The neural challenger is skipped because the current artifacts are tabular
  aggregate rows, not sequence-shaped pitch or batter inputs.
- Preserve timestamp ordering and feature boundaries:
  - starter-game outcomes are target labels only
  - pitcher skill is strikeout ability
  - lineup matchup is opponent K vulnerability
  - workload/leash is opportunity volume
- Do not use v0 artifacts, metrics, feature assumptions, or code paths as
  current modeling evidence.

## Deferred Or Superseded Issues

- `AGE-262` and `AGE-263` are canceled/superseded by `AGE-285`.
- `AGE-207` and `AGE-208` are canceled/superseded by `AGE-291`.
- `AGE-209` waits for `AGE-294`.
- `AGE-210` has its AGE-286 audit dependency satisfied but still waits for
  `AGE-291`.
- `AGE-212` waits for `AGE-293` / `AGE-294`.
