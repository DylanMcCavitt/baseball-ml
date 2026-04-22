# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current issue branch:
  `dylanmccavitt2015/age-148-fit-strikeout-count-distribution-and-ladder-probabilities`
- `main` already includes the merged `AGE-143` docs work, `AGE-144` MLB
  metadata ingest, `AGE-145` sportsbook ingest, `AGE-146` Statcast feature
  tables, and `AGE-147` starter strikeout baseline mean training
- This branch adds `AGE-148`: the first fitted strikeout-count distribution and
  ladder-probability layer on top of the AGE-147 mean model

## What Was Completed In AGE-148

- `src/mlb_props_stack/modeling.py`
  - kept AGE-147 as the only source of the baseline mean expectation
  - added a fitted global negative-binomial count-distribution layer on top of
    the ridge mean predictions
  - fits one global dispersion parameter with a standard-library
    method-of-moments pass over the train split
  - adds reusable helpers for:
    - one-line over or under probability lookup
    - full half-strikeout ladder generation
  - extends `baseline_model.json` with durable distribution metadata:
    - distribution name
    - fit method
    - fitted dispersion alpha
    - variance formula
  - extends `evaluation.json` with count-distribution metrics on train,
    validation, test, and combined held-out rows:
    - mean negative log likelihood
    - mean ranked probability score
    - held-out comparison against a Poisson fallback using the same mean model
  - writes `ladder_probabilities.jsonl` under:
    - `data/normalized/starter_strikeout_baseline/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../`
  - each ladder row includes:
    - starter-game identifiers
    - split name
    - realized strikeout count
    - naive benchmark mean
    - model mean
    - fitted dispersion alpha
    - half-strikeout ladder over or under probabilities
- `src/mlb_props_stack/cli.py`
  - extends the training summary with:
    - fitted dispersion alpha
    - `ladder_probabilities.jsonl` path
- `tests/test_modeling.py`
  - adds deterministic coverage for:
    - ladder artifact creation
    - persisted distribution metadata
    - reusable line and ladder probability helper behavior
    - ladder monotonicity and probability complement checks
- `tests/test_cli.py`
  - adds CLI coverage for the new distribution summary fields
- `README.md`
  - documents the new distribution metadata and `ladder_probabilities.jsonl`
    artifact
- `docs/architecture.md`
  - documents the new count-distribution layer inside the modeling seam
- `docs/modeling.md`
  - documents the current negative-binomial ladder path and new evaluation
    metrics

## Verification Run

These commands were run successfully during AGE-148:

```bash
uv sync --extra dev
python3 -m compileall src tests
uv run pytest
uv run python -m mlb_props_stack
```

Local results:

- `uv run pytest`
  - `39 passed`
- `uv run python -m mlb_props_stack`
  - still prints the runtime summary cleanly

Not run during AGE-148:

```bash
uv run python -m mlb_props_stack train-starter-strikeout-baseline \
  --start-date 2026-04-01 \
  --end-date 2026-04-20
```

Reason:

- the repo still does not commit live AGE-146 feature runs, so the training
  command remains verified through deterministic end-to-end tests rather than a
  networked local dataset build against real feature artifacts

## Recommended Next Issue

- Wire `AGE-145` line snapshots plus the `AGE-147` or `AGE-148` model outputs
  into actual `PropProjection` records and edge evaluation

Why this should go next:

- the repo now has an explicit expected-strikeout mean plus bookmaker-usable
  over or under ladder probabilities
- the next missing seam is turning that output into real `PropProjection`
  records keyed to sportsbook lines so `edge.py` can rank candidate props

## Constraints For The Next Worktree

- Start the next issue worktree from the current `main` after AGE-148 is
  merged.
- Keep AGE-147 as the only source of the baseline mean expectation.
- Reuse the AGE-148 ladder helpers instead of re-deriving line probabilities in
  a separate pricing or edge module.
- Preserve the current leakage rules:
  - no IDs, target columns, or postgame timestamps in the training matrix
  - no same-day pitch rows in the feature inputs themselves
- Treat `baseline_model.json`, `evaluation.json`, and
  `ladder_probabilities.jsonl` as durable debug artifacts, not throwaway local
  output.

## Open Questions

- A real local smoke run of `train-starter-strikeout-baseline` still needs a
  non-test date span with AGE-146 feature runs already materialized under
  `data/normalized/statcast_search/...`.
- The current distribution fit is global. A future issue may decide to compare
  that against line-bucket or feature-conditioned calibration once the
  `PropProjection` path exists.
- Same-day Statcast outcome pulls still hit Baseball Savant directly for the
  training label. If that path becomes rate-limited in practice, the repo may
  need a cached label-build slice later rather than changing the model contract
  here.
