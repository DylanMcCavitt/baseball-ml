# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current issue branch:
  `dylanmccavitt2015/age-147-build-starter-strikeout-expectation-baseline-model`
- `main` already includes the merged `AGE-143` docs work, `AGE-144` MLB
  metadata ingest, `AGE-145` sportsbook ingest, `AGE-146` Statcast feature
  tables, and this branch adds the first starter strikeout baseline training
  loop for `AGE-147`

## What Was Completed In AGE-147

- `src/mlb_props_stack/modeling.py`
  - added the first model-training seam for starter strikeout expectation
  - loads AGE-146 `pitcher_daily_features`, `lineup_daily_features`, and
    `game_context_features` across an explicit date range
  - joins those feature tables into one training row per
    `official_date / game_pk / pitcher_id`
  - derives the target as official starter strikeouts by pulling same-day
    pitcher Statcast rows and counting strikeout events on final pitches for the
    matching `game_pk`
  - writes raw target pulls under:
    - `data/raw/statcast_search_outcomes/date=YYYY-MM-DD/player_id=.../`
  - writes normalized training artifacts under:
    - `data/normalized/starter_strikeout_baseline/start=YYYY-MM-DD_end=YYYY-MM-DD/run=.../`
  - emits:
    - `training_dataset.jsonl`
    - `starter_outcomes.jsonl`
    - `date_splits.json`
    - `baseline_model.json`
    - `evaluation.json`
  - saves train, validation, and test splits by date instead of random row
    shuffles
  - keeps the training matrix explicit:
    - uses only pregame-valid feature fields
    - excludes IDs, timestamps, and target columns from the fitted feature set
  - adds a naive benchmark:
    - `pitcher_k_rate * expected_leash_batters_faced`
  - fits a deterministic ridge-style linear baseline model in the standard
    library and emits coefficient-based feature importance
  - reports:
    - RMSE
    - MAE
    - Spearman rank correlation
    for benchmark vs. model on train, validation, test, and combined held-out
    rows
- `src/mlb_props_stack/cli.py`
  - added:
    - `train-starter-strikeout-baseline --start-date YYYY-MM-DD --end-date YYYY-MM-DD [--output-dir ...]`
  - prints a training summary with row counts and artifact paths
- `tests/test_modeling.py`
  - added deterministic end-to-end coverage for:
    - feature-run loading across dates
    - same-day Statcast outcome target derivation
    - saved date-based splits
    - model artifact contents
    - held-out improvement over the naive benchmark
- `tests/test_cli.py`
  - added CLI coverage for the new baseline training command
- `README.md`
  - documented the baseline training command and artifact layout
- `docs/architecture.md`
  - documented the new baseline-modeling layer and AGE-147 outputs
- `docs/modeling.md`
  - documented the current benchmark, ridge baseline, date split behavior, and
    no-leakage training-matrix rules

## Verification Run

These commands were run successfully during AGE-147:

```bash
uv sync --extra dev
python3 -m compileall src tests
uv run pytest
uv run python -m mlb_props_stack
```

Local results:

- `uv run pytest`
  - `38 passed`
- `uv run python -m mlb_props_stack`
  - still prints the runtime summary cleanly

Not run during AGE-147:

```bash
uv run python -m mlb_props_stack train-starter-strikeout-baseline \
  --start-date 2026-04-01 \
  --end-date 2026-04-20
```

Reason:

- the repo does not commit live AGE-146 feature runs, so the new training
  command was verified through deterministic end-to-end tests rather than a
  networked local dataset build against live Statcast inputs

## Recommended Next Issue

- `AGE-148` — `Fit strikeout count distribution and ladder probabilities`

Why this should go next:

- AGE-147 now produces an explicit expected-strikeout mean per starter-game row
- AGE-148 can convert that expectation into bookmaker-usable
  `P(K >= line + 0.5)` ladder probabilities without inventing a new feature or
  training seam

## Constraints For The Next Worktree

- Start the next issue worktree from the current `main` after AGE-147 is merged.
- Keep AGE-147 as the only source of the baseline mean expectation:
  - do not duplicate the feature-table join or same-day target pull logic in a
    new distribution module
- Preserve the current leakage rules:
  - no IDs, target columns, or postgame timestamps in the training matrix
  - no same-day pitch rows in the feature inputs themselves
- Treat `evaluation.json` metrics and `baseline_model.json` schema as durable
  debug artifacts, not throwaway local output.
- If AGE-148 needs an additional dependency for count fitting, add it
  deliberately in that issue instead of expanding AGE-147’s stdlib baseline.

## Open Questions

- A real local smoke run of `train-starter-strikeout-baseline` still needs a
  non-test date span with AGE-146 feature runs already materialized under
  `data/normalized/statcast_search/...`.
- The current baseline emits a mean expectation only. Probability calibration
  and line-level over/under probabilities still belong to AGE-148.
- Same-day Statcast outcome pulls currently hit Baseball Savant directly for the
  training label. If that path becomes rate-limited in practice, the repo may
  need a cached label-build slice later rather than changing the model contract
  here.
