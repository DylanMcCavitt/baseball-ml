# Next Session Handoff

## Status

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Active issue: `AGE-294` - rebuild betting layer against calibrated strikeout
  distributions
- Current issue branch:
  `dylanmccavitt2015/age-294-rebuild-betting-layer-against-calibrated-strikeout`
- Base state: branch created from `origin/main` at the start of this issue
  after `git fetch origin --prune`.
- PR: pending creation after commit and push.
- Implementation state: code, docs, focused tests, full tests, compile check,
  module smoke, and fixture-backed edge CLI verification are complete.

## What Changed In AGE-294

- Added `src/mlb_props_stack/betting.py` for rebuilt betting-layer helpers:
  exact line pricing from full strikeout count distributions, validation-report
  discovery, validation-derived approval gates, confidence buckets, and line
  buckets.
- Updated `build-edge-candidates` so it prefers rebuilt
  `candidate_strikeout_models/.../model_outputs.jsonl` runs when available,
  while preserving the legacy `starter_strikeout_baseline` ladder path.
- Rebuilt edge rows now include model projection, full probability
  distribution, model confidence, no-vig market probabilities, selected edge,
  validation evidence, approval status, and approval/rejection reason.
- Wager approval is blocked unless the latest model-only validation report says
  `conditional_go_for_betting_layer_rebuild` and has
  `thresholds_observed_from_calibration`.
- Approval thresholds now come from validation evidence:
  `validation_min_edge_pct` cannot be lower than the observed confidence-bucket
  calibration error, and the row confidence bucket must be one of the
  validation-qualified buckets.
- Added pitcher/game/line duplicate grouping for rebuilt distribution rows.
  Only the top row in a correlated group can remain approved; duplicate rows
  are preserved with rejection reasons.
- Extended candidate model output rows with `feature_row_id`,
  `lineup_snapshot_id`, `features_as_of`, `projection_generated_at`, and
  `model_input_refs` so betting-layer timing checks are auditable.
- Updated `docs/architecture.md` and `docs/modeling.md` with the AGE-294
  betting-layer contract.

## Runtime Evidence

- Fixture-backed CLI output directory:
  `/tmp/age294-edge-runtime`
- Command:

```bash
/opt/homebrew/bin/uv run python -m mlb_props_stack build-edge-candidates --date 2026-04-20 --output-dir /tmp/age294-edge-runtime --model-run-dir /tmp/age294-edge-runtime/normalized/candidate_strikeout_models/start=2026-04-16_end=2026-04-20/run=20260420T150000Z
```

- CLI result:
  `line_snapshots=2`, `scored_lines=2`, `actionable_candidates=2`,
  `approved_wagers=1`, `rejected_wagers=1`.
- Inspected artifact:
  `/tmp/age294-edge-runtime/normalized/edge_candidates/date=2026-04-20/run=20260428T011046Z/edge_candidates.jsonl`
  - `line-draftkings`: approved, `model_projection=6.9`,
    `model_over_probability=0.6`, `market_over_probability=0.5`,
    `edge_pct=0.1`, `correlation_group_rank=1`.
  - `line-fanduel`: rejected as a correlated duplicate in the same
    pitcher/game/line group, `correlation_group_rank=2`.

## Files Changed

- `src/mlb_props_stack/betting.py`
- `src/mlb_props_stack/edge.py`
- `src/mlb_props_stack/candidate_models.py`
- `src/mlb_props_stack/cli.py`
- `tests/test_edge.py`
- `tests/test_candidate_models.py`
- `docs/architecture.md`
- `docs/modeling.md`
- `docs/NEXT_SESSION_HANDOFF.md`

## Verification

Commands run:

```bash
/opt/homebrew/bin/uv sync --extra dev
/opt/homebrew/bin/uv run pytest tests/test_edge.py -q
/opt/homebrew/bin/uv run pytest tests/test_candidate_models.py tests/test_model_validation.py -q
/opt/homebrew/bin/uv run pytest tests/test_backtest.py tests/test_edge.py tests/test_runtime_smokes.py -q
/opt/homebrew/bin/uv run pytest tests/test_candidate_models.py tests/test_edge.py -q
/opt/homebrew/bin/uv run pytest
PYTHONPYCACHEPREFIX=/tmp/age294-pycache python3 -m compileall src tests
/opt/homebrew/bin/uv run python -m mlb_props_stack
/opt/homebrew/bin/uv run python -m mlb_props_stack build-edge-candidates --date 2026-04-20 --output-dir /tmp/age294-edge-runtime --model-run-dir /tmp/age294-edge-runtime/normalized/candidate_strikeout_models/start=2026-04-16_end=2026-04-20/run=20260420T150000Z
git diff --check
```

Observed results:

- `uv sync --extra dev`: completed successfully.
- Focused edge tests: `14 passed`.
- Candidate/model-validation focused tests: `3 passed`.
- Backtest/edge/runtime smoke focused tests: `24 passed` with existing
  third-party MLflow/Pydantic warnings.
- Candidate + edge focused tests after contract assertions: `16 passed`.
- Full suite: `226 passed` with existing third-party MLflow/Pydantic warnings.
- `compileall`: completed successfully.
- `python -m mlb_props_stack`: printed the runtime configuration banner.
- Fixture-backed `build-edge-candidates`: completed successfully and produced
  the expected approved/rejected distribution rows.
- `git diff --check`: no whitespace errors.

## Recommended Next Issue

1. Backfill or restore real historical Odds API strikeout-line artifacts plus
   compatible rebuilt candidate-model and model-only validation artifacts.
2. Run `build-edge-candidates` against a real covered date and confirm approval
   remains blocked unless validation evidence is present and green.
3. Extend `build-walk-forward-backtest` to consume rebuilt distribution outputs
   directly once real scoreable market coverage exists for a representative
   historical window.

## Constraints And Risks

- Do not treat the `/tmp/age294-edge-runtime` fixture as betting evidence. It
  proves the rebuilt betting-layer code path and artifact shape only.
- Real season/date betting-layer validation is still blocked by the handoff
  caveat from AGE-293: this worktree and the canonical checkout did not have
  restored real historical Odds API market artifacts available.
- Do not loosen approval gates to force output. Missing validation, no-go
  validation, missing projection timestamps, and correlated duplicate lines
  remain explicit rejections.
- CLV and ROI reporting remain separated in the walk-forward backtest artifacts;
  they are not used to tune the projection model or validation-derived approval
  thresholds.
