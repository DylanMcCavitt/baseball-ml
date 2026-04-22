# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current issue branch:
  `dylanmccavitt2015/age-150-implement-no-vig-pricing-ev-and-conservative-sizing`
- `main` already includes the merged `AGE-143` docs work, `AGE-144` MLB
  metadata ingest, `AGE-145` sportsbook ingest, `AGE-146` Statcast feature
  tables, `AGE-147` starter strikeout baseline mean training, `AGE-148`
  count-distribution ladder probabilities, and `AGE-149` probability
  calibration diagnostics
- This branch adds `AGE-150`: reusable no-vig pricing and conservative sizing
  logic plus the first replayable `edge_candidates` artifact

## What Was Completed In AGE-150

- `src/mlb_props_stack/pricing.py`
  - keeps the existing odds-conversion helpers
  - adds reusable `fractional_kelly()` and `capped_fractional_kelly()` helpers
    so sizing logic is explicit instead of being embedded inside edge code
  - preserves `quarter_kelly()` as a compatibility wrapper
- `src/mlb_props_stack/edge.py`
  - adds `analyze_projection()` so pricing details can be preserved even when a
    prop does not clear the configured edge threshold
  - keeps `evaluate_projection()` returning only actionable `EdgeDecision`
    results
  - adds `build_edge_candidates_for_date()` which:
    - loads the latest AGE-145 odds run for one official date
    - loads the latest AGE-149 model run containing that date
    - materializes contract-valid `PropProjection` objects for exact book lines
    - writes `data/normalized/edge_candidates/date=YYYY-MM-DD/run=.../edge_candidates.jsonl`
    - preserves actionable, below-threshold, training-split, and skipped rows
      with explicit statuses and reasons
- `src/mlb_props_stack/modeling.py`
  - extends saved `ladder_probabilities.jsonl` rows with:
    - `feature_row_id`
    - `lineup_snapshot_id`
    - `features_as_of`
    - `projection_generated_at`
  - uses `features_as_of` as the conservative historical
    `projection_generated_at` until a dedicated inference snapshot exists
- `src/mlb_props_stack/markets.py`
  - extends `PropLine` with `line_snapshot_id` so saved edge rows can be keyed
    to the exact market snapshot they came from
- `src/mlb_props_stack/cli.py`
  - adds `build-edge-candidates --date YYYY-MM-DD [--model-run-dir ...]`
  - renders summary counts for scored, actionable, below-threshold, and skipped
    rows
- `tests/`
  - adds pricing coverage for devig symmetry, negative-edge Kelly behavior, and
    capped sizing
  - adds edge coverage for:
    - preserved below-threshold rows
    - capped Kelly sizing inside projection analysis
    - edge-candidate artifact generation across actionable, below-threshold,
      missing-projection, and missing-lineup cases
  - extends modeling coverage so saved ladder rows keep the new projection input
    refs and timestamps
- `README.md`, `docs/architecture.md`, `docs/modeling.md`
  - document the new `build-edge-candidates` command
  - document the `edge_candidates.jsonl` artifact
  - document the conservative historical projection timestamp rule

## Verification Run

These commands were run successfully during AGE-150:

```bash
uv sync --extra dev
python3 -m compileall src tests
uv run pytest
uv run python -m mlb_props_stack
```

Local results:

- `uv run pytest`
  - `47 passed`
- `uv run python -m mlb_props_stack`
  - still prints the runtime summary cleanly

## Recommended Next Issue

- Build the first walk-forward backtest slice on top of saved
  `edge_candidates.jsonl` rows

Why this should go next:

- the repo can now compare calibrated model probabilities to actual book
  snapshots and preserve rejected rows for audit
- the next missing seam is turning those saved decision rows into
  walk-forward CLV and ROI reporting instead of only per-line pricing outputs

## Constraints For The Next Worktree

- Start the next issue worktree from the current `main` after AGE-150 is
  merged.
- Keep `line_snapshot_id` as the stable key for replayable market history.
- Reuse the saved calibrated ladder probabilities from AGE-149 instead of
  recalibrating inside backtest code.
- Preserve the current historical timestamp rule:
  - `features_as_of <= projection_generated_at <= line.captured_at`
- Preserve skipped or below-threshold rows so later threshold changes can be
  audited against the original market history.

## Open Questions

- `projection_generated_at` currently defaults to `features_as_of` for
  historical edge builds. A future issue may choose to persist a distinct live
  inference timestamp once a dedicated pregame projection artifact exists.
- Lines without a mapped `game_pk`, a mapped pitcher ID, or a saved
  `lineup_snapshot_id` are preserved as skipped rows today. A future issue may
  decide whether projected-lineup artifacts should fill more of those gaps, but
  this slice intentionally did not weaken the input-ref contract.
