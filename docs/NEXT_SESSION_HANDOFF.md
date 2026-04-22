# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current issue branch:
  `dylanmccavitt2015/age-153-add-daily-candidate-generation-and-paper-tracking-workflow`
- Canonical local `main` was fast-forwarded from `origin/main` before this
  worktree was created and was at `bfbde917b5536f39f23eb99dd26e6169e7364482`
- `main` already includes the merged `AGE-143` docs work, `AGE-144` MLB
  metadata ingest, `AGE-145` sportsbook ingest, `AGE-146` Statcast feature
  tables, `AGE-147` starter strikeout baseline mean training, `AGE-148`
  count-distribution ladder probabilities, `AGE-149` probability calibration
  diagnostics, `AGE-150` replayable edge-candidate pricing rows, `AGE-151`
  walk-forward backtest joins, and `AGE-152` CLV / ROI / edge-bucket reporting
- This branch adds `AGE-153`: daily candidate generation, target-date
  inference, cumulative paper-result refreshes, and the first Streamlit slate
  review page

## What Was Completed In AGE-153

- `src/mlb_props_stack/modeling.py`
  - adds `generate_starter_strikeout_inference_for_date()`
  - selects the latest historical baseline run whose coverage still ends before
    the requested target date
  - reconstructs the saved linear baseline plus calibrator from
    `baseline_model.json`
  - writes target-date inference artifacts under
    `data/normalized/starter_strikeout_inference/...`
- `src/mlb_props_stack/paper_tracking.py`
  - adds `build_daily_candidate_workflow()`
  - runs target-date inference, scores the current slate against the latest
    line snapshots, writes `daily_candidates.jsonl`, and refreshes cumulative
    `paper_results.jsonl`
  - keeps `daily_candidates` as the scored slate sheet and `paper_results` as
    actionable paper bets only
- `src/mlb_props_stack/cli.py`
  - adds `build-daily-candidates`
  - defaults `--date` to today in the configured stack timezone when omitted
  - prints the new `daily_candidates` and `paper_results` output paths
- `src/mlb_props_stack/dashboard/app.py`
  - replaces the placeholder banner with the first Streamlit page
  - renders current-slate candidates plus recent paper performance from the
    saved artifacts
- `pyproject.toml`
  - adds Streamlit to the `dev` extra so the documented local dashboard command
    works after `uv sync --extra dev`
- `README.md`, `docs/architecture.md`, `docs/modeling.md`
  - document the new inference, daily candidate, paper-result, and dashboard
    workflow
- `tests/test_cli.py`, `tests/test_paper_tracking.py`, `tests/test_tracking.py`
  - lock the new CLI surface, inference / paper-tracking workflow, and
    dashboard rendering path

## Verification Run

These commands were run successfully during AGE-153:

```bash
uv sync --extra dev
uv run pytest
python3 -m compileall src tests
uv run python -m mlb_props_stack
```

Local results:

- `uv run pytest`
  - `53 passed`
- `python3 -m compileall src tests`
  - compiled all source and test modules successfully
- `uv run python -m mlb_props_stack`
  - still prints the runtime summary cleanly
- focused workflow tests
  - verified target-date inference chooses the latest non-leaky saved baseline
    run
  - verified `paper_results` refreshes prior actionable sheets into settled vs
    pending paper bets
  - verified the Streamlit page reads `daily_candidates` and `paper_results`

## Recommended Next Issue

- Handle moved-point closing-line references when the exact strikeout line
  disappears near first pitch

Why this should go next:

- `AGE-153` now makes paper tracking visible in both artifacts and the
  dashboard, but CLV is still missing whenever the book moves from one exact
  strikeout number to another
- the current sheet and paper workflow already preserve the precise decision
  line, so the next slice can improve closing-line reference quality without
  changing the daily candidate contract

## Constraints For The Next Worktree

- Start the next issue worktree from the current `main` after `AGE-153` is
  merged.
- Keep target-date inference honest:
  - source model runs must still end before the requested slate date
  - do not let calibration metadata or ladder probabilities leak from dates on
    or after the evaluated slate
- Preserve the artifact split:
  - `starter_strikeout_inference` for target-date model output
  - `daily_candidates` for the ranked scored sheet
  - `paper_results` for actionable paper bets only
- Keep `projection_generated_at` conservative against the quoted line snapshot.
  The branch now stores `inference_generated_at` separately for the actual run
  time.
- Keep the Streamlit page lightweight and artifact-backed. Do not turn it into
  a scheduler, notebook replacement, or execution bot in the next slice.

## Open Questions

- How should moved-line closing references be normalized when the paper bet was
  at `5.5` but the latest close only exists at `6.5`?
- Should a future inference slice persist a contract-valid per-projection
  `generated_at` that can still be proven no later than the quoted line
  snapshot, instead of keeping the current conservative
  `projection_generated_at = features_as_of` default?
