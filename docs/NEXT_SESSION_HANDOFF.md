# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Current issue branch: `feat/runtime-review-guardrails`
- `main` already includes the merged `AGE-143` docs work, `AGE-144` MLB
  metadata ingest, `AGE-145` sportsbook ingest, `AGE-146` Statcast feature
  tables, `AGE-147` starter strikeout baseline mean training, `AGE-148`
  count-distribution ladder probabilities, `AGE-149` probability calibration
  diagnostics, `AGE-150` replayable edge-candidate pricing rows, `AGE-151`
  walk-forward backtest joins, `AGE-152` CLV / ROI / edge-bucket reporting, and
  `AGE-153` daily candidate generation plus the first Streamlit slate review
  page
- This branch captures the post-merge runtime review follow-up: proven bug
  fixes from the live AGE-153 repro plus stricter runtime review guidance for
  future PRs and handoffs

## What Was Completed In This Follow-Up

- `src/mlb_props_stack/dashboard/app.py`
  - switches the Streamlit page to an absolute package import so
    `uv run streamlit run ...` boots cleanly
- `src/mlb_props_stack/ingest/statcast_features.py`
  - lets historical feature backfills fall back to the latest complete MLB
    metadata run when the target date is already in the past
  - keeps the stricter pregame-valid requirement for current or future slate
    runs
- `src/mlb_props_stack/modeling.py`
  - skips scratched or otherwise non-starting probable-starter rows whose
    same-game starter outcome cannot honestly be derived, instead of aborting
    the whole baseline training run
- `src/mlb_props_stack/ingest/odds_api.py`
  - tolerates small commence-time drift when matching Odds API events back to
    MLB games
- `tests/test_odds_api_ingest.py`,
  `tests/test_statcast_feature_ingest.py`,
  `tests/test_modeling.py`
  - add regression coverage for the live repro paths above
- `docs/review_runtime_checks.md`
  - records the repo-level runtime review checklist that future thread workers
    and PR reviewers must use when runtime entrypoints change
- `.github/pull_request_template.md`
  - forces PRs to state which runtime checks and outputs were actually
    exercised
- `AGENTS.md`, `README.md`
  - make it explicit that baseline repo checks are necessary but not sufficient
    for runtime-facing changes
- Linear follow-up issues opened:
  - `AGE-188`
    remaining Odds API player/game join gap that still leaves live
    `daily_candidates` empty on some future slates
  - `AGE-189`
    fixture-backed runtime smoke coverage so CI and local review can catch
    dashboard/pipeline failures before merge

## What AGE-153 Looked Like In Live Review

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

These commands were run successfully during the post-merge runtime review:

```bash
uv sync --extra dev
python3 -m compileall src tests
uv run pytest tests/test_modeling.py tests/test_odds_api_ingest.py tests/test_statcast_feature_ingest.py tests/test_paper_tracking.py tests/test_tracking.py
uv run python -m mlb_props_stack
uv run streamlit run src/mlb_props_stack/dashboard/app.py
```

Local results:

- `python3 -m compileall src tests`
  - compiled all source and test modules successfully
- focused pytest run
  - `18 passed`
- `uv run python -m mlb_props_stack`
  - still prints the runtime summary cleanly
- `uv run streamlit run src/mlb_props_stack/dashboard/app.py`
  - the dashboard booted successfully after the import fix
- live historical run verification on April 22, 2026
  - `ingest-statcast-features --date 2026-04-18` through `2026-04-21`
    succeeded after allowing historical metadata fallback
  - `train-starter-strikeout-baseline --start-date 2026-04-18 --end-date 2026-04-21`
    succeeded after skipping missing same-game outcome rows
  - `build-daily-candidates --date 2026-04-23` completed end to end, but the
    final sheet stayed empty because the remaining odds/player join gap still
    dropped all line snapshots to `missing_join_keys`

## Recommended Next Issue

- `AGE-188`
  Fix unresolved Odds API joins so daily candidates can score the slate

Why this should go next:

- live runtime review proved that the current daily workflow can still finish
  with zero scored rows even when some Odds API events match at the event layer
- that gap is the remaining blocker between a booting dashboard and a useful
  non-empty slate review page

## Constraints For The Next Worktree

- Start the next issue worktree from the current `main`, then carry forward the
  runtime review guidance from `docs/review_runtime_checks.md`.
- Keep target-date inference honest:
  - source model runs must still end before the requested slate date
  - do not let calibration metadata or ladder probabilities leak from dates on
    or after the evaluated slate
- Keep the stronger historical-vs-live metadata rule:
  - historical feature backfills may use the latest complete metadata run
  - current or future slates must still require a pregame-valid metadata run
- Preserve the artifact split:
  - `starter_strikeout_inference` for target-date model output
  - `daily_candidates` for the ranked scored sheet
  - `paper_results` for actionable paper bets only
- Keep `projection_generated_at` conservative against the quoted line snapshot.
  The branch now stores `inference_generated_at` separately for the actual run
  time.
- Keep the Streamlit page lightweight and artifact-backed. Do not turn it into
  a scheduler, notebook replacement, or execution bot in the next slice.
- For any runtime-facing PR, record the exact commands, dates, and inspected
  artifacts instead of writing only "all checks passed."

## Open Questions

- Why do matched Odds API events still yield only unmatched
  `prop_line_snapshots` for some future slates, leaving `game_pk` and
  `pitcher_mlb_id` empty?
- How should moved-line closing references be normalized when the paper bet was
  at `5.5` but the latest close only exists at `6.5`?
- Should `AGE-189` turn parts of `docs/review_runtime_checks.md` into
  deterministic fixture-backed CI smoke tests for dashboard boot and
  representative pipeline commands?
