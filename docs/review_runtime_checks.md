# Runtime Review Checks

## Why This Exists

Baseline repo checks can stay green while real runtime entrypoints still fail.
That happened on April 22, 2026 after `AGE-153` merged:

- `uv run pytest` and `uv run python -m mlb_props_stack` passed
- `uv run streamlit run src/mlb_props_stack/dashboard/app.py` failed on a
  relative import
- historical `ingest-statcast-features` failed even though the code path was
  supposed to support past-date backfills
- live workflow output still produced empty `daily_candidates` because
  unmatched same-team Odds API events were still being normalized into
  `prop_line_snapshots`, while some true target-date matched events had no
  pitcher-strikeout markets yet
- before `AGE-189`, CI had no dedicated fixture-backed smoke layer for the
  dashboard file path, historical metadata backfill coverage, or training
  entrypoint coverage

Use this document during implementation review, PR review, and final handoff
any time a change touches runtime entrypoints or generated artifacts.

## Baseline Checks

Run these on every issue unless the repo is already blocked before setup:

```bash
uv sync --extra dev
uv run pytest tests/test_runtime_smokes.py
uv run pytest
python3 -m compileall src tests
uv run python -m mlb_props_stack
```

These are necessary. They are not sufficient for dashboard, ingest, training,
or slate-output changes.

The dedicated fixture-backed smoke suite lives in:

```bash
uv run pytest tests/test_runtime_smokes.py
```

That suite now covers:

- the dashboard file-entry path at `src/mlb_props_stack/dashboard/app.py`
- the historical metadata backfill selection used by
  `ingest-statcast-features`
- the seeded `train-starter-strikeout-baseline` runtime path

## Required Runtime Checks By Change Area

### Dashboard Or Streamlit Changes

If a PR touches `src/mlb_props_stack/dashboard/app.py`, dashboard loaders, or
the artifacts it reads:

```bash
uv run pytest tests/test_runtime_smokes.py
uv run streamlit run src/mlb_props_stack/dashboard/app.py
```

Review requirement:

- confirm the app boots instead of throwing an import or startup exception
- confirm the page loads in the browser
- if the page is expected to read artifacts, confirm whether it shows real rows
  or an honest empty-state message

### CLI, Ingest, Modeling, Backtest, Or Paper-Tracking Changes

Run the actual changed command, not just the module entrypoint:

- `uv run pytest tests/test_runtime_smokes.py`
- `ingest-mlb-metadata`
- `ingest-statcast-features`
- `ingest-odds-api-lines`
- `train-starter-strikeout-baseline`
- `build-edge-candidates`
- `build-walk-forward-backtest`
- `build-daily-candidates`

Review requirement:

- use a representative date or seeded fixture path for the changed code path
- if timestamp-valid historical behavior changed, run at least one historical
  date that should succeed without pregame requirements
- if live-slate behavior changed, run a future or clearly pregame slate date
  instead of a date that has already locked

### Artifact-Producing Changes

Do not stop at process exit code. Inspect the outputs.

For training and calibration work, open:

- `evaluation.json`
- `evaluation_summary.json`
- `evaluation_summary.md`
- `calibration_summary.json`
- `raw_vs_calibrated_probabilities.jsonl`
- `ladder_probabilities.jsonl`

For odds, edge, or daily candidate work, inspect:

- CLI summary counts such as `matched_events`, `unmatched_events`,
  `scored_candidates`, and `actionable_candidates`
- `event_game_mappings.jsonl`
- `prop_line_snapshots.jsonl`
- `edge_candidates.jsonl`
- `daily_candidates.jsonl`
- `paper_results.jsonl`

If a workflow is expected to produce scored rows but writes only skipped or
empty outputs, call that out explicitly in the PR and handoff.

## PR Review And Handoff Requirements

Every PR that touches runtime entrypoints or artifact contracts must record:

- whether `uv run pytest tests/test_runtime_smokes.py` passed
- the exact commands that were run
- the exact dates or fixture inputs used
- whether the outputs were non-empty, partially matched, or intentionally empty
- any external-source caveats such as upstream `502` or `503` responses

Do not write "all checks passed" if only the baseline repo checks ran.

## Current Known Gaps

- future-slate runs can still honestly produce zero scored candidates when the
  matched target-date Odds API events return `bookmakers: []` or otherwise lack
  pitcher-strikeout markets at capture time; treat
  `matched_events_without_props` as an availability signal before assuming a
  join regression
- `AGE-189`
  Missing fixture-backed runtime smoke coverage for dashboard boot and
  representative pipeline entrypoints
