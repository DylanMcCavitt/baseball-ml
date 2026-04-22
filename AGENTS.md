# AGENTS.md

This file is the repo-local operating guide for agents working in this project.
It supplements the shared workflow in `~/.agent-workflow-kit/README.md`.
If this file conflicts with generic guidance, follow this file for work in this
repository.

## Project Identity

- The folder is named `nba-ml`, but the current codebase and product scope are
  `mlb-props-stack`.
- This repository is a Python 3.11+ MLB props modeling scaffold focused on
  narrow, measurable sportsbook markets.
- V1 is intentionally limited to `pitcher strikeout props`.

## Project Scope

The goal of this repo is to build an honest MLB prop evaluation system that:

- estimates event probabilities
- compares those probabilities to sportsbook prices
- ranks edges after vig and sizing logic
- backtests only with information that would have existed at bet time

Current in-scope work includes:

- prop, projection, pricing, and backtest contracts
- modeling inputs and feature definitions for pitcher strikeout markets
- edge detection and candidate ranking
- walk-forward evaluation
- clean seams for future experiment tracking and dashboarding

Current non-goals unless the issue explicitly says otherwise:

- full-game sides or totals
- same-game parlay optimization
- live betting or execution bots
- reinforcement learning for the initial prediction layer
- broad multi-sport expansion
- dependency sprawl during bootstrap slices

## Primary Repo Sources

Read these before changing architecture or issue sequencing:

- `README.md`
- `docs/architecture.md`
- `docs/modeling.md`
- `docs/NEXT_SESSION_HANDOFF.md`

## Required Workflow Rules

- One issue -> one worktree -> one branch -> one PR.
- Start each issue from an explicit issue packet or equivalent scope with:
  scope, non-goals, acceptance criteria, constraints, and verification commands.
- Non-UI and backend-heavy slices default to Codex. UI-heavy dashboard work can
  go to Claude if the user wants that routing.
- Keep PRs reviewable. Target small, focused slices instead of broad mixed work.
- Never merge unless the branch is up to date with `main` and CI is green.

## Handoff Requirement For Every New Issue

At each new worktree or issue start, there must already be a handoff document in
place from the last issue.

- The required artifact in this repo is `docs/NEXT_SESSION_HANDOFF.md`.
- If that handoff is missing, stale, or clearly does not reflect the last
  completed issue, stop and update it before starting new implementation work.
- Do not treat chat context alone as sufficient continuity. The repo must carry
  the handoff artifact forward.
- New work should begin by reading the handoff, then the relevant issue packet,
  then the touched code and tests.

## End-Of-Issue Handoff Rules

Before considering an issue complete, update `docs/NEXT_SESSION_HANDOFF.md`
with:

- current repo and branch state
- what was completed in the issue
- files or modules that changed
- verification commands that were run and their outcome
- the recommended next issue
- open questions, blockers, or constraints the next worktree must respect

If a worktree closes without a useful handoff document, the issue is not fully
handed off.

## Repo Guardrails

- Preserve the standard-library-first baseline unless the issue explicitly
  requires new dependencies.
- Keep `python -m mlb_props_stack` working.
- Keep tests runnable with the documented repo command set.
- Preserve the reserved future seams:
  - `src/mlb_props_stack/tracking.py` for tracking configuration
  - `src/mlb_props_stack/dashboard/app.py` for the future dashboard entrypoint
- Do not add MLflow, Streamlit, Plotly, ingestion systems, schedulers, or
  storage layers casually. Add them only when the issue scope actually requires
  them.

## Modeling And Backtest Guardrails

- Favor honest, timestamp-valid modeling over feature breadth.
- Do not use information that would not have been available at bet time.
- Keep pricing logic, calibration, and evaluation explicit rather than implied.
- Treat CLV and ROI as separate signals.
- Avoid vague "predict baseball" framing. This repo is about a narrow decision
  system for pitcher strikeout props.

## Code Map

- `src/mlb_props_stack/config.py`
  Runtime settings and model defaults.
- `src/mlb_props_stack/tracking.py`
  Reserved tracking seam.
- `src/mlb_props_stack/pricing.py`
  Odds conversion, devig, expected value, fair odds, and Kelly helpers.
- `src/mlb_props_stack/markets.py`
  Core data contracts for props, projections, and decisions.
- `src/mlb_props_stack/edge.py`
  Edge detection and ranking.
- `src/mlb_props_stack/backtest.py`
  Backtest policy and evaluation rules.
- `src/mlb_props_stack/dashboard/app.py`
  Placeholder dashboard entrypoint.
- `tests/`
  Keep behavior locked with targeted, issue-scoped tests.

## Verification

Preferred commands:

```bash
uv sync --extra dev
uv run pytest
uv run python -m mlb_props_stack
```

Fallback if using a virtualenv instead of `uv`:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m mlb_props_stack
```

If you change contracts, pricing logic, CLI behavior, or backtest rules, add or
update focused tests in the same issue.

If you change contracts, pricing logic, CLI behavior, backtest rules, ingest
logic, modeling paths, paper tracking, or dashboard behavior:

- run the baseline repo checks above
- run the affected runtime checks from `docs/review_runtime_checks.md`
- verify the produced artifacts or the loaded UI, not just the process exit
  code

Do not mark an issue or PR as verified based only on `uv run pytest` and
`uv run python -m mlb_props_stack` when the changed code lives behind a
different runtime entrypoint.

## Documentation Expectations

- Keep docs operational and specific.
- When scope changes, update the relevant docs in the same issue.
- Avoid aspirational architecture text that is not connected to the current
  roadmap or issue sequence.
