# AGENTS.md

This repository is the clean reboot of the MLB pitcher strikeout props project.
The product goal is an end-to-end, reviewable decision system:

1. build timestamp-valid pitcher strikeout training data
2. train and evaluate a strikeout distribution model
3. compare model probabilities to sportsbook pitcher strikeout prices
4. produce paper recommendations until evidence gates justify live use
5. make every step visually reviewable through report artifacts and dashboards

## Workflow

- One issue = one branch = one worktree = one PR.
- Start from the Linear issue and the current `WORKFLOW.md`.
- Keep issue progress in the Linear `## Codex Workpad`.
- Do not expand scope when a better idea appears; create a follow-up issue.
- Before `Human Review`, provide a PR, checks, report paths, and reviewer-facing evidence.

## Commands

```bash
uv sync --extra dev
uv run pytest
uv run ruff check
uv run python -m mlb_props_lab feature-registry validate
uv run python -m mlb_props_lab targets build --issue TARGET-SAMPLE
uv run python -m mlb_props_lab statcast build-features --issue STATCAST-SAMPLE
uv run python -m mlb_props_lab report features --issue FEATURE-RESEARCH
uv run python -m mlb_props_lab dashboard
```

## Feature Rules

- Every model feature must be registered before model training consumes it.
- Every registered feature needs source fields, formula, timestamp cutoff,
  missing-value policy, leakage risk, and a required visual.
- Use only information available before the game or odds snapshot being scored.
- Keep post-game target labels out of pre-game feature fields and reports.
- Source-backed does not mean model-approved. Features must survive walk-forward
  validation, ablation, calibration checks, and out-of-time stability checks.

## Definition Of Done

- Tests and lint pass for touched code.
- Report commands touched by the issue run successfully.
- Any changed feature family has visual evidence in a generated report.
- PR description links to the report path and summarizes limitations.
- No real-money betting language is used before evidence gates are passed.
