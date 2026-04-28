---
tracker:
  kind: linear
  api_key: $LINEAR_API_KEY
  project_slug: mlb-props-reboot-b9ac56339eba
  active_states:
    - Ready
    - In Progress
    - Needs Fixes
    - Merging
  terminal_states:
    - Done
    - Canceled
    - Cancelled
    - Duplicate
polling:
  interval_ms: 30000
workspace:
  root: /Users/dylanmccavitt/.codex/symphony-workspaces/mlb-props-reboot
hooks:
  after_create: |
    git clone git@github.com:DylanMcCavitt/baseball-ml.git .
  timeout_ms: 120000
agent:
  max_concurrent_agents: 2
  max_turns: 20
  max_retry_backoff_ms: 300000
codex:
  command: /opt/homebrew/bin/codex --config shell_environment_policy.inherit=all app-server
  approval_policy: never
  thread_sandbox: danger-full-access
  turn_sandbox_policy:
    type: dangerFullAccess
---

You are working on the MLB Props Reboot Linear issue `{{ issue.identifier }}`.

Title: {{ issue.title }}
Status: {{ issue.state }}
URL: {{ issue.url }}

Description:
{% if issue.description %}
{{ issue.description }}
{% else %}
No description provided.
{% endif %}

## Project Contract

This is an end-to-end pitcher strikeout props system. Every implementation must
keep the path visible and testable:

data -> normalization -> features -> distribution model -> market comparison ->
paper recommendations -> dashboard/report review.

## Required Flow

1. Read the issue and update the single Linear `## Codex Workpad`.
2. Move `Ready` issues to `In Progress` before code changes.
3. Create or reuse the issue branch from current `origin/main`.
4. Reproduce or inspect the current behavior before changing code.
5. Implement the smallest issue-sized slice.
6. Run targeted checks plus baseline checks.
7. Generate any required report artifacts under `artifacts/reports/<ISSUE>/`.
8. Update `docs/reviews/<ISSUE>.md` with reviewer-facing evidence.
9. Open or update the PR and link it to Linear.
10. Move the issue to `Human Review` only after checks and report evidence pass.
11. In `Needs Fixes`, address all PR feedback before returning to `Human Review`.
12. In `Merging`, merge only after human approval and green checks, then move to `Done`.

## Feature Guardrails

- Do not train on unregistered model features.
- Do not use post-game information for pre-game predictions.
- Do not hide missing coverage; rejected/skipped rows are part of the report.
- Paper recommendations are allowed; real-money betting automation is out of scope.
- Visual evidence is required for data, normalization, feature, model, and market-report issues.
