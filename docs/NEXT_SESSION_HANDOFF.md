# Next Session Handoff

## Status

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Completed issue: `AGE-295` - reconnect dashboard and approved-wager UX after
  model gates pass
- Current issue branch:
  `dylanmccavitt2015/age-295-reconnect-dashboard-and-approved-wager-ux-after-model-gates`
- Base state: branch created from `origin/main` after `git fetch origin --prune`.
- PR: https://github.com/DylanMcCavitt/baseball-ml/pull/54
- Implementation state: code, docs, focused tests, full tests, compile check,
  module smoke, fixture-backed dashboard/browser check, wager-card CLI smoke,
  and whitespace check are complete.

## What Changed In AGE-295

- Reconnected dashboard board loading to rebuilt `edge_candidates` rows without
  re-applying legacy dashboard approval settings over rebuilt betting-layer
  decisions.
- Rebuilt edge rows now carry feature-group contribution summaries from the
  selected candidate model, confidence bucket, and `research_only` readiness
  status for operator reporting.
- Dashboard board rows preserve rejected rebuilt candidates by default and show
  projected strikeouts, line-specific model/market probabilities, edge, stake,
  approval or rejection reason, and duplicate/correlated line grouping.
- Pitcher detail can resolve rebuilt distribution model outputs, render the
  actual strikeout PMF, show projected strikeouts, over/under probabilities,
  calibration bucket, approval status, feature driver groups, validation state,
  correlation-group rank, and research-only status.
- Fixed real Streamlit query-param parsing so direct links to the pitcher detail
  screen work when browser runtime query params arrive as list values.
- `build-wager-card` can fall back to rebuilt `edge_candidates` when no
  `daily_candidates` sheet exists for the date. Approved rows come only from
  `approval_status=approved` plus `approval_allowed=true`; rejected rows remain
  blocked diagnostics when `--include-rejected` is used.
- Updated README, architecture, and modeling docs with the AGE-295 reporting
  contract and research-only caveat.

## Runtime Evidence

- Fixture-backed dashboard data directory:
  `/tmp/age295-dashboard-runtime`
- Streamlit runtime URL checked in Chrome:
  `http://127.0.0.1:8518/?screen=pitcher&board_date=2026-04-20&pitcher_id=mlb-pitcher%3A700001`
- Browser/UI result:
  - page loaded from the fixture-backed Streamlit server
  - ticker showed `RESEARCH ONLY`
  - pitcher detail showed `STRIKEOUT PMF`, projected K `6.90`, line-specific
    over/under probabilities `60.0% / 40.0%`, calibration bucket `0.6_to_0.7`,
    approval status `approved`, feature driver groups
    `pitcher_skill`, `matchup`, `workload`, and `context`, and guardrails with
    validation/correlation/research-readiness state
- Wager-card CLI fixture result:
  - command read `edge_candidates`
  - `approved_wagers=1`
  - `blocked_candidates=1`
  - blocked row stayed in the diagnostic section with the correlated-duplicate
    rejection reason

## Files Changed

- `README.md`
- `docs/architecture.md`
- `docs/modeling.md`
- `docs/NEXT_SESSION_HANDOFF.md`
- `src/mlb_props_stack/dashboard/app.py`
- `src/mlb_props_stack/dashboard/lib/data.py`
- `src/mlb_props_stack/dashboard/screens/board.py`
- `src/mlb_props_stack/dashboard/screens/pitcher.py`
- `src/mlb_props_stack/edge.py`
- `src/mlb_props_stack/wager_card.py`
- `tests/test_dashboard_app.py`
- `tests/test_dashboard_data.py`
- `tests/test_wager_card.py`

## Verification

Commands run:

```bash
/opt/homebrew/bin/uv sync --extra dev
/opt/homebrew/bin/uv run pytest tests/test_dashboard_data.py tests/test_wager_card.py tests/test_edge.py -q
/opt/homebrew/bin/uv run pytest tests/test_dashboard_data.py tests/test_wager_card.py tests/test_edge.py tests/test_runtime_smokes.py -q
/opt/homebrew/bin/uv run pytest tests/test_dashboard_app.py tests/test_dashboard_data.py tests/test_wager_card.py tests/test_edge.py tests/test_runtime_smokes.py -q
/opt/homebrew/bin/uv run pytest tests/test_wager_card.py tests/test_dashboard_app.py tests/test_dashboard_data.py -q
/opt/homebrew/bin/uv run pytest
PYTHONPYCACHEPREFIX=/tmp/age295-pycache python3 -m compileall src tests
/opt/homebrew/bin/uv run python -m mlb_props_stack
MLB_PROPS_STACK_DATA_DIR=/tmp/age295-dashboard-runtime /opt/homebrew/bin/uv run streamlit run src/mlb_props_stack/dashboard/app.py --server.headless true --server.address 127.0.0.1 --server.port 8518
curl -fsS http://127.0.0.1:8518/_stcore/health
/opt/homebrew/bin/uv run python -m mlb_props_stack build-wager-card --date 2026-04-20 --output-dir /tmp/age295-dashboard-runtime --include-rejected
git diff --check
```

Observed results:

- `uv sync --extra dev`: completed successfully.
- Initial focused dashboard/wager-card/edge tests: `27 passed`.
- Focused runtime smoke expansion: `32 passed` with existing third-party
  MLflow/Pydantic warnings.
- Final focused dashboard/query-param/wager-card/runtime tests: `36 passed`
  with existing third-party MLflow/Pydantic warnings.
- Wager-card/dashboard focused tests after stake-unit polish: `17 passed`.
- Full suite: `229 passed` with existing third-party MLflow/Pydantic warnings.
- `compileall`: completed successfully.
- `python -m mlb_props_stack`: printed the runtime configuration banner.
- Streamlit health endpoint: returned `ok`.
- Browser/UI check: Chrome loaded the fixture-backed pitcher detail page and
  exposed projection, distribution, probabilities, feature drivers, approval
  status, correlation group, and research-only status.
- Fixture-backed `build-wager-card`: completed successfully and wrote
  `/tmp/age295-dashboard-runtime/normalized/wager_card/date=2026-04-20/run=20260428T144650Z/wager_card.jsonl`.
- `git diff --check`: no whitespace errors.

## Recommended Next Issue

1. Restore or backfill real historical Odds API strikeout-line artifacts plus
   compatible rebuilt model and model-only validation artifacts.
2. Run `build-edge-candidates`, the dashboard, and `build-wager-card` against a
   real covered date to confirm approvals remain blocked unless validation and
   betting-layer gates both pass.
3. Extend walk-forward backtest and reporting once real scoreable exact-line
   coverage is available for a representative historical window.

## Constraints And Risks

- The stack remains `research_only`. AGE-295 reconnects reporting surfaces; it
  does not make the model live-discussion eligible.
- The browser/UI and wager-card runtime checks used fixture-backed artifacts in
  `/tmp/age295-dashboard-runtime`, not real betting evidence.
- Real season/date betting-layer validation is still blocked by the AGE-293 and
  AGE-294 handoff caveat: this worktree does not have restored real historical
  Odds API market artifacts available.
- Do not loosen approval gates to force output. Missing validation, no-go
  validation, missing projection timestamps, below-threshold edges, and
  correlated duplicate lines must stay explicit rejections.
- CLV and ROI context are surfaced when present, but they are still separate
  reporting signals and are not used to tune projection or validation-derived
  approval thresholds.
