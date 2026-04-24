# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Working branch: `feat/age-267-optional-feature-artifacts`
- Base commit: `1701e7a` (`origin/main` at issue start)
- Current issue: `AGE-267` - regenerate historical optional-feature artifacts
  with timestamp-valid coverage
- Pull request: <https://github.com/DylanMcCavitt/baseball-ml/pull/40>
- Canonical ignored data directory used for regeneration:
  `/Users/dylanmccavitt/projects/nba-ml/data`
- Status: AGE-267 implementation, local artifact regeneration, verification,
  and PR creation are complete. Merge closeout is still pending at the time this
  handoff was written.

## What Changed In This Slice

- Regenerated the canonical optional-feature artifact window
  `2026-04-18` through `2026-04-23`.
- Hardened umpire ingest so it no longer mines postgame `feed/live` payloads
  and then clamps `captured_at` to first pitch. A home-plate assignment is only
  used when the source capture is at or before `commence_time`; otherwise the
  row is emitted as explicit `missing_umpire_source`.
- Hardened pitcher, lineup, and game-context `features_as_of` handling so
  post-cutoff metadata timestamps from historical completed slates do not
  propagate into regenerated feature rows.
- Added current MLB venue-id aliases to the static park-factor CSV for:
  - Sutter Health Park (`2529`)
  - Nationals Park (`3309`)
  - Target Field (`3312`)
  - Yankee Stadium (`3313`)
  - Globe Life Field (`5325`)
- Regenerated `game_context_features.jsonl` with current schema names:
  `park_k_factor`, `park_k_factor_vs_rhh`, `park_k_factor_vs_lhh`,
  `weather_wind_speed_mph`, `weather_humidity_pct`,
  `ump_called_strike_rate_30d`, and
  `ump_k_per_9_delta_vs_league_30d`.
- Did not train or compare an expanded-feature model, did not change
  stage-gate status, and did not loosen optional-feature selection thresholds.

## Artifact Runs Produced

Canonical weather runs:

| Date | Weather run |
| --- | --- |
| 2026-04-18 | `20260424T142001Z` |
| 2026-04-19 | `20260424T142008Z` |
| 2026-04-20 | `20260424T142016Z` |
| 2026-04-21 | `20260424T142021Z` |
| 2026-04-22 | `20260424T142027Z` |
| 2026-04-23 | `20260424T142035Z` |

Canonical umpire runs:

| Date | Umpire run | OK games | Missing-source games |
| --- | --- | ---: | ---: |
| 2026-04-18 | `20260424T142046Z` | 0 | 15 |
| 2026-04-19 | `20260424T142046Z` | 0 | 15 |
| 2026-04-20 | `20260424T142046Z` | 0 | 10 |
| 2026-04-21 | `20260424T142046Z` | 0 | 15 |
| 2026-04-22 | `20260424T142047Z` | 3 | 12 |
| 2026-04-23 | `20260424T142047Z` | 2 | 7 |

Canonical Statcast feature runs after the park-factor alias fix:

| Date | Statcast feature run | Rows |
| --- | --- | ---: |
| 2026-04-18 | `20260424T142511Z` | 30 |
| 2026-04-19 | `20260424T142513Z` | 30 |
| 2026-04-20 | `20260424T142514Z` | 20 |
| 2026-04-21 | `20260424T142515Z` | 30 |
| 2026-04-22 | `20260424T142517Z` | 30 |
| 2026-04-23 | `20260424T142521Z` | 18 |

## Coverage Delta

Pre-regeneration `check-data-alignment` for `2026-04-18` through
`2026-04-23` showed:

- `wx_cov=0.0%` and `ump_cov=0.0%` on every date.
- `2026-04-22` had `pitcher_feats=0`, `lineup_feats=0`, and
  `context_feats=0`.
- Existing game-context rows used stale keys such as `park_factor` and
  `weather_wind_mph`.

Post-regeneration `check-data-alignment` showed:

- `feat_cov=100.0%` for every date in the window.
- `wx_cov=100.0%` for every date in the window.
- `ump_cov=20.0%` on `2026-04-22` and `22.2%` on `2026-04-23`; older dates
  correctly remain explicit missing-source because no pregame-valid umpire
  assignment capture exists.
- Remaining alignment failures are known odds/outcome gaps:
  - `2026-04-18` through `2026-04-21`: odds coverage is still `0.0%`.
  - `2026-04-22`: outcome and odds coverage still fail.
  - `2026-04-23`: outcome coverage still fails.

Direct artifact inspection after regeneration showed:

- No generated pitcher, lineup, or game-context feature row had
  `features_as_of > commence_time`.
- `game_context_features.jsonl` no longer includes old `park_factor` or
  `weather_wind_mph` keys.
- Park-factor coverage is now non-null for every regenerated game-context row.
- Weather wind speed and humidity are non-null for every outdoor row; fixed-roof
  games remain explicit neutral/missing weather-field rows.
- Umpire assignment rows exist for 2026-04-22 and 2026-04-23, but rolling
  umpire tendency fields remain null because there is not yet enough prior
  timestamp-valid umpire history.
- Lineup aggregate coverage remains limited:
  - `2026-04-18` through `2026-04-21` have no pregame-valid lineup snapshots.
  - `2026-04-22` has 24 non-missing lineup rows, but only 5 rows currently have
    non-null projected lineup K-rate metrics.
  - `2026-04-23` has projected lineup rows, but the current artifact has null
    lineup aggregate rates.

## Files Changed

- `data/static/park_factors/park_k_factors.csv`
- `docs/NEXT_SESSION_HANDOFF.md`
- `src/mlb_props_stack/ingest/game_context.py`
- `src/mlb_props_stack/ingest/lineup_aggregation.py`
- `src/mlb_props_stack/ingest/pitcher_features.py`
- `src/mlb_props_stack/ingest/umpire.py`
- `tests/test_park_factors.py`
- `tests/test_statcast_feature_ingest.py`
- `tests/test_umpire_ingest.py`

## Verification

Commands run:

```bash
uv sync --extra dev
uv run python -m mlb_props_stack check-data-alignment --start-date 2026-04-18 --end-date 2026-04-23 --output-dir /Users/dylanmccavitt/projects/nba-ml/data
uv run python -m mlb_props_stack ingest-weather --date <date> --output-dir /Users/dylanmccavitt/projects/nba-ml/data
uv run python -m mlb_props_stack ingest-umpire --date <date> --output-dir /Users/dylanmccavitt/projects/nba-ml/data
uv run python -m mlb_props_stack ingest-statcast-features --date <date> --output-dir /Users/dylanmccavitt/projects/nba-ml/data
uv run pytest tests/test_umpire_ingest.py tests/test_statcast_feature_ingest.py
uv run pytest tests/test_park_factors.py tests/test_statcast_feature_ingest.py tests/test_weather_ingest.py tests/test_umpire_ingest.py tests/test_data_alignment.py
uv run pytest tests/test_runtime_smokes.py
uv run pytest
python3 -m compileall src tests
uv run python -m mlb_props_stack
```

The `<date>` commands were run for each date from `2026-04-18` through
`2026-04-23`.

Observed results:

- `tests/test_umpire_ingest.py tests/test_statcast_feature_ingest.py`:
  `33 passed`
- `tests/test_park_factors.py tests/test_statcast_feature_ingest.py
  tests/test_weather_ingest.py tests/test_umpire_ingest.py
  tests/test_data_alignment.py`: `71 passed`
- `tests/test_runtime_smokes.py`: `4 passed` with existing third-party
  MLflow/Pydantic warnings
- Full suite: `198 passed` with the same existing third-party warnings
- `python3 -m compileall src tests` completed successfully
- `uv run python -m mlb_props_stack` printed the runtime configuration banner
- `check-data-alignment` still exits nonzero because remaining odds/outcome
  gaps are outside AGE-267 scope; the post-run output now makes weather and
  umpire coverage visible.

## Recommended Next Issue

Work:

1. `AGE-268` - train and compare expanded-feature baseline against core-only
   model
   - Use the regenerated feature window from AGE-267 and confirm which optional
     families survive coverage and variance selection.
   - Do not treat null lineup/umpire rolling metrics as bugs unless the source
     artifacts above should have made them available.
2. `AGE-262` - run the first approved-wager evidence refresh and readiness
   report
   - Use `evaluate-stage-gates` as the readiness report command after a coherent
     expanded/core comparison exists.
3. `AGE-263` - close the sample-size gap for live-use stage gates
   - Broader evidence collection after the first coherent refresh.

Related but not on the critical path:

- `AGE-209` - add per-pitcher and per-game exposure caps to Kelly sizing.
  Keep queued unless grouped board work shows exposure/correlation sizing is
  still materially distorting the wager card before `AGE-262`.

## Constraints And Notes

- Use the canonical ignored data directory for live dashboard/model checks:
  `/Users/dylanmccavitt/projects/nba-ml/data`.
- Do not mine postgame `feed/live` umpire assignments as if they were pregame
  captures.
- Do not loosen optional feature coverage thresholds to make a family appear
  active.
- The active model remains core-only until AGE-268 trains and compares a new
  baseline.
