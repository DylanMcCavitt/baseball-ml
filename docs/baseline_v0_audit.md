# Baseline v0 Audit

Issue: `AGE-286`

Status: the pre-rebuild starter strikeout model is frozen as
`starter-strikeout-baseline-v0`. It is obsolete historical residue, not
infrastructure to preserve, not a benchmark, and not a production or live-use
betting model.

This issue does not train a new model, loosen wager gates, or report readiness
as trusted. It records what the obsolete baseline did and why it must not carry
forward into the projection rebuild.

## What v0 Does

The v0 path in `src/mlb_props_stack/modeling.py`:

- joins AGE-146 `pitcher_daily_features`, `lineup_daily_features`, and
  `game_context_features` by `official_date`, `gamePk`, and `pitcher_id`
- derives the starter strikeout label from same-day Statcast pitcher rows
- splits data chronologically by official date, never by random row sampling
- trains a standard-library ridge-style linear regressor for expected
  strikeouts
- fits one global negative-binomial dispersion parameter on top of the mean
- writes raw and calibrated half-strikeout ladder probabilities
- fits an isotonic probability calibrator from out-of-fold ladder events
- persists auditable feature schema, split, calibration, and reproducibility
  artifacts

Do not preserve v0 as a modeling baseline or performance reference. Its
artifacts, metrics, and feature assumptions should not be used to judge the
rebuild. Any future modeling path should be evaluated from the rebuilt
starter-game dataset and rebuilt feature families only.

## AGE-268 Preserved Evidence

The latest obsolete v0 evidence came from this AGE-268 command:

```bash
uv run python -m mlb_props_stack compare-starter-strikeout-baselines --start-date 2026-04-18 --end-date 2026-04-23 --output-dir /Users/dylanmccavitt/projects/nba-ml/data
```

The generated files were intentionally deleted from the canonical checkout on
2026-04-24 to prevent stale artifact drift. The preserved handoff evidence is
only useful for explaining why v0 is out of scope; it is not current performance
evidence.

Run IDs:

| Variant | Training run | Backtest run |
| --- | --- | --- |
| Core | `20260424T164050Z` | `20260424T164112Z` |
| Expanded | `20260424T164112Z` | `20260424T164113Z` |

The comparison window had six official dates:

- train dates from current `_split_dates` behavior:
  `2026-04-18`, `2026-04-19`, `2026-04-20`, `2026-04-21`
- validation date: `2026-04-22`
- test date: `2026-04-23`

Preserved comparison and backtest counts:

| Metric | Core | Expanded |
| --- | ---: | ---: |
| Held-out RMSE | `2.150421` | `2.108512` |
| Held-out MAE | `1.695608` | `1.722785` |
| Calibrated log loss | `0.243186` | `0.240320` |
| Calibrated ECE | `0.023741` | `0.027515` |
| Snapshot groups | `469` | `469` |
| Scoreable rows | `0` | `0` |
| Final-gate approved wagers | `0` | `0` |

The exact training row count from the deleted AGE-268 artifacts is not retained
in tracked docs. Do not reconstruct or guess it in later reviews. AGE-287 must
build a durable multi-season starter-game dataset and persist row-count
evidence as part of the rebuild.

Expanded active optional features in AGE-268:

- `pitcher_k_rate_vs_rhh`
- `pitcher_k_rate_vs_lhh`
- `pitcher_whiff_rate_vs_rhh`
- `pitcher_whiff_rate_vs_lhh`
- `park_k_factor`
- `park_k_factor_vs_rhh`
- `park_k_factor_vs_lhh`
- `weather_temperature_f`
- `weather_wind_speed_mph`
- `weather_humidity_pct`

Still-excluded optional families:

- lineup aggregates had `0.0` coverage:
  `projected_lineup_k_rate`, `projected_lineup_k_rate_vs_pitcher_hand`,
  `lineup_k_rate_vs_rhp`, `lineup_k_rate_vs_lhp`,
  `projected_lineup_chase_rate`, `projected_lineup_contact_rate`, and
  `lineup_continuity_ratio`
- umpire rolling metrics had `0.0` coverage:
  `ump_called_strike_rate_30d` and `ump_k_per_9_delta_vs_league_30d`

Expanded backtest skip reasons:

| Skip reason | Count |
| --- | ---: |
| `unmatched_event_mapping` | `337` |
| `late_snapshot_after_cutoff` | `74` |
| `invalid_projection` | `48` |
| `missing_projection` | `10` |

The recommendation was `keep_core_only`: expanded improved held-out RMSE but
worsened held-out MAE and calibrated ECE, and neither variant produced
scoreable or final-gate-approved wagers.

## Why Rest Days Are Unsafe Here

`rest_days` is part of the dense core feature set. On a short v0 window, it can
look important for reasons the model cannot distinguish:

- normal starter-cycle rest
- long layoffs
- injury or illness return
- skipped starts, rehab context, or role changes
- weather or schedule disruption
- pitch-limit risk after an abnormal gap

Those situations do not share one monotonic effect on strikeout expectation.
Longer rest can mean recovery, but it can also mean injury, reduced workload,
lost rhythm, a managed return, or a smaller pitch count. The v0 feature does
not encode injured-list status, rehab assignment context, manager leash,
explicit pitch limits, or why the rest interval was abnormal.

The rebuild should treat rest as one part of a richer workload and availability
feature family, not as a standalone continuous signal that can safely absorb
health and usage context.

## Blocked Betting And Readiness Work

The v0 baseline is not trusted for live candidates or approved wagers. The
following work remains blocked by the projection rebuild:

- canceled or superseded by the rebuild track: `AGE-262`, `AGE-263`,
  `AGE-207`, `AGE-208`
- deferred until downstream rebuild milestones:
  `AGE-209` waits for `AGE-294`, `AGE-210` has its AGE-286 audit dependency
  satisfied by this freeze but still waits for `AGE-291`, and `AGE-212` waits
  for `AGE-293` / `AGE-294`
- dashboard and approved-wager UX reconnection waits for model validation,
  scoreable market joins, and rebuilt betting logic in `AGE-291` through
  `AGE-295`

## Assumptions Not Carried Forward

Do not carry these v0 assumptions into the rebuild:

- a short-window ridge baseline is enough for betting decisions
- `rest_days` alone can proxy workload, health, and return-from-layoff context
- sparse lineup and umpire coverage can be ignored without costing matchup
  quality
- a single global negative-binomial dispersion is enough for all pitchers and
  matchups
- small held-out improvements imply betting readiness
- zero scoreable historical market rows can be treated as a minor reporting gap
- final approved wagers can resume before projection validation and market join
  coverage are fixed

The next projection track should rebuild the starter-game dataset first, then
reintroduce pitcher skill, lineup matchup, workload, injury-context,
distribution, and calibration choices with walk-forward evidence.
