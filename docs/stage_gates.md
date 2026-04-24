# Live-Use And Expansion Stage Gates

## Why This Exists

The repo now has enough training, backtest, and paper-tracking seams that it
needs one explicit answer to a simple question:

- is the project still `research_only`
- is it `eligible_for_live_discussion`
- is it `eligible_for_next_market_expansion`

Those labels are strict. Every gate below is conjunctive. If any required
artifact is missing, empty, or below threshold, the status stays
`research_only`.

Positive short-run ROI alone is never enough to promote the system.

Executable checks and threshold constants live in
`src/mlb_props_stack/stage_gates.py` and are exposed through:

```bash
uv run python -m mlb_props_stack evaluate-stage-gates
```

That command writes `stage_gate_report.json` and `stage_gate_report.md` under
`data/normalized/stage_gates/run=<timestamp>/`. Use
`--fail-on-research-only` only when a caller intentionally wants a research-only
status to fail the process.

## Artifact Set To Review

Apply the gates to one coherent artifact set for the same model version:

1. training summary
   `data/normalized/starter_strikeout_baseline/.../evaluation_summary.json`
2. walk-forward backtest summary
   `data/normalized/walk_forward_backtest/.../backtest_runs.jsonl`
3. walk-forward CLV summary
   `data/normalized/walk_forward_backtest/.../clv_summary.jsonl`
4. walk-forward ROI summary
   `data/normalized/walk_forward_backtest/.../roi_summary.jsonl`
5. paper-tracking results
   `data/normalized/paper_results/.../paper_results.jsonl`

Do not mix one strong training run with a different market snapshot window or a
different paper-tracking state.

## Metric Definitions

Use these exact definitions when applying the checklist:

- `held_out_rows`
  `evaluation_summary.json -> row_counts.held_out`
- `scoreable_backtest_rows`
  `backtest_runs.jsonl -> row_counts.actionable + row_counts.below_threshold`
- `backtest_skip_rate`
  `backtest_runs.jsonl -> row_counts.skipped / row_counts.snapshot_groups`
- `backtest_placed_bets`
  `backtest_runs.jsonl -> bet_outcomes.placed_bets`
- `backtest_clv_sample`
  `clv_summary.jsonl -> sample_count` from the `summary_scope=overall` row
- `settled_paper_bets`
  count rows in `paper_results.jsonl` where `settlement_status` is `win`,
  `loss`, or `push`
- `paper_dates`
  distinct `official_date` values across settled paper bets
- `paper_same_line_clv_sample`
  settled paper bets where `same_line_close_available=true`
- `paper_beat_close_rate`
  settled paper bets with `beat_closing_line=true` divided by
  `paper_same_line_clv_sample`
- `paper_median_clv_probability_delta`
  median `clv_probability_delta` across the same settled same-line-close sample
- `paper_roi`
  `sum(profit_units) / sum(stake_fraction)` across settled paper bets

If `paper_same_line_clv_sample` is `0`, CLV is treated as unavailable and the
gate fails.

## Live-Use Discussion Checklist

All of these must pass before any live-usage discussion.

| Gate | Numeric threshold | Read from |
| --- | --- | --- |
| Held-out quality | `held_out_rows >= 100`, model beats benchmark on held-out RMSE and MAE, calibrated ECE `<= 0.03` | `evaluation_summary.json` |
| Backtest coverage | `scoreable_backtest_rows >= 100`, `backtest_placed_bets >= 75`, `backtest_skip_rate <= 0.20` | `backtest_runs.jsonl` |
| Paper sample | `settled_paper_bets >= 100` across `paper_dates >= 30` | `paper_results.jsonl` |
| Market-beating evidence | `paper_same_line_clv_sample >= 75`, `paper_beat_close_rate >= 0.52`, `paper_median_clv_probability_delta > 0`, and `backtest_clv_sample >= 75` with positive overall median CLV delta | `paper_results.jsonl`, `clv_summary.jsonl` |
| Profit corroboration | paper ROI `> 0` and backtest ROI `> 0` | `paper_results.jsonl`, `roi_summary.jsonl` |

Interpretation rules:

- If the model has good held-out RMSE or MAE but zero scoreable backtest rows,
  it is still `research_only`.
- If the model has positive paper ROI on fewer than `100` settled bets, it is
  still `research_only`.
- If paper ROI is positive but median CLV is not positive, it is still
  `research_only`.

## Next-Market Expansion Checklist

Adding the next market is a stricter decision than discussing small live usage
on the current one. Every live-use gate above must already pass, then every
gate below must also pass.

| Gate | Numeric threshold | Read from |
| --- | --- | --- |
| Current-market live gate | every live-use discussion gate already passes | full artifact set |
| Held-out depth | `held_out_rows >= 150`, model still beats benchmark on held-out RMSE and MAE, calibrated ECE `<= 0.025` | `evaluation_summary.json` |
| Backtest depth | `scoreable_backtest_rows >= 250`, `backtest_placed_bets >= 150`, `backtest_skip_rate <= 0.10` | `backtest_runs.jsonl` |
| Paper depth | `settled_paper_bets >= 250` across `paper_dates >= 60` | `paper_results.jsonl` |
| Persistent CLV | `paper_same_line_clv_sample >= 150`, `paper_beat_close_rate >= 0.53`, `paper_median_clv_probability_delta > 0`, and `backtest_clv_sample >= 150` with positive overall median CLV delta | `paper_results.jsonl`, `clv_summary.jsonl` |
| Persistent profitability | paper ROI `> 0` and backtest ROI `> 0` on the larger sample | `paper_results.jsonl`, `roi_summary.jsonl` |

The point of this second gate is not to prove the next market will work. The
point is to avoid expanding while the first market is still under-sampled,
operationally brittle, or only looks good because of one hot stretch.

## Status Resolution

Resolve status in this order:

1. If any live-use gate fails, status is `research_only`.
2. If every live-use gate passes but any expansion gate fails, status is
   `eligible_for_live_discussion`.
3. If every live-use gate and every expansion gate passes, status is
   `eligible_for_next_market_expansion`.

There is no fourth state for "probably ready." If the artifact set does not
clear the thresholds, the repo is still `research_only`.

## Worked Example: Historical Artifact Set

The saved artifact set reviewed for the original stage-gate issue used the
following run ids. The generated local artifacts were removed during the
2026-04-24 model-rebuild cleanup, so treat these values as historical evidence,
not as files expected to exist in a fresh checkout:

- training run `20260422T205727Z`
- walk-forward backtest run `20260422T205734Z`
- latest paper-results run `20260422T190633Z`
  this file is empty, and the earlier `20260422T173038Z` paper-results run is
  also empty

Observed values:

| Metric | Actual value | Required live-use threshold | Result |
| --- | ---: | ---: | --- |
| `held_out_rows` | `48` | `>= 100` | fail |
| held-out RMSE / MAE vs benchmark | model beats both | beat both | pass |
| calibrated ECE | `0.02285` | `<= 0.03` | pass |
| `scoreable_backtest_rows` | `0` | `>= 100` | fail |
| `backtest_placed_bets` | `0` | `>= 75` | fail |
| `backtest_skip_rate` | `1.00` | `<= 0.20` | fail |
| `settled_paper_bets` | `0` | `>= 100` | fail |
| `paper_dates` | `0` | `>= 30` | fail |
| `paper_same_line_clv_sample` | `0` | `>= 75` | fail |
| paper ROI | `n/a` | `> 0` | fail |
| backtest ROI | `n/a` | `> 0` | fail |

Current status from this artifact set:

- `research_only`

Why that status is unambiguous:

- the model summary is encouraging, but the held-out sample is still too small
- the latest saved backtest window has `139` snapshot groups and `139` skipped
  rows, so there are `0` scoreable rows and `0` placed bets
- the latest saved paper-tracking runs contain `0` settled bets, so there is
  no live-usage sample yet
- even if the next few bets happen to win, positive short-run ROI alone would
  still not clear the live-use gate without the required sample and CLV support

## What Should Change Before Rechecking

The next gate review is only worth running after the repo can produce all of
the following on the current pitcher strikeout market:

- non-empty `daily_candidates.jsonl` and `paper_results.jsonl`
- a walk-forward backtest window with scoreable exact-line rows instead of an
  all-skipped summary
- a larger held-out evaluation window than the current `48` held-out rows

Until then, this project should be discussed and operated as `research_only`.
