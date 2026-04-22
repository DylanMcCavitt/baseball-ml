# Starter Strikeout Baseline Evaluation Summary

- Run ID: `20260422T205727Z`
- MLflow run ID: `5fbc851c3c7643daa4add2fb6706eee5`
- MLflow experiment: `mlb-props-stack-starter-strikeout-training`
- Tracking URI: `file:./artifacts/mlruns`
- Date window: `2026-04-18` -> `2026-04-23`
- Reproducibility notes: `data/normalized/starter_strikeout_baseline/start=2026-04-18_end=2026-04-23/run=20260422T205727Z/reproducibility_notes.md`
- Row counts: train=60, validation=20, test=28, held_out=48
- Held-out status: `beating_benchmark`

## Held-Out Performance

| Metric | Benchmark | Model | Model Beats Benchmark |
| --- | ---: | ---: | --- |
| RMSE | 2.518693 | 2.322574 | yes |
| MAE | 2.047866 | 1.960881 | yes |
| Spearman | 0.321431 | 0.275293 | n/a |

## Held-Out Probability Calibration

| Metric | Raw | Calibrated | Improved |
| --- | ---: | ---: | --- |
| Mean Brier Score | 0.052066 | 0.051112 | yes |
| Mean Log Loss | 0.164896 | 0.346381 | no |
| Expected Calibration Error | 0.028622 | 0.022850 | yes |

## Held-Out Count Distribution

- Dispersion alpha: `0.000000`
- Negative binomial beats Poisson on held-out metrics: `{'mean_negative_log_likelihood': False, 'mean_ranked_probability_score': True}`

## Top Feature Importance

| Feature | Coefficient | Absolute Importance |
| --- | ---: | ---: |
| `rest_days` | 0.627519 | 0.627519 |
| `pitcher_k_rate` | 0.545975 | 0.545975 |
| `recent_pitch_count` | 0.370838 | 0.370838 |
| `csw_rate` | 0.338268 | 0.338268 |
| `pitch_sample_size` | 0.316851 | 0.316851 |
| `expected_leash_batters_faced` | 0.177839 | 0.177839 |
| `recent_batters_faced` | -0.176321 | 0.176321 |
| `plate_appearance_sample_size` | 0.077162 | 0.077162 |
| `swinging_strike_rate` | -0.058488 | 0.058488 |
| `last_start_batters_faced` | -0.027478 | 0.027478 |

## Comparison To Previous Run

No previous run was found for this exact date window.
