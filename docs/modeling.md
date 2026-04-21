# Modeling Plan

## V1 Market

`Pitcher strikeout props`

Typical lines:

- 3.5
- 4.5
- 5.5
- 6.5

## Target

Primary target:

- probability that a starting pitcher records at least `n` strikeouts

Derived target:

- full or partial strikeout-count distribution

That lets one projection support:

- posted main line
- alternate ladders
- fair price generation

## Feature Families

### Pitcher skill

- rolling K%
- rolling CSW%
- rolling Whiff%
- called strike rate
- zone%
- chase rate induced
- pitch-level whiff by pitch type
- velocity and movement deltas

### Opponent contact profile

- lineup K% by handedness
- swing aggression / chase profile
- contact quality allowed by zone and pitch type
- expected lineup strength after confirmations

### Usage / leash

- rolling pitch count
- rolling batters faced
- starter inning depth
- manager / team leash tendencies
- bullpen freshness, which affects early pull probability

### Context

- park
- weather
- umpire if available
- rest
- travel

## Modeling Approach

Recommended first pass:

1. predict expected strikeouts with gradient boosting
2. fit a count distribution around that expectation
3. calibrate threshold probabilities out of sample

Candidate model path:

- baseline: Poisson / negative binomial GLM
- production v1: XGBoost or LightGBM regressor for expected Ks
- calibration: isotonic or Platt-style mapping on out-of-fold outputs

## Validation Rules

- use walk-forward validation only
- feature timestamps must be strictly pregame
- lineups must reflect what was known at evaluation time
- line comparisons must use the actual accessible sportsbook line
- report CLV separately from realized ROI
- size paper bets conservatively in backtests

## Promotion Criteria for Live Tracking

A model is not ready for live tracking until it shows all of the following on
held-out data:

- stable calibration across line buckets
- positive median CLV
- positive ROI over a meaningful sample
- no dependence on one narrow date range or one team cluster
- sensible degradation when vig assumptions tighten
