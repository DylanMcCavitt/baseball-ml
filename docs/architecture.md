# Architecture

## Product Thesis

The best first MLB betting system is not a broad winner-picking model. It is a
narrow prop engine centered on measurable baseball process and disciplined
decision rules.

The first market in scope is `pitcher strikeout props`.

## High-Level Flow

```text
schedule + probable starters + confirmed lineups
        +
Statcast / Baseball Savant pitch-level history
        +
sportsbook prop lines and prices
        ->
feature store
        ->
strikeout projection model
        ->
probability calibration
        ->
price conversion / devig / expected value
        ->
bet filter + sizing
        ->
walk-forward backtest and tracking
```

## Core Layers

### 1. Data layer

Owns normalized inputs:

- games
- teams
- pitchers
- hitters
- lineups
- weather / park context
- sportsbook lines
- line movement snapshots

### 2. Feature layer

Owns features that are valid at a specific timestamp:

- pitcher strikeout skill
- pitch-type-specific whiff ability
- hitter K tendency by handedness
- team chase / contact profiles
- expected innings / batters faced
- umpire, park, and weather adjustments

### 3. Projection layer

Owns the actual baseball estimate:

- expected strikeouts
- strikeout distribution
- probability of clearing each listed line

### 4. Decision layer

Owns the market comparison:

- implied probability from book odds
- optional devig for two-way props
- expected value
- thresholding and minimum edge
- position sizing

### 5. Evaluation layer

Owns honesty:

- walk-forward splits only
- no post-lock lineups or prices in feature generation
- no using closing lines if the simulated bettor could not access them
- CLV and ROI reported separately

## Why RL Is Not in V1

Reinforcement learning is more appropriate for:

- timing entry in moving markets
- adaptive sizing
- in-play decision policies
- portfolio allocation across multiple correlated bets

It is not the simplest or strongest first tool for estimating pregame pitcher
strikeout probabilities. V1 should earn the right to add RL by first proving
that the static probabilistic model is calibrated and that any paper edge
survives out-of-sample testing.
