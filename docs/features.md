# Source-Backed Feature Registry

This project predicts MLB starting pitcher strikeout distributions, then compares
those probabilities to sportsbook pitcher strikeout markets. Feature work starts
with source-backed registration before model training.

## Feature Acceptance Rule

Every model feature must declare:

- feature family
- source references and source fields
- formula
- lookback window
- timestamp cutoff
- missing-value policy
- leakage risk
- required visual

A feature is not "good" because it sounds plausible. It must either improve
walk-forward validation or explain model behavior without leaking future data.

## V1 Required Families

### Pitcher Strikeout Skill

Core rolling and season-to-date pitcher strikeout indicators:

- K%, K/9, K-BB%, BB%
- batters faced and pitches per start
- called-strike rate, swinging-strike rate, CSW%, whiff rate
- zone rate and chase/contact allowed

### Arsenal And Stuff

Pitch-level Statcast measurements that describe how the pitcher gets outs:

- pitch mix by pitch type
- release velocity and effective velocity
- release spin rate
- extension and release point
- horizontal and vertical movement
- pitch-specific whiff, CSW, and contact rates
- recent velocity and spin movement trends

### Handedness And Platoon

Pitcher and hitter interaction by handedness:

- pitcher vs LHB/RHB K%, BB%, whiff, CSW, and pitch mix
- opponent projected lineup handedness mix
- hitter strikeout/contact/chase behavior versus pitcher throwing hand

### Opponent Lineup Context

Projected opponent lineup strikeout difficulty:

- projected starter lineup K/contact/chase tendencies
- lineup depth and expected lineup turns
- recent lineup form when timestamp-valid

### Workload And Leash

Expected opportunity to accumulate strikeouts:

- rest days
- recent pitch counts and innings
- season workload
- prior-start hook patterns
- blowup risk and bullpen/team context when source-backed

### Environment And Context

Game-level context:

- park and home/away
- game time and travel/rest proxies
- weather when available before first pitch
- umpire strike-zone tendency when source-backed before game time

### Market Context

Sportsbook line context for downstream comparison:

- book, line, over/under price, timestamp
- consensus/tightest-book price
- devigged implied probability
- line movement and closing-line value

## Source References

- Baseball Savant CSV docs: https://baseballsavant.mlb.com/csv-docs
- MLB Statcast glossary: https://www.mlb.com/glossary/statcast/
- MLB Pitch Movement glossary: https://www.mlb.com/glossary/statcast/pitch-movement/
- MLB Strikeout Rate glossary: https://www.mlb.com/glossary/advanced-stats/strikeout-rate
- FanGraphs Plate Discipline glossary: https://library.fangraphs.com/pitching/plate-discipline-o-swing-z-swing-etc/
- The Odds API betting markets: https://the-odds-api.com/sports-odds-data/betting-markets.html

## Feature Backlog

### V1 Required

- rolling pitcher K%, BB%, K-BB%, K/9, batters faced, and pitches per start
- rolling CSW%, swinging-strike rate, called-strike rate, whiff rate, zone rate
- pitch mix, velocity, effective velocity, spin, extension, release point, movement
- pitcher handedness splits versus LHB/RHB
- projected opponent lineup handedness and strikeout/contact profile
- rest days, recent pitch counts, and recent innings
- sportsbook line, price, book, timestamp, and devigged implied probability

### V1 Optional

- weather and park interaction features
- umpire called-strike tendency
- bullpen availability and team hook tendency
- alternate-line market movement and consensus/tightest-book disagreement

### Later

- pitch sequencing/tunneling
- catcher framing
- minor-league and injury/news signals
- automated feature ablation promotion gates
