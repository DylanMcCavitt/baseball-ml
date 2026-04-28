# MLB Props Reboot

Clean reboot of the MLB pitcher strikeout props project.

The system is intentionally scoped end to end:

1. build timestamp-valid starter and pitch data
2. define source-backed pitcher strikeout features
3. train a strikeout distribution model
4. compare model probabilities to sportsbook `pitcher_strikeouts` prices
5. produce paper recommendations
6. review every stage through durable reports and a local dashboard

## Quick Start

```bash
uv sync --extra dev
uv run pytest
uv run python -m mlb_props_lab feature-registry validate
uv run python -m mlb_props_lab targets build --issue TARGET-SAMPLE
uv run python -m mlb_props_lab statcast build-features --issue STATCAST-SAMPLE
uv run python -m mlb_props_lab report features --issue FEATURE-RESEARCH
uv run python -m mlb_props_lab dashboard
```

Generated report artifacts are written under `artifacts/reports/`.
The static dashboard is written to `artifacts/dashboard/index.html`.

## Current Slice

This reboot slice implements the source-backed feature registry, a
fixture-backed pitcher-start target dataset, report surfaces, and a small
fixture-backed Statcast feature materialization path. It does not train a model
yet. The point is to force every future model feature to declare:

- data source and source fields
- formula
- lookback window
- timestamp cutoff
- missing-value policy
- leakage risk
- required visual

The target dataset builder keeps pre-game identity and schedule fields separate
from post-game strikeout outcomes, and reports missing targets, duplicate
starts, unresolved identities, and timestamp-invalid rows.

See `docs/features.md` for the registry policy and feature backlog.
