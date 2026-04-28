# FEATURE-RESEARCH Review Packet

## Summary

This reboot slice establishes the source-backed feature registry before model
training resumes. It adds machine-readable feature metadata, a generated feature
report, and a dashboard renderer that can load report manifests.

## Evidence

- Registry: `src/mlb_props_lab/resources/feature_registry.json`
- Human docs: `docs/features.md`
- Generate report: `uv run python -m mlb_props_lab report features --issue FEATURE-RESEARCH`
- Generate dashboard: `uv run python -m mlb_props_lab dashboard`

## Review Focus

- Every v1 feature family has source references and timestamp cutoffs.
- Spin, velocity, handedness, lineup, workload, environment, and market context
  are explicitly represented.
- Future model work has a mechanical registry validation gate before training.
