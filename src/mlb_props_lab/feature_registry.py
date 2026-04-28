from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

REQUIRED_FEATURE_FIELDS = {
    "id",
    "name",
    "family",
    "status",
    "source_refs",
    "source_fields",
    "formula",
    "lookback_window",
    "timestamp_cutoff",
    "missing_policy",
    "leakage_risk",
    "required_visual",
}

REQUIRED_FAMILIES = {
    "pitcher_strikeout_skill",
    "arsenal_and_stuff",
    "handedness_and_platoon",
    "opponent_lineup_context",
    "workload_and_leash",
    "environment_and_context",
    "market_context",
}

VALID_STATUSES = {"v1_required", "v1_optional", "later"}


@dataclass(frozen=True)
class RegistryValidationResult:
    feature_count: int
    family_count: int
    required_feature_count: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def default_registry_path() -> Path:
    registry = resources.files("mlb_props_lab.resources").joinpath("feature_registry.json")
    return Path(str(registry))


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path else default_registry_path()
    with registry_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        msg = "feature registry must be a JSON object"
        raise ValueError(msg)
    return data


def validate_registry(data: dict[str, Any]) -> RegistryValidationResult:
    errors: list[str] = []
    features = data.get("features")
    families = data.get("families")
    sources = data.get("sources")

    if not isinstance(features, list):
        errors.append("features must be a list")
        features = []
    if not isinstance(families, list):
        errors.append("families must be a list")
        families = []
    if not isinstance(sources, dict):
        errors.append("sources must be an object")
        sources = {}

    family_set = set(families)
    missing_families = REQUIRED_FAMILIES - family_set
    if missing_families:
        errors.append(f"missing required feature families: {sorted(missing_families)}")

    seen_ids: set[str] = set()
    required_by_family = {family: 0 for family in REQUIRED_FAMILIES}

    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            errors.append(f"feature at index {index} must be an object")
            continue

        missing_fields = REQUIRED_FEATURE_FIELDS - set(feature)
        if missing_fields:
            label = feature.get("id", f"feature[{index}]")
            errors.append(f"{label} missing {sorted(missing_fields)}")

        feature_id = feature.get("id")
        if not isinstance(feature_id, str) or not feature_id:
            errors.append(f"feature at index {index} has invalid id")
        elif feature_id in seen_ids:
            errors.append(f"duplicate feature id: {feature_id}")
        else:
            seen_ids.add(feature_id)

        family = feature.get("family")
        if family not in family_set:
            errors.append(f"{feature_id} uses unknown family: {family}")

        status = feature.get("status")
        if status not in VALID_STATUSES:
            errors.append(f"{feature_id} uses invalid status: {status}")
        elif status == "v1_required" and family in required_by_family:
            required_by_family[family] += 1

        source_refs = feature.get("source_refs")
        if not isinstance(source_refs, list) or not source_refs:
            errors.append(f"{feature_id} must declare at least one source reference")
        else:
            for source_ref in source_refs:
                if source_ref not in sources:
                    errors.append(f"{feature_id} references unknown source: {source_ref}")

        source_fields = feature.get("source_fields")
        if not isinstance(source_fields, list) or not source_fields:
            errors.append(f"{feature_id} must declare source fields")

        for field in (
            "formula",
            "lookback_window",
            "timestamp_cutoff",
            "missing_policy",
            "leakage_risk",
            "required_visual",
        ):
            if not isinstance(feature.get(field), str) or not feature[field].strip():
                errors.append(f"{feature_id} must declare {field}")

    for family, count in sorted(required_by_family.items()):
        if count == 0:
            errors.append(f"family has no v1_required feature: {family}")

    return RegistryValidationResult(
        feature_count=len(features),
        family_count=len(family_set),
        required_feature_count=sum(required_by_family.values()),
        errors=tuple(errors),
    )


def family_summary(data: dict[str, Any]) -> dict[str, dict[str, int]]:
    summary = {
        family: {"v1_required": 0, "v1_optional": 0, "later": 0, "total": 0}
        for family in data.get("families", [])
    }
    for feature in data.get("features", []):
        family = feature["family"]
        status = feature["status"]
        summary.setdefault(family, {"v1_required": 0, "v1_optional": 0, "later": 0, "total": 0})
        summary[family][status] += 1
        summary[family]["total"] += 1
    return summary


def features_by_family(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {family: [] for family in data.get("families", [])}
    for feature in data.get("features", []):
        grouped.setdefault(feature["family"], []).append(feature)
    return grouped
