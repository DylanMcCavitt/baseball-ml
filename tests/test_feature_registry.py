from mlb_props_lab.feature_registry import (
    REQUIRED_FAMILIES,
    load_registry,
    validate_registry,
)


def test_default_feature_registry_is_valid() -> None:
    registry = load_registry()

    result = validate_registry(registry)

    assert result.ok, result.errors
    assert result.family_count == len(REQUIRED_FAMILIES)
    assert result.feature_count >= 30
    assert result.required_feature_count >= len(REQUIRED_FAMILIES)


def test_registry_includes_user_requested_feature_signals() -> None:
    registry = load_registry()
    feature_text = " ".join(
        " ".join(str(value) for value in feature.values()) for feature in registry["features"]
    ).lower()

    assert "spin" in feature_text
    assert "velocity" in feature_text
    assert "handedness" in feature_text or "lhb" in feature_text
    assert "pitch mix" in feature_text
    assert "sportsbook" in feature_text


def test_every_feature_declares_timestamp_and_visual_policy() -> None:
    registry = load_registry()

    for feature in registry["features"]:
        cutoff = feature["timestamp_cutoff"].lower()
        assert any(token in cutoff for token in ("before", "pre", "not available"))
        assert feature["required_visual"].strip()
        assert feature["missing_policy"].strip()
        assert feature["leakage_risk"].strip()


def test_all_required_families_have_v1_required_features() -> None:
    registry = load_registry()

    required_families = {
        feature["family"] for feature in registry["features"] if feature["status"] == "v1_required"
    }

    assert REQUIRED_FAMILIES <= required_families
