from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from html import escape
from importlib import resources
from pathlib import Path
from typing import Any

from mlb_props_lab.feature_registry import load_registry, validate_registry

MATERIALIZED_STATCAST_FEATURE_IDS = (
    "pitcher_k_rate_rolling",
    "pitcher_k9_rolling",
    "pitcher_k_minus_bb_rate",
    "pitcher_batters_faced_rolling",
    "pitcher_pitches_per_start",
    "pitcher_csw_rate",
    "pitcher_whiff_rate",
    "pitch_mix_by_type",
    "release_velocity_by_type",
    "release_spin_rate_by_type",
    "pitch_movement_by_type",
    "pitch_type_whiff_csw_contact",
    "pitcher_platoon_k_bb_whiff",
    "pitcher_platoon_pitch_mix",
)

GAP_STATCAST_FEATURE_IDS = ("projected_lineup_handedness_mix",)
IN_SCOPE_STATCAST_FEATURE_IDS = MATERIALIZED_STATCAST_FEATURE_IDS + GAP_STATCAST_FEATURE_IDS

STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
WALK_EVENTS = {"walk", "intent_walk"}
CALLED_STRIKE_DESCRIPTIONS = {"called_strike"}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
SWING_DESCRIPTIONS = WHIFF_DESCRIPTIONS | {
    "foul",
    "foul_tip",
    "foul_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}
CONTACT_DESCRIPTIONS = SWING_DESCRIPTIONS - WHIFF_DESCRIPTIONS

OUTS_BY_EVENT = {
    "strikeout": 1,
    "field_out": 1,
    "force_out": 1,
    "fielders_choice_out": 1,
    "sac_fly": 1,
    "sac_bunt": 1,
    "grounded_into_double_play": 2,
    "double_play": 2,
    "strikeout_double_play": 2,
}


@dataclass(frozen=True)
class StatcastFeatureBuild:
    report_root: Path
    manifest_path: Path
    feature_matrix_path: Path
    coverage_path: Path
    skipped_rows_path: Path
    visual_paths: tuple[Path, ...]


def default_pitch_sample_path() -> Path:
    path = resources.files("mlb_props_lab.resources").joinpath("statcast_pitch_sample.csv")
    return Path(str(path))


def default_target_sample_path() -> Path:
    path = resources.files("mlb_props_lab.resources").joinpath("statcast_targets_sample.csv")
    return Path(str(path))


def build_statcast_feature_artifacts(
    issue: str,
    output_dir: str | Path = "artifacts/reports",
    run_id: str | None = None,
    pitches_path: str | Path | None = None,
    targets_path: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> StatcastFeatureBuild:
    registry = load_registry(registry_path)
    validation = validate_registry(registry)
    if not validation.ok:
        joined = "\n".join(validation.errors)
        raise ValueError(f"feature registry is invalid:\n{joined}")
    _validate_registered_scope(registry)

    pitches = _read_csv(Path(pitches_path) if pitches_path else default_pitch_sample_path())
    targets = _read_csv(Path(targets_path) if targets_path else default_target_sample_path())

    run = run_id or _make_run_id()
    report_root = Path(output_dir) / issue / run
    visuals_root = report_root / "visuals"
    report_root.mkdir(parents=True, exist_ok=True)
    visuals_root.mkdir(parents=True, exist_ok=True)

    feature_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    registry_by_id = {feature["id"]: feature for feature in registry["features"]}

    for target in targets:
        materialized = materialize_statcast_features_for_target(pitches, target)
        feature_rows.append(_feature_csv_row(target, materialized))
        coverage_rows.extend(_coverage_rows(target, materialized, registry_by_id))
        skipped_rows.extend(_skipped_rows(target, materialized))

    feature_matrix_path = report_root / "statcast_features.csv"
    coverage_path = report_root / "statcast_feature_coverage.csv"
    skipped_rows_path = report_root / "statcast_skipped_rows.csv"
    _write_csv(feature_matrix_path, feature_rows)
    _write_csv(coverage_path, coverage_rows)
    _write_csv(skipped_rows_path, skipped_rows)

    visual_paths = (
        visuals_root / "statcast_feature_presence.svg",
        visuals_root / "statcast_pitcher_skill.svg",
        visuals_root / "statcast_arsenal_stuff.svg",
        visuals_root / "statcast_handedness_platoon.svg",
        visuals_root / "statcast_pitch_mix_coverage.svg",
    )
    _write_presence_svg(visual_paths[0], coverage_rows)
    _write_pitcher_skill_svg(visual_paths[1], feature_rows)
    _write_arsenal_svg(visual_paths[2], feature_rows)
    _write_handedness_svg(visual_paths[3], feature_rows)
    _write_pitch_mix_svg(visual_paths[4], feature_rows)

    manifest_path = report_root / "statcast_feature_manifest.json"
    manifest = {
        "schema_version": "2026-04-28.1",
        "report_type": "statcast_feature_build",
        "issue": issue,
        "run_id": run,
        "generated_at": datetime.now(UTC).isoformat(),
        "pitch_input": str(Path(pitches_path) if pitches_path else default_pitch_sample_path()),
        "target_input": str(Path(targets_path) if targets_path else default_target_sample_path()),
        "target_count": len(targets),
        "materialized_feature_ids": list(MATERIALIZED_STATCAST_FEATURE_IDS),
        "gap_feature_ids": list(GAP_STATCAST_FEATURE_IDS),
        "feature_matrix": feature_matrix_path.name,
        "coverage": coverage_path.name,
        "skipped_rows": skipped_rows_path.name,
        "visuals": [path.relative_to(report_root).as_posix() for path in visual_paths],
        "limitations": [
            (
                "Sample fixture uses prior Statcast pitch rows only; live Statcast fetch is "
                "out of scope."
            ),
            (
                "Projected lineup handedness is reported as a source gap because no lineup "
                "snapshot is present."
            ),
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return StatcastFeatureBuild(
        report_root=report_root,
        manifest_path=manifest_path,
        feature_matrix_path=feature_matrix_path,
        coverage_path=coverage_path,
        skipped_rows_path=skipped_rows_path,
        visual_paths=visual_paths,
    )


def materialize_statcast_features_for_target(
    pitches: list[dict[str, str]],
    target: dict[str, str],
) -> dict[str, Any]:
    pitcher = _required(target, "pitcher")
    target_date = _parse_date(_required(target, "target_game_date"))
    target_cutoff = _parse_datetime(target.get("cutoff_at", ""))

    pitcher_rows = [row for row in pitches if row.get("pitcher") == pitcher]
    eligible_rows = [
        row
        for row in pitcher_rows
        if _row_is_before_cutoff(row, target_date=target_date, target_cutoff=target_cutoff)
    ]
    skipped_same_or_future = len(pitcher_rows) - len(eligible_rows)

    plate_appearances = [row for row in eligible_rows if row.get("events", "").strip()]
    strikeouts = sum(1 for row in plate_appearances if row.get("events") in STRIKEOUT_EVENTS)
    walks = sum(1 for row in plate_appearances if row.get("events") in WALK_EVENTS)
    outs = sum(_outs_for_event(row.get("events", "")) for row in plate_appearances)
    innings = outs / 3 if outs else 0
    games = _group_rows(eligible_rows, "game_pk")
    plate_appearances_by_game = _group_rows(plate_appearances, "game_pk")
    pitch_count_by_game = _group_rows(eligible_rows, "game_pk")

    features: dict[str, Any] = {
        "pitcher_k_rate_rolling": _rate(strikeouts, len(plate_appearances)),
        "pitcher_k9_rolling": _round(strikeouts * 9 / innings) if innings else None,
        "pitcher_k_minus_bb_rate": _rate(strikeouts - walks, len(plate_appearances)),
        "pitcher_batters_faced_rolling": _mean(
            [len(rows) for rows in plate_appearances_by_game.values()]
        ),
        "pitcher_pitches_per_start": _mean([len(rows) for rows in pitch_count_by_game.values()]),
        "pitcher_csw_rate": _csw_rate(eligible_rows),
        "pitcher_whiff_rate": _whiff_rate(eligible_rows),
        "pitch_mix_by_type": _pitch_mix(eligible_rows),
        "release_velocity_by_type": _mean_by_pitch_type(eligible_rows, "release_speed"),
        "release_spin_rate_by_type": _spin_by_pitch_type(eligible_rows),
        "pitch_movement_by_type": _movement_by_pitch_type(eligible_rows),
        "pitch_type_whiff_csw_contact": _pitch_type_outcomes(eligible_rows),
        "pitcher_platoon_k_bb_whiff": _platoon_rates(eligible_rows),
        "pitcher_platoon_pitch_mix": _platoon_pitch_mix(eligible_rows),
    }

    return {
        "target": target,
        "features": features,
        "source_pitch_count": len(eligible_rows),
        "source_plate_appearance_count": len(plate_appearances),
        "source_game_count": len(games),
        "skipped_same_or_future_pitch_count": skipped_same_or_future,
        "cutoff_rule": (
            "include rows with available_at before cutoff_at; if unavailable, use game_date "
            "strictly before target_game_date"
        ),
    }


def _validate_registered_scope(registry: dict[str, Any]) -> None:
    registry_by_id = {feature["id"]: feature for feature in registry["features"]}
    missing = sorted(set(IN_SCOPE_STATCAST_FEATURE_IDS) - set(registry_by_id))
    if missing:
        raise ValueError(f"Statcast materialization references unregistered features: {missing}")

    invalid_statuses = [
        feature_id
        for feature_id in IN_SCOPE_STATCAST_FEATURE_IDS
        if registry_by_id[feature_id]["status"] != "v1_required"
    ]
    if invalid_statuses:
        raise ValueError(
            "Statcast materialization only supports registered v1_required features; "
            f"got {invalid_statuses}"
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _feature_csv_row(target: dict[str, str], materialized: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "target_game_pk": target.get("target_game_pk", ""),
        "pitcher": target.get("pitcher", ""),
        "pitcher_name": target.get("pitcher_name", ""),
        "target_game_date": target.get("target_game_date", ""),
        "cutoff_at": target.get("cutoff_at", ""),
        "source_pitch_count": materialized["source_pitch_count"],
        "source_plate_appearance_count": materialized["source_plate_appearance_count"],
        "source_game_count": materialized["source_game_count"],
        "skipped_same_or_future_pitch_count": materialized["skipped_same_or_future_pitch_count"],
    }
    row.update(materialized["features"])
    return row


def _coverage_rows(
    target: dict[str, str],
    materialized: dict[str, Any],
    registry_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    features = materialized["features"]
    for feature_id in IN_SCOPE_STATCAST_FEATURE_IDS:
        feature = registry_by_id[feature_id]
        value = features.get(feature_id)
        materialized_value = _has_value(value)
        reason = ""
        if feature_id == "projected_lineup_handedness_mix":
            reason = "requires projected lineup snapshot; not present in Statcast pitch fixture"
        elif not materialized_value:
            reason = "insufficient prior Statcast rows before cutoff"
        rows.append(
            {
                "target_game_pk": target.get("target_game_pk", ""),
                "pitcher": target.get("pitcher", ""),
                "feature_id": feature_id,
                "family": feature["family"],
                "status": feature["status"],
                "materialized": "yes" if materialized_value else "no",
                "source_pitch_count": materialized["source_pitch_count"],
                "source_plate_appearance_count": materialized["source_plate_appearance_count"],
                "missing_reason": reason,
                "required_visual": feature["required_visual"],
            }
        )
    return rows


def _skipped_rows(target: dict[str, str], materialized: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "target_game_pk": target.get("target_game_pk", ""),
            "pitcher": target.get("pitcher", ""),
            "reason": "same_or_future_statcast_rows_excluded_by_cutoff",
            "row_count": materialized["skipped_same_or_future_pitch_count"],
            "cutoff_rule": materialized["cutoff_rule"],
        }
    ]


def _row_is_before_cutoff(
    row: dict[str, str],
    *,
    target_date: date,
    target_cutoff: datetime | None,
) -> bool:
    available_at = _parse_datetime(row.get("available_at", ""))
    if target_cutoff and available_at:
        return available_at < target_cutoff
    return _parse_date(row.get("game_date", "")) < target_date


def _parse_date(value: str) -> date:
    if not value:
        msg = "missing required date value"
        raise ValueError(msg)
    return date.fromisoformat(value)


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _required(row: dict[str, str], key: str) -> str:
    value = row.get(key, "").strip()
    if not value:
        raise ValueError(f"missing required field: {key}")
    return value


def _group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "")].append(row)
    return dict(grouped)


def _outs_for_event(event: str) -> int:
    return OUTS_BY_EVENT.get(event, 0)


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return _round(numerator / denominator)


def _round(value: float, digits: int = 6) -> float:
    if math.isnan(value) or math.isinf(value):
        msg = f"invalid numeric feature value: {value}"
        raise ValueError(msg)
    return round(value, digits)


def _mean(values: list[int | float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return _round(sum(numeric) / len(numeric), 3)


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    if not value:
        return None
    return float(value)


def _description(row: dict[str, str]) -> str:
    return row.get("description", "").strip()


def _pitch_type(row: dict[str, str]) -> str:
    return row.get("pitch_type", "").strip() or "UNK"


def _stand(row: dict[str, str]) -> str:
    return row.get("stand", "").strip() or "UNK"


def _csw_rate(rows: list[dict[str, str]]) -> float | None:
    csw = sum(
        1
        for row in rows
        if _description(row) in CALLED_STRIKE_DESCRIPTIONS | WHIFF_DESCRIPTIONS
    )
    return _rate(csw, len(rows))


def _whiff_rate(rows: list[dict[str, str]]) -> float | None:
    swings = [row for row in rows if _description(row) in SWING_DESCRIPTIONS]
    whiffs = sum(1 for row in swings if _description(row) in WHIFF_DESCRIPTIONS)
    return _rate(whiffs, len(swings))


def _pitch_mix(rows: list[dict[str, str]]) -> dict[str, float]:
    total = len(rows)
    if not total:
        return {}
    counts = Counter(_pitch_type(row) for row in rows)
    return {pitch_type: _rate(count, total) for pitch_type, count in sorted(counts.items())}


def _mean_by_pitch_type(rows: list[dict[str, str]], field: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _float(row, field)
        if value is not None:
            values[_pitch_type(row)].append(value)
    return {pitch_type: _mean(numbers) for pitch_type, numbers in sorted(values.items())}


def _spin_by_pitch_type(rows: list[dict[str, str]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _float(row, "release_spin_rate")
        if value is None:
            value = _float(row, "release_spin")
        if value is not None:
            values[_pitch_type(row)].append(value)
    return {pitch_type: _mean(numbers) for pitch_type, numbers in sorted(values.items())}


def _movement_by_pitch_type(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    movement: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"pfx_x": [], "pfx_z": []})
    for row in rows:
        pfx_x = _float(row, "pfx_x")
        pfx_z = _float(row, "pfx_z")
        pitch_type = _pitch_type(row)
        if pfx_x is not None:
            movement[pitch_type]["pfx_x"].append(pfx_x)
        if pfx_z is not None:
            movement[pitch_type]["pfx_z"].append(pfx_z)
    return {
        pitch_type: {
            "pfx_x": _mean(values["pfx_x"]),
            "pfx_z": _mean(values["pfx_z"]),
        }
        for pitch_type, values in sorted(movement.items())
    }


def _pitch_type_outcomes(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped = _group_rows(rows, "pitch_type")
    outcomes = {}
    for pitch_type, pitch_rows in sorted(grouped.items()):
        swings = [row for row in pitch_rows if _description(row) in SWING_DESCRIPTIONS]
        whiffs = sum(1 for row in swings if _description(row) in WHIFF_DESCRIPTIONS)
        contact = sum(1 for row in swings if _description(row) in CONTACT_DESCRIPTIONS)
        csw = sum(
            1
            for row in pitch_rows
            if _description(row) in CALLED_STRIKE_DESCRIPTIONS | WHIFF_DESCRIPTIONS
        )
        outcomes[pitch_type or "UNK"] = {
            "pitches": len(pitch_rows),
            "csw_rate": _rate(csw, len(pitch_rows)),
            "whiff_rate": _rate(whiffs, len(swings)),
            "contact_rate": _rate(contact, len(swings)),
        }
    return outcomes


def _platoon_rates(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped = _group_rows(rows, "stand")
    splits = {}
    for stand, split_rows in sorted(grouped.items()):
        plate_appearances = [row for row in split_rows if row.get("events", "").strip()]
        strikeouts = sum(1 for row in plate_appearances if row.get("events") in STRIKEOUT_EVENTS)
        walks = sum(1 for row in plate_appearances if row.get("events") in WALK_EVENTS)
        swings = [row for row in split_rows if _description(row) in SWING_DESCRIPTIONS]
        whiffs = sum(1 for row in swings if _description(row) in WHIFF_DESCRIPTIONS)
        splits[stand or "UNK"] = {
            "plate_appearances": len(plate_appearances),
            "k_rate": _rate(strikeouts, len(plate_appearances)),
            "bb_rate": _rate(walks, len(plate_appearances)),
            "whiff_rate": _rate(whiffs, len(swings)),
        }
    return splits


def _platoon_pitch_mix(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped = _group_rows(rows, "stand")
    return {stand or "UNK": _pitch_mix(split_rows) for stand, split_rows in sorted(grouped.items())}


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value) and any(_has_value(child) for child in value.values())
    if isinstance(value, list):
        return bool(value) and any(_has_value(child) for child in value)
    return True


def _csv_value(value: Any) -> str | int | float:
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value


def _json_cell(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if isinstance(value, str):
        return json.loads(value) if value else None
    return value


def _make_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    return current.strftime("%Y%m%dT%H%M%SZ")


def _write_presence_svg(path: Path, coverage_rows: list[dict[str, Any]]) -> None:
    by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in coverage_rows:
        by_feature[row["feature_id"]].append(row)
    width = 980
    row_height = 28
    height = 72 + row_height * len(IN_SCOPE_STATCAST_FEATURE_IDS)
    rows = []
    y = 54
    for feature_id in IN_SCOPE_STATCAST_FEATURE_IDS:
        rows_for_feature = by_feature[feature_id]
        total = len(rows_for_feature)
        present = sum(1 for row in rows_for_feature if row["materialized"] == "yes")
        bar_width = int(440 * present / max(total, 1))
        color = "#2364aa" if present == total else "#f2a541" if present else "#c44536"
        rows.append(f'<text x="24" y="{y + 16}" class="label">{escape(feature_id)}</text>')
        rows.append(f'<rect x="390" y="{y}" width="440" height="18" class="track"/>')
        rows.append(
            f'<rect x="390" y="{y}" width="{bar_width}" height="18" fill="{color}"/>'
        )
        rows.append(f'<text x="850" y="{y + 14}" class="count">{present}/{total}</text>')
        y += row_height
    path.write_text(
        _svg_shell(
            width,
            height,
            "Statcast Feature Presence",
            "".join(rows),
            extra_css=".track { fill: #edf1f4; }",
        ),
        encoding="utf-8",
    )


def _write_pitcher_skill_svg(path: Path, feature_rows: list[dict[str, Any]]) -> None:
    labels = [
        ("pitcher_k_rate_rolling", "K rate"),
        ("pitcher_csw_rate", "CSW"),
        ("pitcher_whiff_rate", "Whiff"),
        ("pitcher_k_minus_bb_rate", "K-BB"),
    ]
    width = 980
    height = 84 + 78 * len(feature_rows)
    blocks = []
    y = 58
    for row in feature_rows:
        title = f"{row['pitcher_name']} {row['target_game_date']}"
        blocks.append(f'<text x="24" y="{y}" class="subtitle">{escape(title)}</text>')
        x = 260
        for key, label in labels:
            value = row.get(key)
            rate_value = float(value) if value not in (None, "") else 0
            bar_height = int(52 * max(min(rate_value, 1), 0))
            blocks.append(f'<text x="{x}" y="{y}" class="small">{escape(label)}</text>')
            blocks.append(
                f'<rect x="{x}" y="{y + 10}" width="58" height="52" class="track"/>'
                f'<rect x="{x}" y="{y + 10 + 52 - bar_height}" width="58" '
                f'height="{bar_height}" class="required"/>'
            )
            blocks.append(f'<text x="{x}" y="{y + 78}" class="small">{rate_value:.3f}</text>')
            x += 92
        y += 78
    path.write_text(
        _svg_shell(width, height, "Pitcher Skill: K Rate, CSW, Whiff", "".join(blocks)),
        encoding="utf-8",
    )


def _write_arsenal_svg(path: Path, feature_rows: list[dict[str, Any]]) -> None:
    width = 980
    height = 88 + 96 * len(feature_rows)
    blocks = []
    y = 58
    for row in feature_rows:
        velocity = _json_cell(row, "release_velocity_by_type") or {}
        spin = _json_cell(row, "release_spin_rate_by_type") or {}
        movement = _json_cell(row, "pitch_movement_by_type") or {}
        pitch_types = sorted(set(velocity) | set(spin) | set(movement))
        blocks.append(
            f'<text x="24" y="{y}" class="subtitle">'
            f'{escape(row["pitcher_name"])} velocity/spin coverage</text>'
        )
        x = 300
        for pitch_type in pitch_types:
            velocity_present = pitch_type in velocity
            spin_present = pitch_type in spin
            movement_present = pitch_type in movement
            blocks.append(f'<text x="{x}" y="{y}" class="small">{escape(pitch_type)}</text>')
            blocks.append(
                f'<circle cx="{x + 8}" cy="{y + 24}" r="7" '
                f'fill="{"#2364aa" if velocity_present else "#d7dee5"}"/>'
            )
            blocks.append(
                f'<circle cx="{x + 34}" cy="{y + 24}" r="7" '
                f'fill="{"#3da35d" if spin_present else "#d7dee5"}"/>'
            )
            blocks.append(
                f'<circle cx="{x + 60}" cy="{y + 24}" r="7" '
                f'fill="{"#f2a541" if movement_present else "#d7dee5"}"/>'
            )
            x += 90
        blocks.append(
            '<text x="24" y="44" class="small">'
            "blue velocity, green spin, gold movement</text>"
        )
        y += 96
    path.write_text(
        _svg_shell(width, height, "Arsenal And Stuff: Velocity, Spin, Movement", "".join(blocks)),
        encoding="utf-8",
    )


def _write_handedness_svg(path: Path, feature_rows: list[dict[str, Any]]) -> None:
    width = 980
    height = 88 + 86 * len(feature_rows)
    blocks = []
    y = 58
    for row in feature_rows:
        splits = _json_cell(row, "pitcher_platoon_k_bb_whiff") or {}
        blocks.append(f'<text x="24" y="{y}" class="subtitle">{escape(row["pitcher_name"])}</text>')
        x = 300
        for stand in ("L", "R", "S", "UNK"):
            if stand not in splits:
                continue
            split = splits[stand]
            k_rate = split.get("k_rate") or 0
            whiff_rate = split.get("whiff_rate") or 0
            blocks.append(f'<text x="{x}" y="{y}" class="small">{escape(stand)}HB</text>')
            blocks.append(
                f'<rect x="{x}" y="{y + 12}" width="{int(120 * k_rate)}" height="14" '
                'class="required"/>'
            )
            blocks.append(
                f'<rect x="{x}" y="{y + 34}" width="{int(120 * whiff_rate)}" height="14" '
                'class="optional"/>'
            )
            blocks.append(f'<text x="{x + 128}" y="{y + 24}" class="small">K {k_rate:.3f}</text>')
            blocks.append(
                f'<text x="{x + 128}" y="{y + 46}" class="small">Whiff {whiff_rate:.3f}</text>'
            )
            x += 220
        y += 86
    path.write_text(
        _svg_shell(width, height, "Handedness And Platoon Coverage", "".join(blocks)),
        encoding="utf-8",
    )


def _write_pitch_mix_svg(path: Path, feature_rows: list[dict[str, Any]]) -> None:
    width = 980
    height = 84 + 70 * len(feature_rows)
    colors = ["#2364aa", "#3da35d", "#f2a541", "#c44536", "#6f5cc2", "#607d8b"]
    blocks = []
    y = 58
    for row in feature_rows:
        mix = _json_cell(row, "pitch_mix_by_type") or {}
        blocks.append(
            f'<text x="24" y="{y + 16}" class="subtitle">'
            f'{escape(row["pitcher_name"])}</text>'
        )
        x = 300
        color_index = 0
        for pitch_type, share in sorted(mix.items()):
            width_value = int(440 * float(share))
            color = colors[color_index % len(colors)]
            blocks.append(
                f'<rect x="{x}" y="{y}" width="{width_value}" height="22" fill="{color}"/>'
            )
            if width_value >= 36:
                blocks.append(
                    f'<text x="{x + 4}" y="{y + 16}" class="barlabel">{escape(pitch_type)}</text>'
                )
            x += width_value
            color_index += 1
        blocks.append(f'<rect x="300" y="{y}" width="440" height="22" class="outline"/>')
        y += 70
    path.write_text(
        _svg_shell(width, height, "Pitch Mix By Type Coverage", "".join(blocks)),
        encoding="utf-8",
    )


def _svg_shell(width: int, height: int, title: str, body: str, extra_css: str = "") -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg"
  width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 22px system-ui, sans-serif; fill: #17202a; }}
    .subtitle {{ font: 600 13px system-ui, sans-serif; fill: #243447; }}
    .label {{ font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; fill: #243447; }}
    .small {{ font: 11px system-ui, sans-serif; fill: #566573; }}
    .count {{ font: 12px system-ui, sans-serif; fill: #243447; }}
    .required {{ fill: #2364aa; }}
    .optional {{ fill: #3da35d; }}
    .track {{ fill: #edf1f4; }}
    .outline {{ fill: none; stroke: #243447; stroke-width: 1; }}
    .barlabel {{ font: 10px system-ui, sans-serif; fill: #ffffff; }}
    {extra_css}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="24" y="30" class="title">{escape(title)}</text>
  {body}
</svg>
"""


def relative_path(target: Path, base: Path) -> str:
    return os.path.relpath(target, start=base).replace(os.sep, "/")
