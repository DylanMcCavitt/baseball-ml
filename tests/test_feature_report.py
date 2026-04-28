import json
from pathlib import Path

from mlb_props_lab.dashboard import build_dashboard
from mlb_props_lab.feature_registry import REQUIRED_FAMILIES
from mlb_props_lab.reports import generate_feature_report
from mlb_props_lab.statcast_features import build_statcast_feature_artifacts


def test_feature_report_writes_manifest_markdown_and_family_visuals(tmp_path: Path) -> None:
    report_root = generate_feature_report(
        issue="FEATURE-RESEARCH",
        output_dir=tmp_path,
        run_id="test-run",
    )

    manifest_path = report_root / "manifest.json"
    report_path = report_root / "report.md"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["report_type"] == "feature_registry"
    assert manifest["feature_count"] >= 30
    assert report_path.exists()
    assert "Source-Backed Feature Registry Report" in report_path.read_text(encoding="utf-8")

    visuals = [report_root / visual for visual in manifest["visuals"]]
    assert all(path.exists() for path in visuals)
    for family in REQUIRED_FAMILIES:
        assert report_root / "visuals" / f"{family}.svg" in visuals


def test_dashboard_indexes_feature_report_manifest(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    generate_feature_report(
        issue="FEATURE-RESEARCH",
        output_dir=reports_root,
        run_id="test-run",
    )

    dashboard_path = build_dashboard(
        reports_root=reports_root,
        output=tmp_path / "dashboard" / "index.html",
    )

    dashboard = dashboard_path.read_text(encoding="utf-8")
    assert "MLB Props Reboot Dashboard" in dashboard
    assert "FEATURE-RESEARCH" in dashboard
    assert "feature_registry" in dashboard


def test_feature_report_links_latest_statcast_feature_evidence(tmp_path: Path) -> None:
    build_statcast_feature_artifacts(issue="AGE-317", output_dir=tmp_path, run_id="sample")

    report_root = generate_feature_report(
        issue="AGE-317",
        output_dir=tmp_path,
        run_id="review",
    )

    manifest = json.loads((report_root / "manifest.json").read_text(encoding="utf-8"))
    report = (report_root / "report.md").read_text(encoding="utf-8")

    assert manifest["materialized_feature_run"]["run_id"] == "sample"
    assert "../sample/visuals/statcast_pitcher_skill.svg" in manifest["visuals"]
    assert "Materialized Statcast Feature Evidence" in report
    assert "projected_lineup_handedness_mix" in report
