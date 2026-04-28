from pathlib import Path

from mlb_props_lab.cli import main


def test_cli_validates_feature_registry(capsys) -> None:
    exit_code = main(["feature-registry", "validate"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Feature registry valid" in captured.out


def test_cli_generates_feature_report_and_dashboard(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    dashboard_path = tmp_path / "dashboard" / "index.html"

    report_exit = main(
        [
            "report",
            "features",
            "--issue",
            "FEATURE-RESEARCH",
            "--output-dir",
            str(reports_root),
            "--run-id",
            "test-run",
        ]
    )
    dashboard_exit = main(
        [
            "dashboard",
            "--reports-root",
            str(reports_root),
            "--output",
            str(dashboard_path),
        ]
    )

    assert report_exit == 0
    assert dashboard_exit == 0
    assert (reports_root / "FEATURE-RESEARCH" / "test-run" / "manifest.json").exists()
    assert dashboard_path.exists()
