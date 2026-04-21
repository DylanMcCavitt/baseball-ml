from datetime import date

from mlb_props_stack.cli import main, render_runtime_summary
from mlb_props_stack.ingest import MLBMetadataIngestResult


def test_runtime_summary_includes_future_hooks():
    summary = render_runtime_summary()

    assert "MLB Props Stack" in summary
    assert "tracking_uri=file:./artifacts/mlruns" in summary
    assert "dashboard_module=mlb_props_stack.dashboard.app" in summary


def test_ingest_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = MLBMetadataIngestResult(
        target_date=date(2026, 4, 21),
        run_id="20260421T180000Z",
        schedule_raw_path=tmp_path / "schedule.json",
        feed_live_raw_paths=(tmp_path / "feed.json",),
        games_path=tmp_path / "games.jsonl",
        probable_starters_path=tmp_path / "probable_starters.jsonl",
        lineup_snapshots_path=tmp_path / "lineup_snapshots.jsonl",
        game_count=15,
        probable_starter_count=30,
        lineup_snapshot_count=30,
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.ingest_mlb_metadata_for_date",
        lambda *, target_date, output_dir: result,
    )

    main(
        [
            "ingest-mlb-metadata",
            "--date",
            "2026-04-21",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert "MLB metadata ingest complete for 2026-04-21" in output
    assert "games=15" in output
