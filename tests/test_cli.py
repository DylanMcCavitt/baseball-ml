from datetime import date

from mlb_props_stack.cli import main, render_runtime_summary
from mlb_props_stack.ingest import (
    MLBMetadataIngestResult,
    OddsAPIIngestResult,
    StatcastFeatureIngestResult,
)
from mlb_props_stack.modeling import StarterStrikeoutBaselineTrainingResult


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


def test_odds_api_ingest_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = OddsAPIIngestResult(
        target_date=date(2026, 4, 21),
        run_id="20260421T180000Z",
        mlb_games_path=tmp_path / "games.jsonl",
        mlb_probable_starters_path=tmp_path / "probable_starters.jsonl",
        events_raw_path=tmp_path / "events.json",
        event_odds_raw_paths=(tmp_path / "event-1.json",),
        event_mappings_path=tmp_path / "event_game_mappings.jsonl",
        prop_line_snapshots_path=tmp_path / "prop_line_snapshots.jsonl",
        candidate_event_count=15,
        matched_event_count=14,
        unmatched_event_count=1,
        prop_line_count=28,
        skipped_prop_count=2,
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.ingest_odds_api_pitcher_lines_for_date",
        lambda *, target_date, output_dir, api_key: result,
    )

    main(
        [
            "ingest-odds-api-lines",
            "--date",
            "2026-04-21",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert "Odds API pitcher strikeout ingest complete for 2026-04-21" in output
    assert "candidate_events=15" in output
    assert "prop_line_snapshots=28" in output


def test_statcast_feature_ingest_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = StatcastFeatureIngestResult(
        target_date=date(2026, 4, 21),
        history_start_date=date(2026, 3, 22),
        history_end_date=date(2026, 4, 20),
        run_id="20260421T180000Z",
        mlb_games_path=tmp_path / "games.jsonl",
        mlb_probable_starters_path=tmp_path / "probable_starters.jsonl",
        mlb_lineup_snapshots_path=tmp_path / "lineup_snapshots.jsonl",
        pull_manifest_path=tmp_path / "pull_manifest.jsonl",
        pitch_level_base_path=tmp_path / "pitch_level_base.jsonl",
        pitcher_daily_features_path=tmp_path / "pitcher_daily_features.jsonl",
        lineup_daily_features_path=tmp_path / "lineup_daily_features.jsonl",
        game_context_features_path=tmp_path / "game_context_features.jsonl",
        raw_pull_count=11,
        pitch_level_record_count=1234,
        pitcher_feature_count=2,
        lineup_feature_count=2,
        game_context_feature_count=2,
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.ingest_statcast_features_for_date",
        lambda *, target_date, output_dir, history_days: result,
    )

    main(
        [
            "ingest-statcast-features",
            "--date",
            "2026-04-21",
            "--output-dir",
            str(tmp_path),
            "--history-days",
            "30",
        ]
    )
    output = capsys.readouterr().out

    assert "Statcast feature build complete for 2026-04-21" in output
    assert "raw_pulls=11" in output
    assert "pitch_level_rows=1234" in output


def test_starter_strikeout_training_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = StarterStrikeoutBaselineTrainingResult(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 20),
        run_id="20260421T180000Z",
        row_count=48,
        outcome_count=48,
        dispersion_alpha=0.183742,
        dataset_path=tmp_path / "training_dataset.jsonl",
        outcomes_path=tmp_path / "starter_outcomes.jsonl",
        date_splits_path=tmp_path / "date_splits.json",
        model_path=tmp_path / "baseline_model.json",
        evaluation_path=tmp_path / "evaluation.json",
        ladder_probabilities_path=tmp_path / "ladder_probabilities.jsonl",
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.train_starter_strikeout_baseline",
        lambda *, start_date, end_date, output_dir: result,
    )

    main(
        [
            "train-starter-strikeout-baseline",
            "--start-date",
            "2026-04-01",
            "--end-date",
            "2026-04-20",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert "Starter strikeout baseline training complete for 2026-04-01 -> 2026-04-20" in output
    assert "training_rows=48" in output
    assert "dispersion_alpha=0.183742" in output
    assert "model_path=" in output
    assert "ladder_probabilities_path=" in output
