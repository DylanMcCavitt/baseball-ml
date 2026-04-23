from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def seed_stage_gate_artifacts(output_dir: Path, *, passing: bool) -> dict[str, Path]:
    model_run_id = "20260422T205727Z" if not passing else "20260422T220000Z"
    backtest_run_id = "20260422T205734Z" if not passing else "20260422T221000Z"
    training_dir = (
        output_dir
        / "normalized"
        / "starter_strikeout_baseline"
        / "start=2026-04-01_end=2026-04-23"
        / f"run={model_run_id}"
    )
    training_path = training_dir / "evaluation_summary.json"
    _write_json(
        training_path,
        {
            "run_id": model_run_id,
            "row_counts": {
                "train": 220 if passing else 60,
                "validation": 60 if passing else 20,
                "test": 60 if passing else 28,
                "held_out": 180 if passing else 48,
            },
            "held_out_performance": {
                "beats_benchmark": {"rmse": True, "mae": True},
                "model": {"rmse": 1.8, "mae": 1.3},
                "benchmark": {"rmse": 2.2, "mae": 1.7},
            },
            "held_out_probability_calibration": {
                "calibrated": {
                    "expected_calibration_error": 0.02 if passing else 0.02285
                }
            },
        },
    )

    backtest_dir = (
        output_dir
        / "normalized"
        / "walk_forward_backtest"
        / "start=2026-04-01_end=2026-04-23"
        / f"run={backtest_run_id}"
    )
    backtest_runs_path = backtest_dir / "backtest_runs.jsonl"
    _write_jsonl(
        backtest_runs_path,
        [
            {
                "backtest_run_id": backtest_run_id,
                "model_run_id": model_run_id,
                "row_counts": (
                    {
                        "snapshot_groups": 320,
                        "actionable": 220,
                        "below_threshold": 80,
                        "skipped": 16,
                    }
                    if passing
                    else {
                        "snapshot_groups": 139,
                        "actionable": 0,
                        "below_threshold": 0,
                        "skipped": 139,
                    }
                ),
                "bet_outcomes": (
                    {"placed_bets": 180, "roi": 0.12}
                    if passing
                    else {"placed_bets": 0, "roi": None}
                ),
            }
        ],
    )
    _write_jsonl(
        backtest_dir / "clv_summary.jsonl",
        [
            {
                "backtest_run_id": backtest_run_id,
                "summary_scope": "overall",
                "sample_count": 180 if passing else 0,
                "median_probability_delta": 0.018 if passing else None,
            }
        ],
    )
    _write_jsonl(
        backtest_dir / "roi_summary.jsonl",
        [
            {
                "backtest_run_id": backtest_run_id,
                "summary_scope": "overall",
                "placed_bets": 180 if passing else 0,
                "roi": 0.12 if passing else None,
            }
        ],
    )

    paper_path = (
        output_dir
        / "normalized"
        / "paper_results"
        / "date=2026-04-23"
        / f"run={backtest_run_id}"
        / "paper_results.jsonl"
    )
    if passing:
        start_date = date(2026, 2, 18)
        paper_rows = []
        for index in range(260):
            official_date = start_date + timedelta(days=index % 65)
            beat_close = index < 140
            won = index < 170
            paper_rows.append(
                {
                    "paper_result_id": f"paper-{index}",
                    "official_date": official_date.isoformat(),
                    "settlement_status": "win" if won else "loss",
                    "same_line_close_available": True,
                    "beat_closing_line": beat_close,
                    "clv_probability_delta": 0.02 if beat_close else -0.01,
                    "stake_fraction": 0.01,
                    "profit_units": 0.009 if won else -0.006,
                }
            )
        _write_jsonl(paper_path, paper_rows)
    else:
        _write_jsonl(paper_path, [])

    return {
        "training_summary_path": training_path,
        "backtest_run_dir": backtest_dir,
        "backtest_runs_path": backtest_runs_path,
        "paper_results_path": paper_path,
    }
