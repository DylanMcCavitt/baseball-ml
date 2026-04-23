from __future__ import annotations

import json
from datetime import UTC, datetime

from mlb_props_stack.stage_gates import (
    STATUS_NEXT_MARKET,
    STATUS_RESEARCH_ONLY,
    evaluate_stage_gates,
    render_stage_gate_summary,
)
from tests.stage_gate_fixtures import seed_stage_gate_artifacts


def test_stage_gate_evaluator_keeps_current_like_artifacts_research_only(tmp_path):
    seed_stage_gate_artifacts(tmp_path, passing=False)
    fixed_now = lambda: datetime(2026, 4, 23, 22, 0, tzinfo=UTC)

    result = evaluate_stage_gates(output_dir=tmp_path, now=fixed_now)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    summary = render_stage_gate_summary(result)

    assert result.status == STATUS_RESEARCH_ONLY
    assert result.metrics["held_out_rows"].value == 48
    assert result.metrics["scoreable_backtest_rows"].value == 0
    assert result.metrics["backtest_skip_rate"].value == 1.0
    assert result.metrics["settled_paper_bets"].value == 0
    assert result.metrics["paper_same_line_clv_sample"].value == 0
    assert result.metrics["paper_roi"].value is None
    assert result.metrics["backtest_roi"].value is None
    assert "status=research_only" in summary
    assert "scoreable_backtest_rows=0" in summary
    assert "backtest_skip_rate=1" in summary
    assert "settled_paper_bets=0" in summary
    assert "paper_same_line_clv_sample=0" in summary
    assert "paper_roi=n/a" in summary
    assert "backtest_roi=n/a" in summary
    assert report["status"] == STATUS_RESEARCH_ONLY
    assert report["artifact_paths"]["training_summary_path"]
    assert report["artifact_paths"]["paper_results_path"]


def test_stage_gate_evaluator_promotes_synthetic_full_gate_set(tmp_path):
    seed_stage_gate_artifacts(tmp_path, passing=True)
    fixed_now = lambda: datetime(2026, 4, 23, 22, 5, tzinfo=UTC)

    result = evaluate_stage_gates(output_dir=tmp_path, now=fixed_now)

    assert result.status == STATUS_NEXT_MARKET
    assert all(gate.passed for gate in result.live_use_gates)
    assert all(gate.passed for gate in result.next_market_gates)
    assert result.metrics["held_out_rows"].value == 180
    assert result.metrics["scoreable_backtest_rows"].value == 300
    assert result.metrics["backtest_skip_rate"].value == 0.05
    assert result.metrics["settled_paper_bets"].value == 260
    assert result.metrics["paper_dates"].value == 65
    assert result.metrics["paper_same_line_clv_sample"].value == 260
    assert result.metrics["paper_beat_close_rate"].value == 0.538462
    assert result.metrics["paper_median_clv_probability_delta"].value == 0.02
    assert result.metrics["paper_roi"].value == 0.380769
    assert result.report_path.exists()
    assert result.report_markdown_path.exists()
