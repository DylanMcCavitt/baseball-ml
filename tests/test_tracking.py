from mlb_props_stack.dashboard.app import build_dashboard_banner
from pathlib import Path

from mlb_props_stack.tracking import (
    TrackingConfig,
    log_run_artifact,
    log_run_metrics,
    log_run_params,
    start_experiment_run,
)


class _FakeRunInfo:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id


class _FakeActiveRun:
    def __init__(self, run_id: str) -> None:
        self.info = _FakeRunInfo(run_id)

    def __enter__(self) -> "_FakeActiveRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri = None
        self.experiment_name = None
        self.run_name = None
        self.tags = None
        self.params = None
        self.metrics = None
        self.logged_file = None
        self.logged_dir = None

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def start_run(self, *, run_name: str) -> _FakeActiveRun:
        self.run_name = run_name
        return _FakeActiveRun("mlflow-run-123")

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags = tags

    def log_params(self, params: dict[str, str]) -> None:
        self.params = params

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics = metrics

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.logged_file = (path, artifact_path)

    def log_artifacts(self, path: str, artifact_path: str | None = None) -> None:
        self.logged_dir = (path, artifact_path)


def test_tracking_config_uses_reserved_mlflow_location():
    config = TrackingConfig()

    assert config.tracking_uri == "file:./artifacts/mlruns"
    assert config.training_experiment_name == "mlb-props-stack-starter-strikeout-training"
    assert config.backtest_experiment_name == "mlb-props-stack-walk-forward-backtest"
    assert config.dashboard_module == "mlb_props_stack.dashboard.app"


def test_tracking_helpers_start_and_log_to_mlflow(monkeypatch, tmp_path):
    fake_mlflow = _FakeMlflow()
    tracking_root = tmp_path / "mlruns"
    config = TrackingConfig(tracking_uri=f"file:{tracking_root}")
    artifact_file = tmp_path / "artifact.txt"
    artifact_file.write_text("artifact", encoding="utf-8")
    artifact_dir = tmp_path / "artifact_dir"
    artifact_dir.mkdir()

    monkeypatch.setattr(
        "mlb_props_stack.tracking.import_module",
        lambda name: fake_mlflow,
    )

    with start_experiment_run(
        experiment_name=config.training_experiment_name,
        run_name="training-run",
        tags={"run_kind": "training"},
        config=config,
    ) as run:
        log_run_params({"path": Path("data"), "keep": True})
        log_run_metrics({"rmse": 2.123456, "missing": None})
        log_run_artifact(artifact_file, artifact_path="files")
        log_run_artifact(artifact_dir, artifact_path="dirs")

    assert run.run_id == "mlflow-run-123"
    assert fake_mlflow.tracking_uri == config.tracking_uri
    assert fake_mlflow.experiment_name == config.training_experiment_name
    assert fake_mlflow.run_name == "training-run"
    assert fake_mlflow.tags == {"run_kind": "training"}
    assert fake_mlflow.params == {"path": "data", "keep": "true"}
    assert fake_mlflow.metrics == {"rmse": 2.123456}
    assert fake_mlflow.logged_file == (str(artifact_file), "files")
    assert fake_mlflow.logged_dir == (str(artifact_dir), "dirs")
    assert tracking_root.exists()


def test_dashboard_banner_marks_placeholder_module():
    assert "current slate candidates" in build_dashboard_banner()
