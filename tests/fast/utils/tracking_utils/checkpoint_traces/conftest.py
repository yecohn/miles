from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def mlflow_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Run an MLflow run against a throwaway local SQLite store, restoring MLflow's global state afterwards."""
    mlflow = pytest.importorskip("mlflow")
    # synchronous export: each async flush waits out the exporter's 1 s queue poll, too slow for a fast test
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    try:
        experiment_id = mlflow.create_experiment("traces", artifact_location=(tmp_path / "artifacts").as_uri())
        with mlflow.start_run(experiment_id=experiment_id) as run:
            yield run
    finally:
        mlflow.flush_trace_async_logging()
        while mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous_uri)
