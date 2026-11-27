from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from tests.fast.utils.tracking_utils.checkpoint_traces.fixtures import make_sample

from miles.utils.tracking_utils.checkpoint_traces.mlflow_sink import MlflowTableSink
from miles.utils.tracking_utils.checkpoint_traces.record import TraceRecord

mlflow = pytest.importorskip("mlflow")


@pytest.fixture
def mlflow_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Run an MLflow run against a throwaway local file store."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    try:
        with mlflow.start_run() as run:
            yield run
    finally:
        while mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous_uri)


def _records(step: int) -> list[TraceRecord]:
    """Two records for one step, one with structured prompt and reward."""
    return [
        TraceRecord.from_sample(make_sample(0), step=step, dataset="probe"),
        TraceRecord.from_sample(
            make_sample(1, prompt=[{"role": "user", "content": "hi"}], reward={"score": 1.0}),
            step=step,
            dataset="probe",
        ),
    ]


def test_table_accumulates_rows_of_every_step(mlflow_run: Any) -> None:
    """Publishing two steps leaves one long table with the rows of both steps."""
    sink = MlflowTableSink.for_active_run(dataset="probe")

    sink.publish(step=0, records=_records(0))
    sink.publish(step=2, records=_records(2))

    table = mlflow.load_table("traces/probe.json", run_ids=[mlflow_run.info.run_id])
    assert sorted(table["step"].tolist()) == [0, 0, 2, 2]
    assert set(table.columns) >= {
        "step",
        "prompt_id",
        "prompt",
        "response",
        "expected",
        "reward",
        "response_length",
        "truncated",
    }
    assert '{"score": 1.0}' in table["reward"].tolist()


def test_upload_targets_the_run_active_when_built(mlflow_run: Any) -> None:
    """The run is captured on the building thread, so a later upload still lands in it."""
    sink = MlflowTableSink.for_active_run(dataset="probe")
    mlflow.end_run()

    sink.publish(step=1, records=_records(1))

    table = mlflow.load_table("traces/probe.json", run_ids=[mlflow_run.info.run_id])
    assert table["step"].tolist() == [1, 1]
