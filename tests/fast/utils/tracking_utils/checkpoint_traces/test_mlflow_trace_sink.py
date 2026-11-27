import threading
from typing import Any

import pytest
from tests.fast.utils.tracking_utils.checkpoint_traces.fixtures import make_sample

from miles.utils.tracking_utils.checkpoint_traces.mlflow_sink import MlflowTraceSink
from miles.utils.tracking_utils.checkpoint_traces.record import TraceRecord
from miles.utils.tracking_utils.checkpoint_traces.step import BASELINE_STEP, TraceStep
from miles.utils.types import Sample

mlflow = pytest.importorskip("mlflow")


def _publish(sink: MlflowTraceSink, *, step: int, samples: list[Sample]) -> None:
    """Publish ``samples`` of one probe eval at ``step``."""
    records = [TraceRecord.from_sample(sample, step=step, dataset="probe") for sample in samples]
    sink.publish(step=TraceStep(step=step, label=f"step_{step:06d}"), records=records)


def _search(run: Any, filter_string: str | None = None) -> list[Any]:
    """Return the run's traces matching ``filter_string``."""
    return mlflow.search_traces(
        locations=[run.info.experiment_id], run_id=run.info.run_id, filter_string=filter_string, return_type="list"
    )


def test_one_tagged_trace_per_record(mlflow_run: Any) -> None:
    """Each record becomes one single-span trace of the run, filterable by step and prompt."""
    sink = MlflowTraceSink.for_active_run(dataset="probe", versions=False)

    _publish(sink, step=0, samples=[make_sample(0), make_sample(1)])
    _publish(sink, step=2, samples=[make_sample(0), make_sample(1)])

    assert len(_search(mlflow_run)) == 4
    [trace] = _search(mlflow_run, "tags.step = '2' AND tags.prompt_id = '1'")
    assert {
        key: trace.info.tags[key] for key in ("step", "prompt_id", "dataset", "reward", "truncated", "status")
    } == {
        "step": "2",
        "prompt_id": "1",
        "dataset": "probe",
        "reward": "1.0",
        "truncated": "false",
        "status": "completed",
    }
    [span] = trace.data.spans
    assert span.name == "probe/1"
    assert span.inputs == {"prompt": "question 1"}
    assert span.outputs == "answer 1"
    assert span.attributes["expected"] == "1"


def test_structured_and_missing_values_of_a_baseline_eval(mlflow_run: Any) -> None:
    """Dict and missing rewards become text tags, chat prompts stay structured, and the baseline step is filterable."""
    messages = [{"role": "user", "content": "hi"}]
    samples = [
        make_sample(0, reward={"score": 1.0}, prompt=messages),
        make_sample(1, reward=None, label=None, status=Sample.Status.TRUNCATED),
    ]

    _publish(MlflowTraceSink.for_active_run(dataset="probe", versions=False), step=BASELINE_STEP, samples=samples)

    by_prompt = {trace.info.tags["prompt_id"]: trace for trace in _search(mlflow_run, "tags.step = '-1'")}
    assert sorted(by_prompt) == ["0", "1"]
    assert by_prompt["0"].info.tags["reward"] == '{"score": 1.0}'
    assert by_prompt["0"].data.spans[0].inputs == {"prompt": messages}
    assert by_prompt["1"].info.tags["reward"] == "none"
    assert by_prompt["1"].info.tags["truncated"] == "true"
    assert by_prompt["1"].info.tags["status"] == "truncated"


def test_publish_from_another_thread_links_to_the_run_and_keeps_it_running(mlflow_run: Any) -> None:
    """The hook uploads on a worker thread with no active run; traces still reach the run, which stays open."""
    sink = MlflowTraceSink.for_active_run(dataset="probe", versions=False)
    worker = threading.Thread(target=lambda: _publish(sink, step=3, samples=[make_sample(0)]))

    worker.start()
    worker.join(timeout=10.0)

    assert not worker.is_alive()
    assert [trace.info.tags["step"] for trace in _search(mlflow_run)] == ["3"]
    assert mlflow.get_run(mlflow_run.info.run_id).info.status == "RUNNING"


def test_traces_link_by_run_id_alone(mlflow_run: Any) -> None:
    """A run active on no thread is still the traces' run, because the sink names it explicitly.

    MLflow otherwise falls back to the newest run active on any thread, which would hide a missing run id."""
    run = mlflow.MlflowClient().create_run(experiment_id=mlflow_run.info.experiment_id)

    _publish(MlflowTraceSink(run_id=run.info.run_id), step=4, samples=[make_sample(0)])

    assert [trace.info.tags["step"] for trace in _search(run)] == ["4"]
    assert _search(mlflow_run) == []
    assert mlflow.get_run(run.info.run_id).info.status == "RUNNING"
