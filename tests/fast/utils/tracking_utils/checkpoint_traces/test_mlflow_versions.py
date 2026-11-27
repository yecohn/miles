from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from tests.fast.utils.tracking_utils.checkpoint_traces.fixtures import make_args, make_eval_data, make_sample

from miles.utils.tracking_utils.checkpoint_traces import log_eval_traces
from miles.utils.tracking_utils.checkpoint_traces.mlflow_sink import MlflowTraceSink
from miles.utils.tracking_utils.checkpoint_traces.mlflow_versions import MlflowStepVersions
from miles.utils.tracking_utils.checkpoint_traces.record import TraceRecord
from miles.utils.tracking_utils.checkpoint_traces.step import BASELINE, TraceStep
from miles.utils.types import Sample

mlflow = pytest.importorskip("mlflow")

_STEP_0 = TraceStep(step=0, label="step_000000")


def _publish(sink: MlflowTraceSink, *, step: TraceStep, samples: list[Sample]) -> None:
    """Publish ``samples`` of one probe eval at ``step``."""
    records = [TraceRecord.from_sample(sample, step=step.step, dataset="probe") for sample in samples]
    sink.publish(step=step, records=records)


def _versioned_sink() -> MlflowTraceSink:
    """Build the trace sink of the active run with per-step versions on."""
    return MlflowTraceSink.for_active_run(dataset="probe", versions=True)


def _models(run: Any) -> dict[str, Any]:
    """Return the logged models of the run's experiment by name."""
    models = mlflow.search_logged_models(experiment_ids=[run.info.experiment_id], output_format="list")
    return {model.name: model for model in models}


def _traces(run: Any, **filters: str) -> list[Any]:
    """Return the experiment's traces matching ``filters`` (e.g. ``model_id=...``)."""
    return mlflow.search_traces(locations=[run.info.experiment_id], return_type="list", **filters)


def _prompt_ids(traces: Sequence[Any]) -> list[str]:
    """Return the sorted prompt ids of ``traces``."""
    return sorted(trace.info.tags["prompt_id"] for trace in traces)


def _metrics(model: Any) -> dict[str, tuple[int, float]]:
    """Return a logged model's metrics as ``{key: (step, value)}``."""
    return {metric.key: (metric.step, metric.value) for metric in model.metrics}


def test_each_step_gets_one_version_holding_its_traces(mlflow_run: Any) -> None:
    """The baseline and step 0 become two logged models of the run, each owning that eval's traces."""
    sink = _versioned_sink()

    _publish(sink, step=BASELINE, samples=[make_sample(0), make_sample(1)])
    _publish(sink, step=_STEP_0, samples=[make_sample(2)])

    models = _models(mlflow_run)
    assert sorted(models) == ["baseline", "step_000000"]
    for name, step in (("baseline", "-1"), ("step_000000", "0")):
        assert models[name].source_run_id == mlflow_run.info.run_id
        assert {key: models[name].tags[key] for key in ("step", "dataset", "weights")} == {
            "step": step,
            "dataset": "probe",
            "weights": "not saved",
        }
    baseline_traces = _traces(mlflow_run, model_id=models["baseline"].model_id)
    assert _prompt_ids(baseline_traces) == ["0", "1"]
    assert {trace.info.tags["step"] for trace in baseline_traces} == {"-1"}
    assert _prompt_ids(_traces(mlflow_run, model_id=models["step_000000"].model_id)) == ["2"]
    assert len(_traces(mlflow_run, run_id=mlflow_run.info.run_id)) == 3


def test_repeated_step_reuses_its_version(mlflow_run: Any) -> None:
    """Re-running an eval of the same step adds its traces to the existing version."""
    sink = _versioned_sink()

    _publish(sink, step=_STEP_0, samples=[make_sample(0)])
    _publish(sink, step=_STEP_0, samples=[make_sample(1)])

    models = _models(mlflow_run)
    assert list(models) == ["step_000000"]
    assert _prompt_ids(_traces(mlflow_run, model_id=models["step_000000"].model_id)) == ["0", "1"]


def test_versions_carry_the_step_scores(mlflow_run: Any) -> None:
    """Each version holds its mean reward (missing counted as 0) and truncated ratio, at the eval's step."""
    sink = _versioned_sink()
    samples = [make_sample(0, reward=1.0), make_sample(1, reward=None), make_sample(2, reward=0.5)]
    samples.append(make_sample(3, reward=0.5, status=Sample.Status.TRUNCATED))

    _publish(sink, step=BASELINE, samples=samples)
    _publish(sink, step=_STEP_0, samples=[make_sample(0, reward={"score": 1.0})])

    models = _models(mlflow_run)
    assert _metrics(models["baseline"]) == {"probe/reward": (-1, 0.5), "probe/truncated_ratio": (-1, 0.25)}
    assert _metrics(models["step_000000"]) == {"probe/truncated_ratio": (0, 0.0)}
    run_keys = set(mlflow.get_run(mlflow_run.info.run_id).data.metrics)
    assert run_keys == {"probe/reward", "probe/truncated_ratio"}


def test_versions_off_leaves_traces_unlinked(mlflow_run: Any, tmp_path: Path) -> None:
    """With the flag off, the hook's MLflow upload creates no logged model and links no trace to one."""
    args = make_args(tmp_path / "traces", use_mlflow=True, checkpoint_traces_mlflow_timeout=10.0)

    log_eval_traces(3, args, make_eval_data({"probe": [make_sample(0)]}), None)

    assert _models(mlflow_run) == {}
    [trace] = _traces(mlflow_run, run_id=mlflow_run.info.run_id)
    assert "mlflow.modelId" not in trace.info.trace_metadata


class _UnreachableVersions(MlflowStepVersions):
    """Step versions whose logged model cannot be found or created."""

    def version_for(self, *, step: TraceStep) -> str:
        """Fail to find or create the version."""
        raise RuntimeError("logged models unavailable")


class _ScorelessVersions(MlflowStepVersions):
    """Step versions whose scores cannot be logged."""

    def log_scores(self, *, model_id: str, step: TraceStep, records: Sequence[TraceRecord]) -> None:
        """Fail to log the scores."""
        raise RuntimeError("metrics unavailable")


@pytest.mark.parametrize(
    ("versions_class", "linked"), [(_UnreachableVersions, False), (_ScorelessVersions, True)], ids=["model", "scores"]
)
def test_failing_versions_still_upload_the_traces(
    mlflow_run: Any, versions_class: type[MlflowStepVersions], linked: bool
) -> None:
    """A failing version step costs only the grouping or the scores: the traces still reach the run."""
    run_id = mlflow_run.info.run_id
    versions = versions_class(run_id=run_id, experiment_id=mlflow_run.info.experiment_id, dataset="probe")
    sink = MlflowTraceSink(run_id=run_id, versions=versions)

    _publish(sink, step=_STEP_0, samples=[make_sample(0), make_sample(1)])

    traces = _traces(mlflow_run, run_id=run_id)
    assert _prompt_ids(traces) == ["0", "1"]
    assert all(("mlflow.modelId" in trace.info.trace_metadata) is linked for trace in traces)


def test_hook_names_each_version_like_its_trace_file(mlflow_run: Any, tmp_path: Path) -> None:
    """End to end, every JSONL file has a version of the same name."""
    directory = tmp_path / "traces"
    args = make_args(
        directory, use_mlflow=True, checkpoint_traces_mlflow_timeout=10.0, checkpoint_traces_mlflow_versions=True
    )

    for rollout_id in (0, 3):
        log_eval_traces(rollout_id, args, make_eval_data({"probe": [make_sample(0)]}), None)

    assert sorted(path.stem for path in directory.iterdir()) == ["baseline", "step_000003"]
    assert sorted(_models(mlflow_run)) == ["baseline", "step_000003"]
