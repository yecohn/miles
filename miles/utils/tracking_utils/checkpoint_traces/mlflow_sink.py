import json
from collections.abc import Sequence
from types import ModuleType
from typing import Any

from typing_extensions import Self

from miles.utils.tracking_utils.checkpoint_traces.record import TraceRecord
from miles.utils.tracking_utils.checkpoint_traces.step import TraceStep


class MlflowTraceSink:
    """TraceSink strategy logging each record as one single-span MLflow trace linked to the run."""

    def __init__(self, *, run_id: str) -> None:
        self._run_id = run_id

    @classmethod
    def for_active_run(cls) -> Self:
        """Bind to the calling thread's active MLflow run, since MLflow keeps the active run per thread."""
        if (run := _mlflow().active_run()) is None:
            raise RuntimeError("--use-mlflow is set but there is no active MLflow run")
        return cls(run_id=run.info.run_id)

    def publish(self, *, step: TraceStep, records: Sequence[TraceRecord]) -> None:
        """Log one trace per record and wait until MLflow has exported them."""
        mlflow = _mlflow()
        for record in records:
            with mlflow.start_span(name=f"{record.dataset}/{record.prompt_id}", run_id=self._run_id) as span:
                span.set_inputs({"prompt": record.prompt})
                span.set_outputs(record.response)
                if record.expected is not None:
                    span.set_attributes({"expected": record.expected})
                mlflow.update_current_trace(tags=trace_tags(record))
        mlflow.flush_trace_async_logging()


def trace_tags(record: TraceRecord) -> dict[str, str]:
    """Return the string tags that make a record's trace filterable in the MLflow UI."""
    return {
        "step": str(record.step),
        "prompt_id": str(record.prompt_id),
        "dataset": record.dataset,
        "reward": _reward_tag(record.reward),
        "truncated": str(record.truncated).lower(),
        "status": record.status,
    }


def _reward_tag(reward: float | dict[str, Any] | None) -> str:
    """Render a reward as tag text: a number as is, a dict as JSON, a missing reward as ``none``."""
    if reward is None:
        return "none"
    if isinstance(reward, dict):
        return json.dumps(reward, ensure_ascii=False)
    return str(reward)


def _mlflow() -> ModuleType:
    """Import MLflow on first use."""
    # mlflow is an optional dependency, needed only when the run logs to MLflow
    import mlflow

    return mlflow
