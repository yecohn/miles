import json
import logging
from collections.abc import Sequence
from typing import Any

from typing_extensions import Self

from miles.utils.tracking_utils.checkpoint_traces.mlflow_import import import_mlflow
from miles.utils.tracking_utils.checkpoint_traces.mlflow_versions import MlflowStepVersions
from miles.utils.tracking_utils.checkpoint_traces.record import TraceRecord
from miles.utils.tracking_utils.checkpoint_traces.step import TraceStep

logger = logging.getLogger(__name__)


class MlflowTraceSink:
    """TraceSink strategy logging each record as one single-span MLflow trace linked to the run.

    With ``versions``, each step's traces are also grouped under that step's logged model, which gets its scores.
    """

    def __init__(self, *, run_id: str, versions: MlflowStepVersions | None = None) -> None:
        self._run_id = run_id
        self._versions = versions

    @classmethod
    def for_active_run(cls, *, dataset: str, versions: bool) -> Self:
        """Bind to the calling thread's active MLflow run, since MLflow keeps the active run per thread."""
        if (run := import_mlflow().active_run()) is None:
            raise RuntimeError("--use-mlflow is set but there is no active MLflow run")
        step_versions = (
            MlflowStepVersions(run_id=run.info.run_id, experiment_id=run.info.experiment_id, dataset=dataset)
            if versions
            else None
        )
        return cls(run_id=run.info.run_id, versions=step_versions)

    def publish(self, *, step: TraceStep, records: Sequence[TraceRecord]) -> None:
        """Log one trace per record, grouped under the step's version when enabled, then the version's scores."""
        model_id = self._version_id(step)
        self._log_traces(model_id=model_id, records=records)
        if self._versions is not None and model_id is not None:
            _log_scores(versions=self._versions, model_id=model_id, step=step, records=records)

    def _version_id(self, step: TraceStep) -> str | None:
        """Return the step's logged model id, or None when versions are off or cannot be reached."""
        if self._versions is None:
            return None
        try:
            return self._versions.version_for(step=step)
        except Exception:
            logger.warning(
                f"checkpoint traces: no MLflow version for {step.label}, traces stay ungrouped", exc_info=True
            )
            return None

    def _log_traces(self, *, model_id: str | None, records: Sequence[TraceRecord]) -> None:
        """Log one trace per record and wait until MLflow has exported them."""
        mlflow = import_mlflow()
        for record in records:
            with mlflow.start_span(name=f"{record.dataset}/{record.prompt_id}", run_id=self._run_id) as span:
                span.set_inputs({"prompt": record.prompt})
                span.set_outputs(record.response)
                if record.expected is not None:
                    span.set_attributes({"expected": record.expected})
                mlflow.update_current_trace(tags=trace_tags(record), model_id=model_id)
        mlflow.flush_trace_async_logging()


def _log_scores(
    *, versions: MlflowStepVersions, model_id: str, step: TraceStep, records: Sequence[TraceRecord]
) -> None:
    """Log the step's scores on its version, warning instead of raising on failure."""
    try:
        versions.log_scores(model_id=model_id, step=step, records=records)
    except Exception:
        logger.warning(f"checkpoint traces: could not log the scores of {step.label}", exc_info=True)


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
