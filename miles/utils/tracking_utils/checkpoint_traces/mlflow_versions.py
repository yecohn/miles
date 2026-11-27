from collections.abc import Sequence

from miles.utils.tracking_utils.checkpoint_traces.mlflow_import import import_mlflow
from miles.utils.tracking_utils.checkpoint_traces.record import TraceRecord
from miles.utils.tracking_utils.checkpoint_traces.step import TraceStep


class MlflowStepVersions:
    """One weightless MLflow logged model per eval step of a run, holding that step's traces and scores."""

    def __init__(self, *, run_id: str, experiment_id: str, dataset: str) -> None:
        self._run_id = run_id
        self._experiment_id = experiment_id
        self._dataset = dataset

    def version_for(self, *, step: TraceStep) -> str:
        """Return the id of the step's logged model, creating it on the step's first eval in this run."""
        mlflow = import_mlflow()
        existing = mlflow.search_logged_models(
            experiment_ids=[self._experiment_id],
            filter_string=f"name = '{step.label}' AND source_run_id = '{self._run_id}'",
            output_format="list",
        )
        if existing:
            return existing[0].model_id
        model = mlflow.create_external_model(
            name=step.label,
            source_run_id=self._run_id,
            experiment_id=self._experiment_id,
            model_type="checkpoint",
            tags={"step": str(step.step), "dataset": self._dataset, "weights": "not saved"},
        )
        return model.model_id

    def log_scores(self, *, model_id: str, step: TraceStep, records: Sequence[TraceRecord]) -> None:
        """Log the step's scores against its logged model; MLflow also adds them to the run's metrics."""
        client = import_mlflow().MlflowClient()
        for key, value in step_scores(dataset=self._dataset, records=records).items():
            client.log_metric(self._run_id, key, value, step=step.step, model_id=model_id, synchronous=True)


def step_scores(*, dataset: str, records: Sequence[TraceRecord]) -> dict[str, float]:
    """Score one eval: the truncated ratio, and the mean reward when rewards are numbers (missing counts as 0).

    The keys avoid miles' own ``eval/...`` metrics, since MLflow also logs model metrics on the run."""
    if not records:
        return {}
    scores = {f"{dataset}/truncated_ratio": sum(record.truncated for record in records) / len(records)}
    numeric = [record.reward for record in records if not isinstance(record.reward, dict)]
    if len(numeric) == len(records):
        scores[f"{dataset}/reward"] = sum(0.0 if reward is None else reward for reward in numeric) / len(records)
    return scores
