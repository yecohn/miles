import json
from collections.abc import Sequence
from types import ModuleType
from typing import Any

from typing_extensions import Self

from miles.utils.tracking_utils.checkpoint_traces.record import TraceRecord

_COLUMNS = ("step", "prompt_id", "prompt", "response", "expected", "reward", "response_length", "truncated")


class MlflowTableSink:
    """TraceSink strategy appending each eval step's records to one long MLflow table of the run."""

    def __init__(self, *, run_id: str, dataset: str) -> None:
        self._run_id = run_id
        self._artifact_file = f"traces/{dataset}.json"

    @classmethod
    def for_active_run(cls, *, dataset: str) -> Self:
        """Bind to the calling thread's active MLflow run, since MLflow keeps the active run per thread."""
        if (run := _mlflow().active_run()) is None:
            raise RuntimeError("--use-mlflow is set but there is no active MLflow run")
        return cls(run_id=run.info.run_id, dataset=dataset)

    def publish(self, *, step: int, records: Sequence[TraceRecord]) -> None:
        """Append ``records`` to the run's trace table."""
        _mlflow().log_table(data=_to_columns(records), artifact_file=self._artifact_file, run_id=self._run_id)


def _mlflow() -> ModuleType:
    """Import MLflow on first use."""
    # mlflow is an optional dependency, needed only when the run logs to MLflow
    import mlflow

    return mlflow


def _to_columns(records: Sequence[TraceRecord]) -> dict[str, list[Any]]:
    """Lay ``records`` out as table columns, with structured cells as JSON text so every column is uniform."""
    rows = [record.to_dict() for record in records]
    return {column: [_to_cell(row[column]) for row in rows] for column in _COLUMNS}


def _to_cell(value: Any) -> Any:
    """Keep scalars and turn lists and dicts into JSON text."""
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=False)
    return value
