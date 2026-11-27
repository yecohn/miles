import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.tracking_utils.checkpoint_traces.best_effort import BackgroundUploads, BestEffortSink
from miles.utils.tracking_utils.checkpoint_traces.config import CheckpointTracesConfig
from miles.utils.tracking_utils.checkpoint_traces.jsonl_sink import JsonlTraceSink
from miles.utils.tracking_utils.checkpoint_traces.mlflow_sink import MlflowTraceSink
from miles.utils.tracking_utils.checkpoint_traces.record import TraceRecord
from miles.utils.tracking_utils.checkpoint_traces.sink import TraceSink
from miles.utils.tracking_utils.checkpoint_traces.step import BASELINE_FILE_NAME, TraceStep, trace_step

logger = logging.getLogger(__name__)

RemoteSinkFactory = Callable[[ImmutableNamespace], TraceSink | None]


class CheckpointTracesHook:
    """Eval-rollout log hook writing readable traces of one eval dataset locally, then to a remote sink."""

    config_class = CheckpointTracesConfig

    def __init__(self, *, remote_sink: RemoteSinkFactory) -> None:
        self._remote_sink = remote_sink

    def __call__(self, rollout_id: int, args: ImmutableNamespace, data: dict[str, Any], extra_metrics: Any) -> bool:
        """Publish this eval's traces, never raising, and return False so the default eval metrics still log."""
        try:
            self._publish(rollout_id=rollout_id, args=args, data=data)
        except Exception:
            logger.warning(f"checkpoint traces: failed to publish traces of eval {rollout_id}", exc_info=True)
        return False

    def _publish(self, *, rollout_id: int, args: ImmutableNamespace, data: dict[str, Any]) -> None:
        """Build the traced dataset's records and publish them to every configured sink."""
        if args.checkpoint_traces_dir is None:
            logger.warning("checkpoint traces: --checkpoint-traces-dir is not set, no traces are written")
            return
        directory = Path(args.checkpoint_traces_dir)
        step = _trace_step(rollout_id=rollout_id, args=args, directory=directory)
        if (records := _select_records(step=step, args=args, data=data)) is None:
            return
        JsonlTraceSink(directory=directory).publish(step=step, records=records)
        if (remote := self._remote_sink(args)) is not None:
            remote.publish(step=step, records=records)


def _trace_step(*, rollout_id: int, args: ImmutableNamespace, directory: Path) -> TraceStep:
    """Number this eval, treating it as the baseline only while the folder has none."""
    return trace_step(
        rollout_id=rollout_id,
        start_rollout_id=args.start_rollout_id,
        skip_eval_before_train=args.skip_eval_before_train,
        baseline_exists=(directory / BASELINE_FILE_NAME).exists(),
    )


def _select_records(*, step: TraceStep, args: ImmutableNamespace, data: dict[str, Any]) -> list[TraceRecord] | None:
    """Return the traced dataset's records, or None with a warning when this eval did not run it."""
    dataset = args.checkpoint_traces_dataset
    if dataset not in data:
        logger.warning(f"checkpoint traces: eval dataset {dataset!r} not in this eval ({sorted(data)}), skipping")
        return None
    return [TraceRecord.from_sample(sample, step=step.step, dataset=dataset) for sample in data[dataset]["samples"]]


# Hooks are loaded by dotted path, so no constructor can inject this; it remembers the in-flight MLflow upload.
_MLFLOW_UPLOADS = BackgroundUploads()


def _best_effort_mlflow_sink(args: ImmutableNamespace) -> TraceSink | None:
    """Build the best-effort MLflow trace sink when the run logs to MLflow."""
    if not args.use_mlflow:
        return None
    return BestEffortSink(
        inner=MlflowTraceSink.for_active_run(),
        timeout=args.checkpoint_traces_mlflow_timeout,
        uploads=_MLFLOW_UPLOADS,
    )


log_eval_traces = CheckpointTracesHook(remote_sink=_best_effort_mlflow_sink)
