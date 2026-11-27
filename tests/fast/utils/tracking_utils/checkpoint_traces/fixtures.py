import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.tracking_utils.checkpoint_traces.record import TraceRecord
from miles.utils.tracking_utils.checkpoint_traces.step import TraceStep
from miles.utils.types import Sample

TRACE_FIELDS = {
    "step",
    "dataset",
    "prompt_id",
    "prompt",
    "response",
    "expected",
    "reward",
    "response_length",
    "truncated",
    "status",
}


def make_sample(index: int, **overrides: Any) -> Sample:
    """Build a completed eval sample carrying token-level arrays that must never reach a trace."""
    values: dict[str, Any] = {
        "index": index,
        "prompt": f"question {index}",
        "response": f"answer {index}",
        "label": str(index),
        "reward": float(index % 2),
        "response_length": 3,
        "status": Sample.Status.COMPLETED,
        "tokens": [1, 2, 3, 4],
        "rollout_log_probs": [-0.1, -0.2, -0.3],
        "loss_mask": [1, 1, 1],
    }
    return Sample(**(values | overrides))


def make_eval_data(samples_by_dataset: dict[str, list[Sample]]) -> dict[str, dict[str, Any]]:
    """Shape eval samples the way the eval rollout hands them to the log hook."""
    return {
        name: {
            "rewards": [sample.reward for sample in samples],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in samples],
            "samples": samples,
        }
        for name, samples in samples_by_dataset.items()
    }


def make_args(traces_dir: Path | str | None, **overrides: Any) -> ImmutableNamespace:
    """Build the merged run-and-hook namespace the hook receives on a fresh run."""
    values: dict[str, Any] = {
        "checkpoint_traces_dir": None if traces_dir is None else str(traces_dir),
        "checkpoint_traces_dataset": "probe",
        "checkpoint_traces_mlflow_timeout": 0.2,
        "use_mlflow": False,
        "start_rollout_id": 0,
        "skip_eval_before_train": False,
    }
    return ImmutableNamespace.model_validate(values | overrides)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per line."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class RecordingSink:
    """Trace sink that remembers every published batch."""

    def __init__(self) -> None:
        self.published: list[tuple[TraceStep, list[TraceRecord]]] = []

    def publish(self, *, step: TraceStep, records: Sequence[TraceRecord]) -> None:
        """Remember the batch."""
        self.published.append((step, list(records)))


class RaisingSink:
    """Trace sink whose upload always fails."""

    def publish(self, *, step: TraceStep, records: Sequence[TraceRecord]) -> None:
        """Fail the upload."""
        raise RuntimeError("upload failed")


class BlockingSink:
    """Trace sink whose upload hangs until released, counting how many uploads started."""

    def __init__(self) -> None:
        self.started = 0
        self.release = threading.Event()

    def publish(self, *, step: TraceStep, records: Sequence[TraceRecord]) -> None:
        """Hang until released."""
        self.started += 1
        self.release.wait(timeout=10.0)
