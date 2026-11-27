import json
import os
from collections.abc import Sequence
from pathlib import Path

from miles.utils.tracking_utils.checkpoint_traces.record import TraceRecord

BASELINE_FILE_NAME = "baseline.jsonl"


def trace_file_name(
    *, step: int, start_rollout_id: int | None, skip_eval_before_train: bool, baseline_exists: bool
) -> str:
    """Name the file of one eval: the first pre-training eval of a fresh run is the baseline, the rest are steps.

    ``start_rollout_id`` is None in the rollout manager unless the flag was passed: the trainer resolves it later,
    and only the driver sees the result. An unset value is therefore read as a fresh run."""
    is_pre_training_eval = step == 0 and not start_rollout_id and not skip_eval_before_train
    if is_pre_training_eval and not baseline_exists:
        return BASELINE_FILE_NAME
    return f"step_{step:06d}.jsonl"


class JsonlTraceSink:
    """TraceSink strategy writing each eval step as one atomically replaced JSONL file in a folder."""

    def __init__(self, *, directory: Path, start_rollout_id: int | None, skip_eval_before_train: bool) -> None:
        self._directory = directory
        self._start_rollout_id = start_rollout_id
        self._skip_eval_before_train = skip_eval_before_train

    def publish(self, *, step: int, records: Sequence[TraceRecord]) -> None:
        """Write ``records`` as one JSON object per line to this step's file."""
        self._directory.mkdir(parents=True, exist_ok=True)
        name = trace_file_name(
            step=step,
            start_rollout_id=self._start_rollout_id,
            skip_eval_before_train=self._skip_eval_before_train,
            baseline_exists=(self._directory / BASELINE_FILE_NAME).exists(),
        )
        _write_atomically(path=self._directory / name, text="".join(_to_json_line(record) for record in records))


def _to_json_line(record: TraceRecord) -> str:
    """Serialise one record as a UTF-8 JSON line."""
    return json.dumps(record.to_dict(), ensure_ascii=False) + "\n"


def _write_atomically(*, path: Path, text: str) -> None:
    """Write ``text`` to a temp file next to ``path`` and rename it over ``path``."""
    # not mkstemp: its 0600 mode would hide the traces from other users of a shared run folder
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
