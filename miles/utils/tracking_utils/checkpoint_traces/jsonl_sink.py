import json
import os
from collections.abc import Sequence
from pathlib import Path

from miles.utils.tracking_utils.checkpoint_traces.record import TraceRecord
from miles.utils.tracking_utils.checkpoint_traces.step import TraceStep


class JsonlTraceSink:
    """TraceSink strategy writing each eval step as one atomically replaced JSONL file in a folder."""

    def __init__(self, *, directory: Path) -> None:
        self._directory = directory

    def publish(self, *, step: TraceStep, records: Sequence[TraceRecord]) -> None:
        """Write ``records`` as one JSON object per line to the step's file."""
        self._directory.mkdir(parents=True, exist_ok=True)
        text = "".join(_to_json_line(record) for record in records)
        _write_atomically(path=self._directory / step.file_name, text=text)


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
