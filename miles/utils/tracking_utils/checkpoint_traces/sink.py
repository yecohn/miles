from collections.abc import Sequence
from typing import Protocol

from miles.utils.tracking_utils.checkpoint_traces.record import TraceRecord
from miles.utils.tracking_utils.checkpoint_traces.step import TraceStep


class TraceSink(Protocol):
    """Strategy pattern: a destination that publishes the trace records of one eval step."""

    def publish(self, *, step: TraceStep, records: Sequence[TraceRecord]) -> None:
        """Publish ``records`` of eval ``step``."""
        ...
