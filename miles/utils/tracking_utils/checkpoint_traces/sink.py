from collections.abc import Sequence
from typing import Protocol

from miles.utils.tracking_utils.checkpoint_traces.record import TraceRecord


class TraceSink(Protocol):
    """Strategy pattern: a destination that publishes the trace records of one eval step."""

    def publish(self, *, step: int, records: Sequence[TraceRecord]) -> None:
        """Publish ``records`` of eval ``step``."""
        ...
