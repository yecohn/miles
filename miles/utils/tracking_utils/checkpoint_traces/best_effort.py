import logging
import threading
from collections.abc import Callable, Sequence
from functools import partial

from miles.utils.tracking_utils.checkpoint_traces.record import TraceRecord
from miles.utils.tracking_utils.checkpoint_traces.sink import TraceSink
from miles.utils.tracking_utils.checkpoint_traces.step import TraceStep

logger = logging.getLogger(__name__)


class BackgroundUploads:
    """Runs one upload at a time on a daemon thread, waiting a bounded time and never queueing behind a hung one."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None

    def run(self, target: Callable[[], None], *, timeout: float) -> None:
        """Run ``target`` in the background and wait for it at most ``timeout`` seconds."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("checkpoint traces: previous upload still running, skipping this one")
            return
        self._thread = threading.Thread(target=target, name="checkpoint-traces-upload", daemon=True)
        self._thread.start()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning(f"checkpoint traces: upload still running after {timeout}s, continuing without it")


class BestEffortSink:
    """Decorator pattern: a TraceSink publishing through an inner sink that can neither stall nor crash the caller."""

    def __init__(self, *, inner: TraceSink, timeout: float, uploads: BackgroundUploads) -> None:
        self._inner = inner
        self._timeout = timeout
        self._uploads = uploads

    def publish(self, *, step: TraceStep, records: Sequence[TraceRecord]) -> None:
        """Publish through the inner sink in the background, waiting at most the timeout."""
        self._uploads.run(partial(self._publish_logging_errors, step=step, records=records), timeout=self._timeout)

    def _publish_logging_errors(self, *, step: TraceStep, records: Sequence[TraceRecord]) -> None:
        """Publish through the inner sink, logging instead of raising on failure."""
        try:
            self._inner.publish(step=step, records=records)
        except Exception:
            logger.warning(f"checkpoint traces: upload of step {step.step} failed", exc_info=True)
