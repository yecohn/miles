"""Readable per-eval traces of one eval dataset, written as JSONL and logged as an MLflow table.

Use as ``--custom-eval-rollout-log-function-path miles.utils.tracking_utils.checkpoint_traces.log_eval_traces``.
"""

from miles.utils.tracking_utils.checkpoint_traces.hook import log_eval_traces

__all__ = ["log_eval_traces"]
