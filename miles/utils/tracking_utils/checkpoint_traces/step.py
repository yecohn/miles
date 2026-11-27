from dataclasses import dataclass

BASELINE_STEP = -1
BASELINE_FILE_NAME = "baseline.jsonl"


@dataclass(frozen=True)
class TraceStep:
    """Where one eval's traces go: the step its records carry and the JSONL file they are written to."""

    step: int
    file_name: str


def trace_step(
    *, rollout_id: int, start_rollout_id: int | None, skip_eval_before_train: bool, baseline_exists: bool
) -> TraceStep:
    """Number one eval: the first pre-training eval of a fresh run is the baseline, every other eval its rollout.

    Miles numbers both the eval before training and the eval after the first step as rollout 0, so the baseline
    gets its own step. ``start_rollout_id`` is None in the rollout manager unless the flag was passed: the trainer
    resolves it later, and only the driver sees the result. An unset value is therefore read as a fresh run."""
    is_pre_training_eval = rollout_id == 0 and not start_rollout_id and not skip_eval_before_train
    if is_pre_training_eval and not baseline_exists:
        return TraceStep(step=BASELINE_STEP, file_name=BASELINE_FILE_NAME)
    return TraceStep(step=rollout_id, file_name=f"step_{rollout_id:06d}.jsonl")
