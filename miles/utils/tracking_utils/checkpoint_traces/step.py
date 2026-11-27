from dataclasses import dataclass

BASELINE_STEP = -1


@dataclass(frozen=True)
class TraceStep:
    """One eval's place in the run: the step its records carry and the label naming its file and version."""

    step: int
    label: str

    @property
    def file_name(self) -> str:
        """Return the JSONL file name of this eval."""
        return f"{self.label}.jsonl"


BASELINE = TraceStep(step=BASELINE_STEP, label="baseline")


def trace_step(
    *, rollout_id: int, start_rollout_id: int | None, skip_eval_before_train: bool, baseline_exists: bool
) -> TraceStep:
    """Number one eval: the first pre-training eval of a fresh run is the baseline, every other eval its rollout.

    Miles numbers both the eval before training and the eval after the first step as rollout 0, so the baseline
    gets its own step. ``start_rollout_id`` is None in the rollout manager unless the flag was passed: the trainer
    resolves it later, and only the driver sees the result. An unset value is therefore read as a fresh run."""
    is_pre_training_eval = rollout_id == 0 and not start_rollout_id and not skip_eval_before_train
    if is_pre_training_eval and not baseline_exists:
        return BASELINE
    return TraceStep(step=rollout_id, label=f"step_{rollout_id:06d}")
