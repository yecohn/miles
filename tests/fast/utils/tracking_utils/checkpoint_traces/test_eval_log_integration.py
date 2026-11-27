from pathlib import Path
from types import SimpleNamespace

from tests.fast.utils.tracking_utils.checkpoint_traces.fixtures import make_eval_data, make_sample, read_jsonl

from miles.ray.rollout.metrics import log_eval_rollout_data
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.function_registry import function_registry
from miles.utils.tracking_utils.checkpoint_traces import log_eval_traces
from miles.utils.tracking_utils.checkpoint_traces.config import CheckpointTracesConfig

_HOOK_PATH = "test.checkpoint_traces.log_eval_traces"


def test_traces_are_written_and_default_eval_metrics_still_logged(tmp_path: Path) -> None:
    """Plugged in as the eval log hook, traces are written and the default eval metrics are kept."""
    hook = CustomFunctionConfig(path=_HOOK_PATH, config=CheckpointTracesConfig(checkpoint_traces_dir=str(tmp_path)))
    args = ImmutableNamespace(
        custom_eval_rollout_log_function_path=hook,
        use_mlflow=False,
        start_rollout_id=0,
        skip_eval_before_train=True,
        log_passrate=False,
        wandb_always_use_train_step=False,
        ci_test=False,
        ci_assert_prefill_lag_max=None,
        use_session_server=False,
        n_samples_per_eval_prompt=1,
        advantage_estimator="grpo",
        reward_key=None,
        eval_reward_key=None,
        log_reward_category=None,
        n_samples_per_prompt=1,
        sglang=SimpleNamespace(common_value=lambda name: None),
    )
    data = make_eval_data({"probe": [make_sample(0), make_sample(1)]})

    with function_registry.temporary(_HOOK_PATH, log_eval_traces):
        metrics = log_eval_rollout_data(6, args, data)

    assert metrics["eval/probe"] == 0.5
    assert [row["prompt_id"] for row in read_jsonl(tmp_path / "step_000006.jsonl")] == [0, 1]
