import os
import time
from pathlib import Path

import pytest
from tests.fast.utils.tracking_utils.checkpoint_traces.fixtures import (
    TRACE_FIELDS,
    BlockingSink,
    RaisingSink,
    RecordingSink,
    make_args,
    make_eval_data,
    make_sample,
    read_jsonl,
)

from miles.utils.tracking_utils.checkpoint_traces import log_eval_traces
from miles.utils.tracking_utils.checkpoint_traces.best_effort import BackgroundUploads, BestEffortSink
from miles.utils.tracking_utils.checkpoint_traces.config import CheckpointTracesConfig
from miles.utils.tracking_utils.checkpoint_traces.hook import CheckpointTracesHook
from miles.utils.tracking_utils.checkpoint_traces.sink import TraceSink
from miles.utils.tracking_utils.checkpoint_traces.step import TraceStep, trace_step
from miles.utils.types import Sample


def _hook_with_remote(sink: TraceSink) -> CheckpointTracesHook:
    """Build a hook whose remote upload goes best-effort through ``sink``."""
    uploads = BackgroundUploads()
    return CheckpointTracesHook(
        remote_sink=lambda args: BestEffortSink(
            inner=sink, timeout=args.checkpoint_traces_mlflow_timeout, uploads=uploads
        )
    )


class TestTraceRows:
    def test_writes_one_row_per_sample_of_the_chosen_dataset_in_index_order(self, tmp_path: Path) -> None:
        """Only the traced dataset is written, one readable row per sample, without token-level fields."""
        data = make_eval_data({"probe": [make_sample(0), make_sample(1), make_sample(2)], "other": [make_sample(7)]})

        returned = log_eval_traces(4, make_args(tmp_path), data, None)

        assert returned is False
        assert sorted(p.name for p in tmp_path.iterdir()) == ["step_000004.jsonl"]
        rows = read_jsonl(tmp_path / "step_000004.jsonl")
        assert [row["prompt_id"] for row in rows] == [0, 1, 2]
        assert all(set(row) == TRACE_FIELDS for row in rows)
        assert rows[1] == {
            "step": 4,
            "dataset": "probe",
            "prompt_id": 1,
            "prompt": "question 1",
            "response": "answer 1",
            "expected": "1",
            "reward": 1.0,
            "response_length": 3,
            "truncated": False,
            "status": "completed",
        }

    def test_dict_reward_none_reward_message_prompt_and_unicode_round_trip(self, tmp_path: Path) -> None:
        """Structured rewards, missing rewards, chat prompts and non-ASCII text survive serialisation."""
        messages = [{"role": "user", "content": "Combien font 2+2 ? é"}]
        samples = [
            make_sample(0, reward={"score": 1.0, "pred": "4"}, prompt=messages),
            make_sample(1, reward=None, status=Sample.Status.TRUNCATED),
        ]

        log_eval_traces(2, make_args(tmp_path), make_eval_data({"probe": samples}), None)

        path = tmp_path / "step_000002.jsonl"
        assert "é" in path.read_text(encoding="utf-8")
        first, second = read_jsonl(path)
        assert first["reward"] == {"score": 1.0, "pred": "4"}
        assert first["prompt"] == messages
        assert second["reward"] is None
        assert second["truncated"] is True
        assert second["status"] == "truncated"

    def test_hook_declares_its_flags(self) -> None:
        """The hook publishes its CLI flags through ``config_class``."""
        assert log_eval_traces.config_class is CheckpointTracesConfig
        assert CheckpointTracesConfig().checkpoint_traces_dir is None


class TestBaseline:
    def test_first_step_zero_of_a_fresh_run_is_the_baseline(self, tmp_path: Path) -> None:
        """The pre-training eval lands in baseline.jsonl as step -1; the eval after step 0 keeps step 0."""
        sink = RecordingSink()
        hook = CheckpointTracesHook(remote_sink=lambda args: sink)

        hook(0, make_args(tmp_path), make_eval_data({"probe": [make_sample(0)]}), None)
        baseline = (tmp_path / "baseline.jsonl").read_text(encoding="utf-8")
        hook(0, make_args(tmp_path), make_eval_data({"probe": [make_sample(0, response="x")]}), None)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["baseline.jsonl", "step_000000.jsonl"]
        assert (tmp_path / "baseline.jsonl").read_text(encoding="utf-8") == baseline
        assert [row["step"] for row in read_jsonl(tmp_path / "baseline.jsonl")] == [-1]
        assert read_jsonl(tmp_path / "step_000000.jsonl")[0]["step"] == 0
        assert read_jsonl(tmp_path / "step_000000.jsonl")[0]["response"] == "x"
        assert [[record.step for record in records] for _, records in sink.published] == [[-1], [0]]
        assert [step.step for step, _ in sink.published] == [-1, 0]

    def test_unresolved_start_step_counts_as_a_fresh_run(self, tmp_path: Path) -> None:
        """The rollout manager sees no start step unless the flag was passed; the first eval is still the baseline."""
        args = make_args(tmp_path, start_rollout_id=None)

        log_eval_traces(0, args, make_eval_data({"probe": [make_sample(0)]}), None)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["baseline.jsonl"]
        assert read_jsonl(tmp_path / "baseline.jsonl")[0]["step"] == -1

    def test_trace_files_are_readable_by_other_users(self, tmp_path: Path) -> None:
        """Trace files take the process umask, so others on a shared run folder can read them."""
        previous_umask = os.umask(0o022)
        try:
            log_eval_traces(1, make_args(tmp_path), make_eval_data({"probe": [make_sample(0)]}), None)
        finally:
            os.umask(previous_umask)

        assert (tmp_path / "step_000001.jsonl").stat().st_mode & 0o777 == 0o644

    def test_file_name_derives_from_the_step_label(self) -> None:
        """The label is the one name of a step; its JSONL file is the label plus the extension."""
        baseline = trace_step(rollout_id=0, start_rollout_id=None, skip_eval_before_train=False, baseline_exists=False)
        later = trace_step(rollout_id=3, start_rollout_id=None, skip_eval_before_train=False, baseline_exists=True)

        assert (baseline.step, baseline.label, baseline.file_name) == (-1, "baseline", "baseline.jsonl")
        assert (later.step, later.label, later.file_name) == (3, "step_000003", "step_000003.jsonl")

    @pytest.mark.parametrize("overrides", [{"skip_eval_before_train": True}, {"start_rollout_id": 3}])
    def test_no_baseline_without_an_eval_before_training(self, tmp_path: Path, overrides: dict) -> None:
        """Runs that skip the pre-training eval, or resume, never produce a baseline file."""
        log_eval_traces(0, make_args(tmp_path, **overrides), make_eval_data({"probe": [make_sample(0)]}), None)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["step_000000.jsonl"]
        assert read_jsonl(tmp_path / "step_000000.jsonl")[0]["step"] == 0


class TestBestEffort:
    @pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
    def test_failing_remote_upload_keeps_the_local_file(self, tmp_path: Path) -> None:
        """A raising remote sink neither propagates nor prevents the JSONL file."""
        hook = _hook_with_remote(RaisingSink())

        assert hook(1, make_args(tmp_path), make_eval_data({"probe": [make_sample(0)]}), None) is False
        assert (tmp_path / "step_000001.jsonl").exists()

    def test_hung_remote_upload_returns_after_the_timeout_and_blocks_the_next_one(self, tmp_path: Path) -> None:
        """A hung upload costs at most the timeout, and no second upload starts while it hangs."""
        sink = BlockingSink()
        hook = _hook_with_remote(sink)
        data = make_eval_data({"probe": [make_sample(0)]})
        try:
            started = time.monotonic()
            assert hook(1, make_args(tmp_path), data, None) is False
            assert time.monotonic() - started < 2.0
            assert hook(2, make_args(tmp_path), data, None) is False
            assert sink.started == 1
        finally:
            sink.release.set()
        assert (tmp_path / "step_000001.jsonl").exists()
        assert (tmp_path / "step_000002.jsonl").exists()

    def test_remote_sink_receives_the_same_records(self, tmp_path: Path) -> None:
        """The remote sink gets the step and the records written locally."""
        sink = RecordingSink()
        hook = CheckpointTracesHook(remote_sink=lambda args: sink)

        hook(5, make_args(tmp_path), make_eval_data({"probe": [make_sample(0), make_sample(1)]}), None)

        [(step, records)] = sink.published
        assert step == TraceStep(step=5, label="step_000005")
        assert [record.step for record in records] == [5, 5]
        assert [record.prompt_id for record in records] == [0, 1]

    def test_remote_upload_is_skipped_without_mlflow(self, tmp_path: Path) -> None:
        """The default hook never touches MLflow when the run does not use it."""
        log_eval_traces(1, make_args(tmp_path, use_mlflow=False), make_eval_data({"probe": [make_sample(0)]}), None)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["step_000001.jsonl"]


class TestNothingToWrite:
    def test_invalid_directory_does_not_raise(self, tmp_path: Path) -> None:
        """A directory below a regular file cannot be created, and the hook still returns normally."""
        blocker = tmp_path / "file"
        blocker.write_text("x")

        result = log_eval_traces(1, make_args(blocker / "traces"), make_eval_data({"probe": [make_sample(0)]}), None)

        assert result is False

    def test_no_directory_writes_nothing(self, tmp_path: Path) -> None:
        """Without a traces directory the hook is a no-op."""
        sink = RecordingSink()
        hook = CheckpointTracesHook(remote_sink=lambda args: sink)

        assert hook(1, make_args(None), make_eval_data({"probe": [make_sample(0)]}), None) is False
        assert sink.published == []

    def test_missing_dataset_writes_nothing(self, tmp_path: Path) -> None:
        """An eval without the traced dataset leaves the directory untouched."""
        log_eval_traces(1, make_args(tmp_path), make_eval_data({"other": [make_sample(0)]}), None)

        assert list(tmp_path.iterdir()) == []

    def test_malformed_data_does_not_raise(self, tmp_path: Path) -> None:
        """Unexpected eval data is logged, never raised into the training loop."""
        assert log_eval_traces(1, make_args(tmp_path), {"probe": {"samples": [object()]}}, None) is False
