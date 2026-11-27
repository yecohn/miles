from miles.utils.args.schema import A, Arg, BaseConfig


class CheckpointTracesConfig(BaseConfig):
    """CLI flags of the checkpoint-traces eval log hook."""

    checkpoint_traces_dir: A[
        str | None,
        Arg(help="Folder for per-eval trace JSONL files; the hook does nothing when unset."),
    ] = None
    checkpoint_traces_dataset: A[
        str,
        Arg(help="Name of the eval dataset whose samples are written as traces; other datasets are ignored."),
    ] = "probe"
    checkpoint_traces_mlflow_timeout: A[
        float,
        Arg(help="Seconds an eval waits for the MLflow trace upload before moving on."),
    ] = 10.0
    checkpoint_traces_mlflow_versions: A[
        bool,
        Arg(help="Group each eval's MLflow traces under one MLflow logged model per step, with its probe scores."),
    ] = False
