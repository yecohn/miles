from types import ModuleType


def import_mlflow() -> ModuleType:
    """Import MLflow on first use."""
    # mlflow is an optional dependency, needed only when the run logs to MLflow
    import mlflow

    return mlflow
