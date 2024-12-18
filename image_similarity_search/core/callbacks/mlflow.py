from image_similarity_search.utils import LOGGER, colorstr

try:
    import dagshub
    import mlflow

except ImportError:
    LOGGER.warning("MLflow not found. Install it with `pip install mlflow`.")
    mlflow = None


try:
    import os

    import mlflow

    assert hasattr(mlflow, "__version__")  # verify package is not directory

    PREFIX = colorstr("MLflow: ")
except (ImportError, AssertionError):
    mlflow = None


def sanitize_dict(x):
    """Sanitize dictionary keys by removing parentheses and converting values to floats."""
    return {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}


def on_pretrain_routine_end(state, *args, **kwargs):
    """
    Log training parameters to MLflow at the end of the pretraining routine.

    This function sets up MLflow logging based on environment variables and state arguments. It sets the tracking URI,
    experiment name, and run name, then starts the MLflow run if not already active. It finally logs the parameters
    from the state.

    Args:
        state (image_similarity_search.core.state.Basestate): The training object with arguments and parameters to log.

    Global:
        mlflow: The imported mlflow module to use for logging.

    Environment Variables:
        DAGSHUB_REPO_OWNER (str): The owner of the repository to log to.
        DAGSHUB_REPO_NAME (str): The name of the repository to log to.
    """
    global mlflow

    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    try:
        active_run = mlflow.active_run() or mlflow.start_run()
        uri = mlflow.get_tracking_uri()
        LOGGER.info("%slogging run_id(%s) to %s", PREFIX, active_run.info.run_id, uri)
        mlflow.log_params(dict(state.hyp))
    except (mlflow.exceptions.MlflowException, ValueError, TypeError) as e:
        LOGGER.warning(
            "%sWARNING ⚠️ Failed to initialize: %s\n%sWARNING ⚠️ Not tracking this run",
            PREFIX,
            e,
            PREFIX,
        )


def on_fit_epoch_end(state, *args, **kwargs):
    """Log training metrics at the end of each fit epoch to MLflow."""
    if mlflow:
        mlflow.log_metrics(metrics=sanitize_dict(state.metrics), step=state.epoch)


def on_model_save(state, *args, **kwargs):
    if mlflow:
        mlflow.pytorch.log_model(state.model, "latest")

        if state.best_fitness is not None and state.best_fitness == state.fitness:
            mlflow.pytorch.log_model(state.model, "best")

        if state.save_period > 0 and state.epoch % state.save_period == 0:
            mlflow.pytorch.log_model(state.model, f"epoch_{state.epoch}")


def on_train_end(state, *args, **kwargs):
    """Log model artifacts at the end of the training."""
    if not mlflow:
        return
    mlflow.end_run()
    LOGGER.debug(f"{PREFIX}mlflow run ended")
    LOGGER.info(f"{PREFIX}results logged to {mlflow.get_tracking_uri()}\n")


callbacks = (
    {
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
        "on_model_save": on_model_save,
    }
    if mlflow
    else {}
)
