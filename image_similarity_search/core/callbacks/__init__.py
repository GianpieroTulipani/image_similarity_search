from .base import Callbacks
from .mlflow import callbacks as mlflow_callbacks

INTEGRATIONS = {"mlflow": mlflow_callbacks}

DEFAULT_CALLBACKS = Callbacks()


def add_integration(callbacks: Callbacks, name: str):
    if name in INTEGRATIONS:
        callbacks.add_callbacks(name, INTEGRATIONS[name])
    else:
        raise ValueError(f"Integration '{name}' not found in {INTEGRATIONS.keys()}")
