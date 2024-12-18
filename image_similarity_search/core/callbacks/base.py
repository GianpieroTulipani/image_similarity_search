from typing import Callable


class Callbacks:
    """ "
    Handles all registered callbacks for YOLOv5 Hooks
    """

    def __init__(self):
        # Define the available callbacks
        self._callbacks = {
            "on_pretrain_routine_start": [],
            "on_pretrain_routine_end": [],
            "on_train_start": [],
            "on_train_epoch_start": [],
            "on_train_batch_start": [],
            "optimizer_step": [],
            "on_before_zero_grad": [],
            "on_train_batch_end": [],
            "on_train_epoch_end": [],
            "on_val_start": [],
            "on_val_batch_start": [],
            "on_val_image_end": [],
            "on_val_batch_end": [],
            "on_val_end": [],
            "on_fit_epoch_end": [],
            "on_model_save": [],
            "on_train_end": [],
            "teardown": [],
        }

    def add_callbacks(self, name: str, callbacks: dict[str, Callable]):
        for hook, callback in callbacks.items():
            self.register_action(hook, name=name, callback=callback)

    def register_action(self, hook, name="", callback=None):
        """
        Register a new action to a callback hook

        Args:
            hook: The callback hook name to register the action to
            name: The name of the action for later reference
            callback: The callback to fire
        """
        assert (
            hook in self._callbacks
        ), f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({"name": name, "callback": callback})

    def get_registered_actions(self, hook=None):
        """ "
        Returns all the registered actions by callback hook

        Args:
            hook: The name of the hook to check, defaults to all
        """
        return self._callbacks[hook] if hook else self._callbacks

    def run(self, hook, state, *args, **kwargs):
        """
        Run all the actions for a given hook

        Args:
            hook: The name of the hook to run
        """
        for action in self._callbacks[hook]:
            action["callback"](state, *args, **kwargs)
