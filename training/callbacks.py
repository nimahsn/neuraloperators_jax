"""
Callback functions for training Equinox models. Each callback function should have the following signature:
    Args:
    - model: eqx.Module, the model to be trained
    - opt_state: optax.OptState, the optimizer state
    - epoch: int, the current epoch number
    Returns:
    - None
"""

import os, sys
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from typing import Callable, Dict, Tuple, Any
from jaxtyping import ArrayLike, PRNGKeyArray
from utils.serial import save_model
from training.metrics import test_unrolling
from pathlib import Path

class Callback:
    """
    Base class for all callbacks. Child classes should implement the `call` method.
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.call(*args, **kwds)

    def call(self, model: eqx.Module, opt_state: optax.OptState, history: Dict[str, ArrayLike], training_config: Dict[str, ArrayLike], epoch: int) \
        -> Tuple[eqx.Module, optax.OptState, Dict[str, ArrayLike], Dict[str, ArrayLike]]:
        """
        The main method to be implemented in the child class.

        Args:
        - model: eqx.Module, the model to be trained
        - opt_state: optax.OptState, the optimizer state
        - history: Dict[str, ArrayLike], the history of the training process
        - training_config: Dict[str, ArrayLike], the configuration of the training process
        - epoch: int, the current epoch number

        Returns:
        - Tuple containing the updated model, optimizer state, history, and training configuration
        """
        raise NotImplementedError("The `call` method should be implemented in the child class.")

class CheckpointCallback(Callback):
    """
    A callback class for saving the model and optimizer state at specified intervals.
    """
    def __init__(self, checkpoint_dir: str, checkpoint_name: str, hyper_params: Dict, save_every: int = 50):
        """
        Initializes the CheckpointCallback instance.

        Args:
        - checkpoint_dir: str, the directory to save the checkpoint.
        - checkpoint_name: str, the name of the checkpoint file.
        - hyper_params: Dict, hyperparameters to be saved along with the model.
        - save_every: int, the frequency of saving the checkpoint. Default is 10 (save every 10 epochs).
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.hyper_params = hyper_params
        self.save_every = save_every
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def call(self, model: eqx.Module, opt_state: optax.OptState, history: Dict[str, ArrayLike], training_config: Dict[str, ArrayLike], epoch: int) \
        -> Tuple[eqx.Module, optax.OptState, Dict[str, ArrayLike], Dict[str, ArrayLike]]:
        """
        Saves the model and optimizer state at the specified interval and returns the inputs unchanged.

        Args:
        - model: eqx.Module, the model being trained.
        - opt_state: optax.OptState, the current optimizer state.
        - history: Dict[str, ArrayLike], a dictionary to store training history.
        - training_config: Dict[str, ArrayLike], a dictionary containing training configurations.
        - epoch: int, the current epoch count.

        Returns:
        - Tuple containing the model, optimizer state, history, and training configuration unchanged.
        """
        if epoch % self.save_every == 0:
            name = self.checkpoint_name + f"_epoch_{epoch}"
            path = os.path.join(self.checkpoint_dir, name)
            save_model(path, self.hyper_params, model)
            print(f"Checkpoint saved at {path}")
        return model, opt_state, history, training_config

class UnrollingTestCallback(Callback):
    """
    A callback class for performing unrolling tests at specified intervals.
    """
    def __init__(self, trajectories: str, history_steps: int, future_steps: int, total_steps: int, test_every: int = 10,
                 add_history: bool = True, history_key: str = "unrolling_error"):
        """
        Initializes the UnrollingTestCallback instance.

        Args:
        - trajectories: str, path to the trajectories data.
        - history_steps: int, the number of history steps to consider.
        - future_steps: int, the number of future steps to predict.
        - total_steps: int, the total number of steps in each trajectory.
        - test_every: int, the frequency of performing the unrolling test. Default is 10 (test every 10 epochs).
        """
        self.trajectories = trajectories
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.total_steps = total_steps
        self.test_every = test_every
        self.add_history = add_history
        self.history_key = history_key

    def call(self, model: eqx.Module, opt_state: optax.OptState, history: Dict[str, ArrayLike], training_config: Dict[str, ArrayLike], epoch: int) \
        -> Tuple[eqx.Module, optax.OptState, Dict[str, ArrayLike], Dict[str, ArrayLike]]:
        """
        Performs the unrolling test at the specified interval and returns the inputs unchanged.

        Args:
        - model: eqx.Module, the model being trained.
        - opt_state: optax.OptState, the current optimizer state.
        - history: Dict[str, Any], a dictionary to store training history.
        - training_config: Dict[str, Any], a dictionary containing training configurations.
        - epoch: int, the current epoch count.

        Returns:
        - Tuple containing the model, optimizer state, history, and training configuration unchanged.
        """
        if epoch % self.test_every == 0:
            error = test_unrolling(model, self.trajectories, self.history_steps, self.future_steps, self.total_steps, "mean")
            if self.add_history:
                history[self.history_key] = history.get(self.history_key, []) + [error]
            print(f"Unrolling test error at epoch {epoch}: {error}")
        return model, opt_state, history, training_config

