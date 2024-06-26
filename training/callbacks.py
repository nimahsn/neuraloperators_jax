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
from typing import Callable, Dict
from jaxtyping import ArrayLike, PRNGKeyArray
from utils.serial import save_model
from pathlib import Path

def checkpoint_callback(checkpoint_dir: str, checkpoint_name: str, hyper_params: Dict, save_every: int = 10) -> Callable[[eqx.Module, optax.OptState, int], None]:
    """
    A callback function to save the model and optimizer state after each epoch.

    Args:
    - checkpoint_dir: str, the directory to save the checkpoint
    - checkpoint_name: str, the name of the checkpoint file
    - save_every: int, the frequency of saving the checkpoint. Default is 1 (save after each epoch).

    Returns:
    - Callable[[eqx.Module, optax.OptState, int], None], the callback function to save the checkpoint
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    def checkpoint_cb(model, opt_state, epoch):
        if epoch % save_every == 0:
            name = checkpoint_name + f"_epoch_{epoch}"
            path = os.path.join(checkpoint_dir, name)
            save_model(path, hyper_params, model)

    return checkpoint_cb

