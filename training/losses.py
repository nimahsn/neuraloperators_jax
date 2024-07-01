"""
Loss functions for equinox models. Models are vmapped inside the loss functions to allow for batched inputs. 
"""

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import ArrayLike
from typing import Callable

def loss_mse(model: eqx.Module, inputs: ArrayLike, outputs: ArrayLike) -> ArrayLike:
    preds = eqx.filter_vmap(model)(inputs)
    loss = jnp.mean((preds - outputs)**2)
    return loss

def loss_nmse(model: eqx.Module, inputs: ArrayLike, outputs: ArrayLike) -> ArrayLike:
    # Normalized MSE
    preds = eqx.filter_vmap(model)(inputs)
    loss = (preds - outputs) ** 2 / outputs ** 2
    loss = jnp.mean(loss)    
    return loss

