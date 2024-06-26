import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import ArrayLike
from typing import Callable
from modules.auxiliary import autoregressive_predict

def error_unrolling(model: eqx.Module, 
                    u: ArrayLike, 
                    history_steps: int, 
                    future_steps: int, 
                    error_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
                    total_steps: int = None,
                    reduce: str = "mean") -> ArrayLike:
    """
    Computes the error of the unrolling scheme for a given model and input data.

    Args:
    - model: eqx.Module, the model to be used
    - u: ArrayLike, the full pde trajectry
    - history_steps: int, the number of input time steps
    - future_steps: int, the number of predicted time steps at each unrolling step
    - error_fn: Callable[[ArrayLike, ArrayLike], ArrayLike], the error function to be used
    - total_steps: int, the total number of steps to be unrolled (including the input steps). \
        If None, the total number of steps is equal to the length of the input trajectory
    - reduce: str, the reduction method to be used on the temporal dimension. Options are "none", "mean", "sum"

    Returns:
    - error: ArrayLike, the error of the unrolling scheme
    """
    if total_steps is None:
        total_steps = u.shape[0]
    inputs = u[:history_steps]
    outputs = u[history_steps:total_steps]
    predictions = autoregressive_predict(model, inputs, history_steps, 
                                         future_steps, total_steps - history_steps)[history_steps:]
    error = error_fn(predictions, outputs)
    if reduce == "mean":
        error = jnp.mean(error)
    elif reduce == "sum":
        error = jnp.sum(error)
    return error
