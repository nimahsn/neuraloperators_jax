import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import ArrayLike
from typing import Callable, Union
from modules.auxiliary import autoregressive_predict
from torch.utils.data import DataLoader

def test_unrolling(model: eqx.Module, 
                    trajectories: Union[ArrayLike, DataLoader], 
                    history_steps: int, 
                    future_steps: int, 
                    total_steps: int = None,
                    reduce: str = "mean") -> ArrayLike:
    """
    Computes the error of the unrolling scheme for a given model and a batch of trajectories.

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
    
    if isinstance(trajectories, DataLoader):
            error = jnp.concatenate([test_unrolling(model, u, history_steps, future_steps, total_steps, "none") for u, *_ in trajectories], axis=0)
    else:
        if total_steps is None:
            total_steps = trajectories.shape[1]
        u_input = trajectories[:, :history_steps]
        u_target = trajectories[:, history_steps:]
        num_iters = jnp.ceil((total_steps - history_steps) / future_steps).astype(jnp.int32)
        predictions = jnp.zeros((u_input.shape[0], num_iters * future_steps + history_steps, *trajectories.shape[2:]))
        predictions = predictions.at[:, :history_steps].set(u_input)
        inputs = u_input
        for i in range(num_iters):
            preds = eqx.filter_jit(eqx.filter_vmap(model))(inputs)
            predictions = predictions.at[:, history_steps + i*future_steps:history_steps + (i+1)*future_steps].set(preds)
            inputs = predictions[:, (i+1)*future_steps:history_steps + (i+1)*future_steps]
        preds = predictions[:total_steps]
        error = jnp.mean((preds[:, history_steps:] - u_target) ** 2, axis=-1)

    if reduce == "mean":
        error = jnp.mean(error)
    elif reduce == "sum":
        error = jnp.sum(error)
    return error

