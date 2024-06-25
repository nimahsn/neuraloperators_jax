import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from torch.utils.data import DataLoader
from typing import Callable, Tuple, List, Sequence, Dict
from jaxtyping import ArrayLike
import torch.utils.data
import os, sys

TRAIN_LOSS_KEY = "train_loss"
VAL_LOSS_KEY = "val_loss"


def train_step(model: eqx.Module, 
               inputs: ArrayLike, 
               outputs: ArrayLike,
               loss_fn_batch: Callable[[eqx.Module, ArrayLike, ArrayLike], ArrayLike],
               optimizer: optax.GradientTransformation,
               opt_state: optax.OptState,
               loss_has_aux: bool = False) -> Tuple[eqx.Module, optax.OptState, ArrayLike]:
    """
    Train step function for a single batch of data.
    
    Args:
    - model: eqx.Module, the model to be trained
    - inputs: ArrayLike, the input data batch
    - outputs: ArrayLike, the output data batch
    - loss_fn_batch: Callable[[eqx.Module, ArrayLike, ArrayLike], ArrayLike], the loss function for a batch of data. 
    - optimizer: optax.GradientTransformation, the optimizer to be used
    - opt_state: optax.OptState, the optimizer state
    - loss_has_aux: bool, whether the loss function returns an auxiliary output

    Returns:
    - model: eqx.Module, the updated model
    - opt_state: optax.OptState, the updated optimizer state
    - loss: ArrayLike, the loss value
    - aux: ArrayLike, the auxiliary output of the loss function (if loss_has_aux=True)
    """
    if loss_has_aux:
        (loss, aux), grad = eqx.filter_value_and_grad(loss_fn_batch, has_aux=loss_has_aux)(model, inputs, outputs)
    else:
        loss, grad = eqx.filter_value_and_grad(loss_fn_batch, has_aux=loss_has_aux)(model, inputs, outputs)
    updates, opt_state = optimizer.update(grad, opt_state)
    model = optax.apply_updates(model, updates)
    if loss_has_aux:
        return model, opt_state, loss, aux
    else:
        return model, opt_state, loss

def fit_plain(model: eqx.Module, 
                dataloader_train: DataLoader, 
                loss_fn_batch: Callable[[eqx.Module, ArrayLike, ArrayLike], ArrayLike],
                optimizer: optax.GradientTransformation,
                num_epochs: int,
                dataloader_val: DataLoader=None,
                opt_state: optax.OptState=None,
                loss_has_aux: bool = False,
                train_step_fn: Callable[[eqx.Module, 
                                         ArrayLike, 
                                         ArrayLike, 
                                         Callable[[eqx.Module, ArrayLike, ArrayLike], ArrayLike], 
                                         optax.GradientTransformation, optax.OptState, bool], 
                                         Tuple[eqx.Module, optax.OptState, ArrayLike]] = train_step,
                callbacks: List[Callable[[eqx.Module, optax.OptState, int], None]] = [],
                history: Dict[str, ArrayLike] = None,
                print_every: int=100) -> eqx.Module:
    """
    A plain fit function for training a model.

    Args:
    - model: eqx.Module, the model to be trained
    - dataloader_train: DataLoader, the training data loader
    - loss_fn_batch: Callable[[eqx.Module, ArrayLike, ArrayLike], ArrayLike], the loss function for a batch of data. Should be vmapped.
    - optimizer: optax.GradientTransformation, the optimizer to be used
    - num_epochs: int, the number of epochs to train the model
    - dataloader_val: DataLoader, the validation data loader. If None, no validation is performed.
    - opt_state: optax.OptState, the optimizer state. If None, the optimizer state is initialized.
    - loss_has_aux: bool, whether the loss function returns an auxiliary output
    - train_step_fn: Callable[[eqx.Module, ArrayLike, ArrayLike, Callable[[eqx.Module, ArrayLike, ArrayLike], \
        ArrayLike], optax.GradientTransformation, optax.OptState, bool], Tuple[eqx.Module, optax.OptState, ArrayLike]], \
            the function to be used for a single training step. Default is train_step. The given function will be jitted with eqx.filter_jit.
    - callbacks: List[Callable[[eqx.Module, optax.OptState, int], None], a list of callback functions to be called after each epoch. \
        Each callback function should take the model, optimizer state, and epoch number as arguments.
    - print_every: int, the number of steps between reporting the training and validation loss. If None, no printing is performed.

    Returns:
    - model: eqx.Module, the trained model
    - opt_state: optax.OptState, the final optimizer state
    - history: Dict[str, ArrayLike], a dictionary containing the training and validation loss history
    """
    if opt_state is None:
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array_like))
    if history is None:
        history = {TRAIN_LOSS_KEY: []}
        if dataloader_val is not None:
            history[VAL_LOSS_KEY] = []

    stepping_fn = eqx.filter_jit(train_step_fn)
    jitted_loss_fn_batch = eqx.filter_jit(loss_fn_batch)

    for epoch in range(num_epochs):
        for step, (inputs, outputs, *_) in enumerate(dataloader_train):
            model, opt_state, loss = stepping_fn(model, inputs, outputs, loss_fn_batch, optimizer, opt_state, loss_has_aux)
            if print_every is not None and step % print_every == 0:
                print(f"Epoch {epoch}, step {step}, loss: {loss}")
        if dataloader_val is not None:
            val_loss = 0
            for inputs, outputs, *_ in dataloader_val:
                val_loss += jitted_loss_fn_batch(model, inputs, outputs)
            val_loss /= len(dataloader_val)
            print(f"Validation loss: {val_loss}")
            history[VAL_LOSS_KEY].append(val_loss)
        history[TRAIN_LOSS_KEY].append(loss)
        for callback in callbacks:
            callback(model, opt_state, epoch)

    return model, opt_state, history

    