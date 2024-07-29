import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from .losses import loss_mse
from torch.utils.data import DataLoader
from typing import Callable, Tuple, List, Sequence, Dict
from jaxtyping import ArrayLike, PRNGKeyArray
import torch.utils.data
import os, sys
from functools import partial

TRAIN_LOSS_KEY = "train_loss"
VAL_LOSS_KEY = "val_loss"
EQUIVARIANCE_LOSS_KEY = "equivariance_loss"
REGRESSION_LOSS_KEY = "regression_loss"


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
                print_every: int=100,
                **kwargs) -> Tuple[eqx.Module, optax.OptState, Dict[str, ArrayLike]]:
    """
    A plain fit function for training a model.

    Args:
    - model: eqx.Module, the model to be trained
    - dataloader_train: DataLoader, the training data loader
    - loss_fn_batch: Callable[[eqx.Module, ArrayLike, ArrayLike], ArrayLike], the loss function for a batch of data. \
        Should be vmapped.
    - optimizer: optax.GradientTransformation, the optimizer to be used
    - num_epochs: int, the number of epochs to train the model
    - dataloader_val: DataLoader, the validation data loader. If None, no validation is performed.
    - opt_state: optax.OptState, the optimizer state. If None, the optimizer state is initialized.
    - loss_has_aux: bool, whether the loss function returns an auxiliary output
    - train_step_fn: Callable[[eqx.Module, ArrayLike, ArrayLike, Callable[[eqx.Module, ArrayLike, ArrayLike], \
        ArrayLike], optax.GradientTransformation, optax.OptState, bool], Tuple[eqx.Module, optax.OptState, ArrayLike]], \
            the function to be used for a single training step. Default is train_step. The given function will be jitted\
                  with eqx.filter_jit.
    - callbacks: List[Callable[[eqx.Module, optax.OptState, int], None], a list of callback functions to be called after each epoch. \
        Each callback function should take the model, optimizer state, and epoch number as arguments.
    - print_every: int, the number of steps between reporting the training and validation loss. If None, no printing is performed.
    - kwargs: additional keyword arguments to be passed to the train_step_fn

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
            model, opt_state, loss, *aux = stepping_fn(model, inputs, outputs, loss_fn_batch, optimizer, opt_state, 
                                                       loss_has_aux, **kwargs)
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
            model, opt_state, history, kwargs = callback(model, opt_state, history, kwargs, epoch)

    return model, opt_state, history

def fit_symmetric(model: eqx.Module, 
                dataloader_train: DataLoader, 
                symmetries: List[Callable],
                optimizer: optax.GradientTransformation,
                num_epochs: int,
                window_size: int,
                total_steps: int,
                max_pushback_steps: int=0,
                first_pushback_epoch: int=20,
                dataloader_val: DataLoader=None,
                opt_state: optax.OptState=None,
                callbacks: List[Callable[[eqx.Module, optax.OptState, int], None]] = [],
                history: Dict[str, ArrayLike] = None,
                print_every: int=100,
                *,
                key: PRNGKeyArray,
                **kwargs) -> Tuple[eqx.Module, optax.OptState, Dict[str, ArrayLike]]:
    """
    A symmetry enforcing fit function for training a model.

    Args:
    - model: eqx.Module, the model to be trained
    - dataloader_train: DataLoader, the training data loader
    - symmetries: List[Callable], a list of symmetry functions to be enforced. See data.symmetries
    - optimizer: optax.GradientTransformation, the optimizer to be used
    - num_epochs: int, the number of epochs to train the model
    - window_size: int, the size of input and output windows
    - total_steps: int, the total number of temporal steps in each trajectory
    - max_pushback_steps: int, the maximum number of pushback steps. If 0, no pushback is performed.
    - first_pushback_epoch: int, the epoch at which pushback steps are first introduced
    - dataloader_val: DataLoader, the validation data loader. If None, no validation is performed.
    - opt_state: optax.OptState, the optimizer state. If None, the optimizer state is initialized.
    - callbacks: List[Callable[[eqx.Module, optax.OptState, int], None], a list of callback functions to be called after each epoch. \
        Each callback function should take the model, optimizer state, and epoch number as arguments.
    - history: Dict[str, ArrayLike], a dictionary containing the training and validation loss history
    - print_every: int, the number of steps between reporting the training and validation loss. If None, no printing is performed.
    - key: PRNGKeyArray, the random key to be used for training
    - kwargs: additional keyword arguments to be passed to the training step function

    Returns:
    - model: eqx.Module, the trained model
    - opt_state: optax.OptState, the final optimizer state
    - history: Dict[str, ArrayLike], a dictionary containing the training and validation loss history
    """
    if opt_state is None:
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array_like))
    if history is None:
        history = {TRAIN_LOSS_KEY: [], EQUIVARIANCE_LOSS_KEY: [], REGRESSION_LOSS_KEY: []}
        if dataloader_val is not None:
            history[VAL_LOSS_KEY] = []

    alpha = kwargs.get("alpha", 0.5)
    gamma = kwargs.get("gamma", 1.0)
    assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"
    assert max_pushback_steps * window_size < total_steps - 2 * window_size, "Pushback steps too large"

    def loss(params, static, input_1, input_2, label_1, label_2, TX_1, symmetry_params, pushback_steps: int,\
             alpha: float, gamma: float):
        """
        Loss function for symmetry enforcing training.

        Args:
        - params: eqx.Module, the model parameters
        - static: eqx.Module, the static model parameters
        - input_1: ArrayLike, the first input data batch
        - input_2: ArrayLike, the second input data batch, augmented input data from the first batch
        - label_1: ArrayLike, the first output data batch
        - label_2: ArrayLike, the second output data batch
        - TX_1: ArrayLike, the sampling grid for the first input data batch
        - symmetry_params: ArrayLike, the augmentation parameters for the lie symmetries
        - pushback_steps: int, the number of pushback steps
        - alpha: float, balance paramter between the original and augmented data. \
            If alpha=1, only the original data is used, if 0, only the augmented data is used. Should be between 0 and 1.
        - gamma: float, the weight of the equivariance loss
        """
        model = eqx.combine(params, static)
        preds_1 = eqx.filter_vmap(model)(input_1)
        preds_2 = eqx.filter_vmap(model)(input_2)
        for i in range(pushback_steps):
            input_1 = eqx.filter_vmap(model)(input_1)
            input_2 = eqx.filter_vmap(model)(input_2)
        preds_1 = eqx.filter_vmap(model)(jax.lax.stop_gradient(input_1))
        preds_2 = eqx.filter_vmap(model)(jax.lax.stop_gradient(input_2))

        loss_pred = (2 * alpha * jnp.mean(jnp.square(preds_1 - label_1)) + 2 * (1-alpha) * jnp.mean(jnp.square(preds_2 - label_2))) / 2
        for i, symmetry in enumerate(symmetries):
            preds_1, TX_1 = jax.vmap(symmetry, in_axes=(0, 0, 0))(preds_1, TX_1, symmetry_params[i])
        loss_ag = jnp.mean(jnp.square(preds_1 - label_2))

        losses = {REGRESSION_LOSS_KEY: loss_pred, EQUIVARIANCE_LOSS_KEY: loss_ag}
        return loss_pred + loss_ag * gamma, losses

    @partial(jax.jit, static_argnames=("static", "pushback_steps", "alpha", "gamma"))
    def jitted_inner_step(params, static, opt_state, U, T, X, start_time, pushback_steps: int,\
                          alpha: float, gamma: float, *, key: PRNGKeyArray):
        symmetry_params = jax.random.uniform(key, shape=(len(symmetries), U.shape[0]), minval=-0.5, maxval=0.5)
        TX = jnp.stack([T, X], axis=1)
        U_ag, TX_ag = U, TX
        for i, symmetry in enumerate(symmetries):
            U_ag, TX_ag = jax.vmap(symmetry, in_axes=(0, 0, 0))(U_ag, TX_ag, symmetry_params[i])
        U_ag_history = jax.lax.dynamic_slice_in_dim(U_ag, start_time, window_size, axis=1)
        U_ag_future = jax.lax.dynamic_slice_in_dim(U_ag, start_time + (pushback_steps + 1) * window_size,
                                                   window_size, axis=1)
        U_history = jax.lax.dynamic_slice_in_dim(U, start_time, window_size, axis=1)
        U_future = jax.lax.dynamic_slice_in_dim(U, start_time + (pushback_steps + 1) * window_size,
                                                window_size, axis=1)
        TX_future = jax.lax.dynamic_slice_in_dim(TX, start_time + (pushback_steps + 1) * window_size,
                                                 window_size, axis=2)
        loss_grad_f = eqx.filter_value_and_grad(loss, has_aux=True)
        (loss_val, loss_dict), grads = loss_grad_f(params, static, U_history, U_ag_history, U_future, U_ag_future, 
                                                   TX_future, symmetry_params, pushback_steps, alpha, gamma)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_dict

    params, static = eqx.partition(model, eqx.is_array_like)
    mean_regression = 0.0
    mean_equivariance = 0.0
    mean_train = 0.0

    for epoch in range(num_epochs):
        step = 0
        if epoch < first_pushback_epoch:
            max_pushback = 0
        else:
            max_pushback = max_pushback_steps
        for U, T, X, *_ in dataloader_train:
            key, sub_key_1, sub_key_2 = jax.random.split(key, 3)
            max_start_time = total_steps - (2 * window_size + max_pushback * window_size)
            start_times = jnp.arange(max_start_time)
            start_times_shuff = jax.random.permutation(sub_key_1, start_times, independent=True)
            pushback_steps = jax.random.choice(sub_key_2, jnp.arange(max_pushback + 1), shape=(len(start_times_shuff),))
            for start, push in zip(start_times_shuff, pushback_steps):
                key, sub_key = jax.random.split(key)
                params, opt_state, loss_dict = jitted_inner_step(params, static, opt_state, U, T, X, start, push.item(),
                                                                 alpha, gamma, key=sub_key)
                step += 1
                mean_regression += loss_dict[REGRESSION_LOSS_KEY]
                mean_equivariance += loss_dict[EQUIVARIANCE_LOSS_KEY]
                mean_train += mean_equivariance + mean_regression

            if print_every is not None and step % print_every == 0:
                print(f"Epoch {epoch}, step {step}, loss: {loss_dict}")    
        history[TRAIN_LOSS_KEY].append(mean_train / step)
        history[REGRESSION_LOSS_KEY].append(mean_regression / step)
        history[EQUIVARIANCE_LOSS_KEY].append(mean_equivariance / step)
        mean_train = 0.0
        mean_regression = 0.0
        mean_equivariance = 0.0
        
        model = eqx.combine(params, static)
        if dataloader_val is not None:
            val_loss = 0.0
            for inputs, labels, *_ in dataloader_val:
                val_loss += eqx.filter_jit(loss_mse)(model, inputs, labels)
            val_loss /= len(dataloader_val)
            print(f"Validation loss: {val_loss}")
            history[VAL_LOSS_KEY].append(val_loss)  
        for callback in callbacks:
            model, opt_state, history, kwargs = callback(model, opt_state, history, kwargs, epoch)
    
    history = jax.tree.map(lambda x: jax.device_get(x), history)
    return model, opt_state, history

