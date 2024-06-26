import jax
import jax.numpy as jnp
from jax import random, vmap, jit
import equinox as eqx
from functools import partial
from jaxtyping import ArrayLike

class FourierFeatures(eqx.Module):
    '''
    Fourier features layer. Generates a set of sinusoids with random weights and frequencies.

    Args:
        weights: jax.Array, shape (input_dim, num_features), optional
            The weights of the sinusoids. If None, they are randomly generated.
        frequency: float, optional
            The frequency of the sinusoids.
        scale: float, optionala
            The scale of the sinusoids.
        input_dim: int, optional
            The dimension of the input space.
        num_features: int, optional
            The number of features to generate.
        key: jax.random.PRNGKey, optional
            The random key to use for generating the weights.
        dtype: jax.numpy.dtype, optional
            The dtype of the weights.
    '''
    
    weights: jax.Array = eqx.field(static=True)
    frequency: float = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    def __init__(self, weights=None, frequency=2*jnp.pi, scale=1., input_dim=None, num_features=None,
                 key=random.PRNGKey(0), dtype=jnp.float32):
        if weights is None and (input_dim is None or num_features is None):
            raise ValueError('Must specify either weights or input_dim and num_features.')
        self.scale = scale
        if weights is None:
            key, subkey = random.split(key)
            weights = random.normal(subkey, (input_dim, num_features), dtype=dtype)
        self.weights = weights
        self.frequency = frequency

    def __call__(self, inputs, **kwargs):
        return jnp.concatenate([self.scale * jnp.sin(self.frequency * jnp.dot(inputs, self.weights)),
                                self.scale * jnp.cos(self.frequency * jnp.dot(inputs, self.weights))])

def autoregressive_predict(model: eqx.Module, inputs: ArrayLike, history_steps: int, future_steps: int, total_steps: int) -> ArrayLike:
    """
    Autoregressive prediction function for a model.

    Args:
    - model: eqx.Module, the model to be used for prediction
    - input: ArrayLike, the input data for prediction
    - history_steps: int, the number of history steps to use for prediction, i.e. the number of input steps
    - future_steps: int, the number of future steps to predict, i.e. the number of output steps
    - total_steps: int, the total number of steps to predict

    Returns:
    - prediction: ArrayLike, the predicted output of all time steps, including the input steps
    """
    num_iters = jnp.ceil(total_steps / future_steps).astype(jnp.int32)
    predictions = jnp.zeros((num_iters * future_steps + history_steps, *inputs.shape[1:]))
    predictions = predictions.at[:history_steps].set(inputs)
    for i in range(num_iters):
        preds = model(inputs)
        predictions = predictions.at[history_steps + i*future_steps:history_steps + (i+1)*future_steps].set(preds)
        inputs = predictions[(i+1)*future_steps:history_steps + (i+1)*future_steps]

    return predictions[:history_steps + total_steps]
        
