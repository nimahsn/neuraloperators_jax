import jax
import jax.numpy as jnp
from jax import random, vmap, jit
import equinox as eqx
from functools import partial

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

