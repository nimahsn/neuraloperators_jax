import jax
import jax.numpy as jnp
from typing import List, Union, Tuple
from jaxtyping import ArrayLike, PRNGKeyArray

def fourier_shift(u: ArrayLike, eps: float=0., dim: int=-1, order: int=0) -> jax.Array:
    """
    Apply a Fourier shift to a field.

    Args:
        u: The field to shift.
        eps: The parameter of the transformation.
        dim: The dimension to shift.
        order: The order of the derivative.

    Returns:
        The shifted field.
    """
    
    assert dim < 0
    n = u.shape[dim]
    u_hat = jnp.fft.rfft(u, axis=dim, norm='ortho')
    
    omega = jnp.arange(n//2 + 1)
    if n % 2 ==0:
        omega = omega.at[-1].set(omega[-1] * 0)

    shift = jnp.exp(-2j * jnp.pi * omega * eps)
    shift = (-2j * jnp.pi * omega) ** order * shift
    u_hat = u_hat * shift
    return jnp.fft.irfft(u_hat, n=n, axis=dim, norm='ortho')

def linear_shift(u: jax.Array, eps: float=0., dim: int=-1, order: int=0) -> jax.Array:
    raise NotImplementedError("Not implemented yet.")

def to_coords(t: jax.Array, x: jax.Array) -> jax.Array:
    """
    Create a stack of time and space coordinates.

    Args:
        t: The time coordinates.
        x: The space coordinates.

    Returns:
        A stack of time and space mesh.
    """

    T, X = jnp.meshgrid(t, x)
    return jnp.stack([T, X], axis=-1)

def translation_group(sample: Tuple[jax.Array], eps: float=None, min_eps=-0.5, max_eps=0.5, shift_fn=fourier_shift, *, key: PRNGKeyArray=None) -> jax.Array:
    """
    Apply the spatial translation group transformation to a sample.

    Args:
        sample: A tuple of the form (u, X) where u is the field and X is the coordinates. X should be a stack of time and space coordinates.
        length: The length of the domain. If None, it is inferred from the coordinates.
        eps: The parameter of the transformation.
        shift_fn: The function to apply the shift.

    Returns:
        A tuple of the form (u', X) where u' is the transformed field.
    """
    if eps is None:
        eps = jax.random.uniform(key, shape=(), minval=min_eps, maxval=max_eps)
    u, X = sample
    output = shift_fn(u, eps=eps, dim=-1)
    return output, X

def scale_group(sample: Tuple[jax.Array], eps: float=None, min_eps=-0.5, max_eps=0.5, shift_fn=fourier_shift, *, key: PRNGKeyArray=None) -> jax.Array:
    """
    Apply the scale group transformation to a sample.

    Args:
        sample: A tuple of the form (u, X) where u is the field and X is the coordinates. X should be a stack of time and space coordinates.
        length: The length of the domain. If None, it is inferred from the coordinates.
        eps: The parameter of the transformation.
        shift_fn: The function to apply the shift.

    Returns:
        A tuple of the form (u', X) where u' is the transformed field.
    """    
    if eps is None:
        eps = jax.random.uniform(key, shape=(), minval=min_eps, maxval=max_eps)
    u, X = sample
    X = X.at[0, ...].set(X[0, ...] * jnp.exp(-3 * eps))
    X = X.at[1, ...].set(X[1, ...] * jnp.exp(-eps))
    u = u * jnp.exp(2 * eps)

    return u, X

def gallilean_group(sample: Tuple[jax.Array], eps: float=None, min_eps=-0.5, max_eps=0.5, shift_fn=fourier_shift, *, key: PRNGKeyArray=None) -> jax.Array:
    """
    Apply the Gallilean group transformation to a sample.

    Args:
        sample: A tuple of the form (u, X) where u is the field and X is the coordinates. X should be a stack of time and space coordinates.
        length: The length of the domain. If None, it is inferred from the coordinates.
        eps: The parameter of the transformation.
        shift_fn: The function to apply the shift.

    Returns:
        A tuple of the form (u', X) where u' is the transformed field.
    """
    if eps is None:
        eps = jax.random.uniform(key, shape=(), minval=min_eps, maxval=max_eps)
    u, X = sample
    length = X[1, 0, -1]
    t = X[0, :, 0]
    shift = -(eps * t[:, None]) / length
    output = shift_fn(u, eps=shift, dim=-1) - eps, X
    return output

