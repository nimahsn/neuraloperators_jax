import jax
from jax import random, grad, vmap, jit
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal
import equinox as eqx
from jaxtyping import ArrayLike, Complex, Array, PRNGKeyArray
from typing import Callable

from functools import partial

DATA_INPUT = 'x'
DATA_OUTPUT = 'u'


class SpectraclConv1d(eqx.Module):
    """
    Spectral Convolutional Layer for 1d inputs. The layer first transforms the input
    into Fourier space, applies a convolution in Fourier space, and then transforms
    the result back to the spatial domain. The layer is parameterized by a set of
    spectral weights. The input is assumed to be real-valued of shape `(in_c, d1)`.

    Args:
        in_c: int
            The number of input channels.
        out_c: int
            The number of output channels.
        k_modes: int
            The layer truncates the Fourier series to the first `k_modes` modes. Remaining modes are set to zero.
    """
    weight_r: Array
    in_c: int = eqx.field(static=True)
    out_c: int = eqx.field(static=True)
    k_modes: int = eqx.field(static=True)

    def __init__(self, in_c: int, out_c: int, k_modes: int, key: PRNGKeyArray, initializer: Callable = jax.nn.initializers.glorot_normal):
        self.in_c = in_c
        self.out_c = out_c
        self.k_modes = k_modes
        self.weight_r = initializer()(key, (in_c, out_c, k_modes), jnp.complex64)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """
        Apply the layer to the input.

        Args:
            x: ArrayLike
                The input tensor. Should be of shape `(in_c, d1)`.

        Returns:
            ArrayLike
                The output tensor. Will be of shape `(out_c, d1)`.
        """
        x_ft = jnp.fft.rfft(x) # shape: (in_c, d1//2 + 1)
        # WARNING: rfft uses the conjugate symmetry property of the Fourier transform to reduce the number of computations.
        # So, when truncating to k_modes, maybe we should truncate to k_modes//2 + 1, to keep the conjugate symmetry property?
        # apply the convolution in Fourier space for the first k modes
        out_ft = jnp.zeros((self.out_c, x_ft.shape[1]), dtype=jnp.complex64)
        out_ft = out_ft.at[:, :self.k_modes].set(jnp.einsum('ik,iok->ok', x_ft[:, :self.k_modes], self.weight_r))
        return jnp.fft.irfft(out_ft) # shape: (out_c, d1)        


class FNOBlock1d(eqx.Module):
    spec_conv: eqx.Module
    residual_local_transform: eqx.nn.Conv1d
    k_modes: int = eqx.field(static=True)
    activation: Callable
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(self, k_modes: int, in_channels: int, out_channels: int, activation: Callable, key: PRNGKeyArray):
        self.k_modes = k_modes
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        key, subkey = random.split(key)
        self.spec_conv = SpectraclConv1d(in_channels, out_channels, k_modes, key)
        self.residual_local_transform = eqx.nn.Conv1d(in_channels, out_channels, 1, 1, key=subkey)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """
        Apply the layer to the input.

        Args:
            x: ArrayLike
                The input tensor. Should be of shape `(in_c, d1)`.

        Returns:
            ArrayLike
                The output tensor. Will be of shape `(out_c, d1)`.
        """
        return self.activation(self.spec_conv(x) + self.residual_local_transform(x))
        
class FNO1d(eqx.Module):
    fourier_blocks: tuple[FNOBlock1d]
    projection_input: eqx.Module
    projection_output: eqx.Module
    in_channels: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(self, in_channels: int, out_channels: int, width: int, k_modes: int, depth: int, activation: Callable, key: PRNGKeyArray, p_in: eqx.Module=None, p_out: eqx.Module=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        key, *subkeys = random.split(key, depth + 3)
        self.projection_input = p_in if p_in is not None else eqx.nn.Conv1d(in_channels, width, 1, 1, key=subkeys[0])
        self.projection_output = p_out if p_out is not None else eqx.nn.Sequential([eqx.nn.Conv1d(width, 128, 1, 1, key=subkeys[1]),
                                                                                    eqx.nn.Conv1d(128, out_channels, 1, 1, key=subkeys[2])])
        self.fourier_blocks = tuple(FNOBlock1d(k_modes, width, width, activation, subkey) for subkey in subkeys[3:])

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """
        Apply the layer to the input.

        Args:
            x: ArrayLike
                The input tensor. Should be of shape `(in_c, d1)`.

        Returns:
            ArrayLike
                The output tensor. Will be of shape `(out_c, d1)`.
        """
        x = self.projection_input(x)
        for block in self.fourier_blocks:
            x = block(x)
        return self.projection_output(x)
    
