import jax
from jax import random, grad, vmap, jit
import jax.numpy as jnp
from jax.nn.initializers import glorot_uniform
import equinox as eqx
from jaxtyping import ArrayLike, Complex, Array, PRNGKeyArray
from typing import Callable, Tuple, Union, List

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
        modes: int
            The layer truncates the Fourier series to the first `modes` modes. Remaining modes are set to zero.
    """
    weight_real: Array
    weight_imag: Array
    in_c: int = eqx.field(static=True)
    out_c: int = eqx.field(static=True)
    modes: int = eqx.field(static=True)
    norm: str = eqx.field(static=True)

    def __init__(self, in_c: int, out_c: int, modes: int, initializer: jax.nn.initializers.Initializer = None, 
                 fft_norm: str="ortho", *, key: PRNGKeyArray):
        self.in_c = in_c
        self.out_c = out_c
        self.modes = modes // 2 + 1
        key_real, key_imag = jax.random.split(key)
        if not initializer is None:
            self.weight_real = initializer(key_real, (in_c, out_c, self.modes))
            self.weight_imag = initializer(key_imag, (in_c, out_c, self.modes))
        else:
            scale = 1.0 / (in_c * out_c)
            self.weight_real = random.uniform(key_real, (in_c, out_c, self.modes), minval=-scale, maxval=scale)
            self.weight_imag = random.uniform(key_imag, (in_c, out_c, self.modes), minval=-scale, maxval=scale)
        self.norm = fft_norm

    def __call__(self, x: ArrayLike, **kwargs) -> ArrayLike:
        """
        Apply the layer to the input.

        Args:
            x: ArrayLike
                The input tensor. Should be of shape `(in_c, d1)`.

        Returns:
            ArrayLike
                The output tensor. Will be of shape `(out_c, d1)`.
        """
        x_ft = jnp.fft.rfft(x, norm=self.norm) # shape: (in_c, d1//2 + 1)
        out_ft = jnp.zeros((self.out_c, x_ft.shape[1]), dtype=jnp.complex64) # (out_c, d1//2)
        out_ft = out_ft.at[:, :self.modes].set(jnp.einsum('ik,iok->ok', x_ft[:, :self.modes], self.weight_real + 1j * self.weight_imag))
        return jnp.fft.irfft(out_ft, norm=self.norm) # shape: (out_c, d1)        
    
class SpectralConv2d(eqx.Module):

    weight_1: Array
    weight_2: Array
    in_c: int = eqx.field(static=True)
    out_c: int = eqx.field(static=True)
    modes: List[int] = eqx.field(static=True)
    norm: str = eqx.field(static=True)

    def __init__(self, in_c: int, out_c: int, modes: List[int], initializer: jax.nn.initializers.Initializer, 
                 fft_norm: str="ortho", *, key: PRNGKeyArray):
        assert all(i % 2 == 0 for i in modes) 
        self.in_c = in_c
        self.out_c = out_c
        modes[-1] = modes[-1] // 2 + 1
        self.modes = modes
        key_1, key_2 = random.split(key)
        self.weight_1 = initializer(key_1, (in_c, out_c, self.modes[0]//2, self.modes[1]), dtype=jnp.complex64)
        self.weight_2 = initializer(key_2, (in_c, out_c, self.modes[0]//2, self.modes[1]), dtype=jnp.complex64)
        self.norm = fft_norm

    def __call__(self, x: Array, **kwargs) -> Array:
        """
        Args:
            x: ArrayLike
                Input tensor of shape `(in_c, d1, d2)`.

        Returns:
            ArrayLike
                The output tensor. 
        """
        x_ft = jnp.fft.rfft2(x, norm=self.norm)     # (in_c, d1, d2 // 2 + 1)
        out_ft = jnp.zeros((self.out_c, *x_ft.shape[1:]), dtype=jnp.complex64)
        out_ft = out_ft.at[:, :self.modes[0]//2, :self.modes[1]].set(jnp.einsum('ixy,ioxy->oxy', 
                                                                                x_ft[:, :self.modes[0]//2, :self.modes[1]], 
                                                                                self.weight_1))
        out_ft = out_ft.at[:, -self.modes[0]//2:, :self.modes[1]].set(jnp.einsum('ixy,ioxy->oxy', 
                                                                                 x_ft[:, -self.modes[0]//2:, :self.modes[1]], 
                                                                                 self.weight_2))
        return jnp.fft.irfft2(out_ft, s=x.shape[1:], norm=self.norm)
    

class FNOBlock(eqx.Module):
    spec_conv: eqx.Module
    residual_net: eqx.Module
    k_modes: List[int] = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(self, in_channels: int, out_channels: int, k_modes: List[int], activation: Callable,
                 spec_conv: eqx.Module, residual_net: eqx.Module):
        self.k_modes = k_modes
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spec_conv = spec_conv
        self.residual_net = residual_net

    def __call__(self, x: Array, **kwargs) -> Array:
        """
        Args:
            x: ArrayLike
                Input tensor of shape `(in_c, d1, ..., dn)`.

        Returns:
            ArrayLike
                The output tensor of shape `(out_c, d1, ..., dn)`.
        """
        return self.activation(self.spec_conv(x) + self.residual_net(x))

class FNOBlock1d(FNOBlock):

    def __init__(self, in_channels: int, out_channels: int, k_modes: int, activation: Callable, key: PRNGKeyArray):
        key, subkey = random.split(key)
        spec_conv = SpectraclConv1d(in_channels, out_channels, k_modes, glorot_uniform(), key=key)
        residual_net = eqx.nn.Conv1d(in_channels, out_channels, 1, 1, key=subkey)
        super().__init__(in_channels, out_channels, [k_modes], activation, spec_conv, residual_net)
    
class FNOBlock2d(FNOBlock):
    
    def __init__(self, in_channels: int, out_channels: int, k_modes: List[int], activation: Callable, key: PRNGKeyArray):
        key, subkey = random.split(key)
        spec_conv = SpectralConv2d(in_channels, out_channels, k_modes, glorot_uniform(), key=key)
        residual_net = eqx.nn.Conv2d(in_channels, out_channels, 1, 1, key=subkey)
        super().__init__(in_channels, out_channels, k_modes, activation, spec_conv, residual_net)
            
class FNO(eqx.Module):

    fourier_blocks: eqx.nn.Sequential
    projection_input: eqx.Module
    projection_output: eqx.Module
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(self, in_channels: int, out_channels: int, fourier_blocks: eqx.nn.Sequential, p_in: eqx.Module=None, 
                 p_out: eqx.Module=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        if p_in is None or p_out is None:
            print("Warning: Using Identity for input/output projection.")
        self.projection_input = p_in if p_in is not None else eqx.nn.Identity()
        self.projection_output = p_out if p_out is not None else eqx.nn.Identity()
        self.fourier_blocks = fourier_blocks
    
    def __call__(self, x: ArrayLike, **kwargs) -> ArrayLike:
        """
        Apply the layer to the input.

        Args:
            x: ArrayLike
                The input tensor. Should be of shape `(in_c, d1)`.

        Returns:
            ArrayLike
                The output tensor. Will be of shape `(out_c, d1)`.
        """
        return self.projection_output(self.fourier_blocks(self.projection_input(x)))
    
    