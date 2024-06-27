import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Tuple, Union, List
from jaxtyping import PRNGKeyArray

class DoubleConv(eqx.Module):
    conv1: eqx.nn.Conv
    conv2: eqx.nn.Conv
    activation: Callable = eqx.field(static=True)

    def __init__(self, num_spatial_dims: int, in_channels: int, out_channels: int, activation: Callable, 
                 kernel_size: int = 3, padding: Union[int, str] = "valid", *args, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key, 2)
        self.conv1 = eqx.nn.Conv(num_spatial_dims, in_channels, out_channels, kernel_size, 
                                      1, padding, *args, key=key1)
        self.conv2 = eqx.nn.Conv(num_spatial_dims, out_channels, out_channels, kernel_size,
                                      1, padding, *args, key=key2)
        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply the double convolution block to the input tensor x.

        Args:
            x: input tensor of shape (in_channels, *spatial_dims)

        Returns:
            output tensor of shape (out_channels, *spatial_dims)
        """
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x
    
class UNetConv(eqx.Module):
    lifting_block: eqx.Module
    downsample_blocks: List[eqx.nn.Conv]
    left_blocks: List[DoubleConv]
    right_blocks: List[DoubleConv]
    upsample_blocks: List[eqx.nn.ConvTranspose]
    projection_block: eqx.nn.Conv

    def __init__(self, 
                 num_input_channels: int, 
                 num_output_channels: int, 
                 num_spatial_dims: int, 
                 num_levels: int,
                 activation: Callable,
                 num_lifted_channels: int=64, 
                 list_num_channels: List[int]=None,
                 conv_kernel_size: int=3,
                 conv_padding: Union[str, int]="same", 
                 downsample_kernel_size: int=3, 
                 downsample_stride: int=2, 
                 downsample_padding: Union[str, int]= 1,
                 *, 
                 key):
        
        if list_num_channels is None:
            list_num_channels = [num_lifted_channels * 2**i for i in range(num_levels + 1)]
        else:
            assert len(list_num_channels) == num_levels + 1, "list_num_channels must have length num_levels + 1"

        key, lift_key, proj_key = jax.random.split(key, 3)
        self.lifting_block = eqx.nn.Conv(num_spatial_dims, num_input_channels, num_lifted_channels, conv_kernel_size, 
                                              1, conv_padding, key=lift_key)
        self.projection_block = eqx.nn.Conv(num_spatial_dims, list_num_channels[0], num_output_channels, 1, 
                                              1, conv_padding, key=proj_key)
        
        self.downsample_blocks = []
        self.left_blocks = []
        self.right_blocks = []
        self.upsample_blocks = []

        for up_c, down_c in zip(list_num_channels[:-1], list_num_channels[1:]):
            key, down_key, left_key, right_key, up_key = jax.random.split(key, 5)

            self.downsample_blocks.append(eqx.nn.Conv(num_spatial_dims, up_c, up_c, downsample_kernel_size, 
                                                      downsample_stride, downsample_padding, key=down_key))
            self.left_blocks.append(DoubleConv(num_spatial_dims, up_c, down_c, activation, conv_kernel_size,
                                                conv_padding, key=left_key))
            self.right_blocks.insert(0, DoubleConv(num_spatial_dims, down_c, up_c, activation, conv_kernel_size,
                                                 conv_padding, key=right_key))
            self.upsample_blocks.insert(0, eqx.nn.ConvTranspose(num_spatial_dims, down_c, up_c, downsample_kernel_size, 
                                                           downsample_stride, padding=1, output_padding=1, key=up_key))
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply the UNet to the input tensor x.

        Args:
            x: input tensor of shape (num_input_channels, *spatial_dims)

        Returns:
            output tensor of shape (num_output_channels, *spatial_dims)
        """

        skip_memory = []
        x = self.lifting_block(x)

        for down, left in zip(self.downsample_blocks, self.left_blocks):
            skip_memory.append(x)
            x = down(x)
            x = left(x)

        for up, right in zip(self.upsample_blocks, self.right_blocks):
            x = up(x)
            x = jnp.concatenate([x, skip_memory.pop()], axis=0)
            x = right(x)

        return self.projection_block(x)
    
    