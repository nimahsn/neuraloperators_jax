import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad, value_and_grad, jacfwd, jacrev
import optax
import equinox as eqx
from functools import partial

class DeepONet(eqx.Module):
    '''
    DeepONet model. Combines a branch network and a trunk network.

    Args:
        branch_net: equinox.Module
            The branch network.
        trunk_net: equinox.Module
            The trunk network.
    '''

    branch_net: eqx.Module
    trunk_net: eqx.Module

    def __init__(self, branch_net, trunk_net):
        self.branch_net = branch_net
        self.trunk_net = trunk_net

    def __call__(self, sensors, inputs):
        b = self.branch_net(sensors)
        t = self.trunk_net(inputs)
        return jnp.sum(b * t, axis=-1, keepdims=True)
    
