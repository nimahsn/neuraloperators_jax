"""
utilities for saving and loading models.
"""

import json
import os
import equinox as eqx
import jax.random as jr

def save_model(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def load_model(filename, make):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = make(key=jr.PRNGKey(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model)