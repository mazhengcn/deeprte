import collections
import io
import os
from typing import Mapping

import haiku as hk
import jax.numpy as jnp
import numpy as np


def to_flat_dict(d, parent_key="", sep="//"):
    """Flattens a dictionary, keeping empty leaves."""
    items = []
    for k, v in d.items():
        path = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.Mapping):
            items.extend(to_flat_dict(v, path, sep=sep).items())
        else:
            items.append((path, v))

    # Keeps the empty dict if it was set explicitly.
    if parent_key and not d:
        items.append((parent_key, {}))

    return dict(items)


def flat_dict_to_rte_data(flat_dict):
    """Convert a dictionary of NumPy arrays to Haiku parameters."""
    rte_data = {}
    for path, array in flat_dict.items():
        scope, name = path.split("/")
        if scope not in rte_data:
            rte_data[scope] = {}
        rte_data[scope][name] = np.asarray(array)

    return rte_data


def flat_params_to_haiku(params: Mapping[str, np.ndarray]) -> hk.Params:
    """Convert a dictionary of NumPy arrays to Haiku parameters."""
    hk_params = {}
    for path, array in params.items():
        scope, name = path.split("//")
        if scope not in hk_params:
            hk_params[scope] = {}
        hk_params[scope][name] = jnp.array(array)

    return hk_params


def get_model_haiku_params(model_name: str, data_dir: str) -> hk.Params:
    """Get the Haiku parameters from a model name."""

    path = os.path.join(data_dir, "params", f"params_{model_name}.npz")

    with open(path, "rb") as f:
        params = np.load(io.BytesIO(f.read()), allow_pickle=False)

    return flat_params_to_haiku(params)
