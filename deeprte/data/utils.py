# Copyright 2022 Zheng Ma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for geometry."""

import jax
import jax.numpy as jnp
import numpy as np


def cartesian_product(*arrays):
    """Compute cartesian product of arrays
    with different shapes in an efficient manner.

    Args:
        arrays: each array shoud be rank 2 with shape (N_i, d_i).
        inds: indices for each array, should be rank 1.

    Returns:
        Cartesian product of arrays with shape (N_1, N_2, ..., N_n, sum(d_i)).
    """
    d = [*map(lambda x: x.shape[-1], arrays)]
    ls = [*map(len, arrays)]
    inds = [*map(np.arange, ls)]

    dtype = np.result_type(*arrays)
    arr = np.empty(ls + [sum(d)], dtype=dtype)

    for i, ind in enumerate(np.ix_(*inds)):
        arr[..., sum(d[:i]) : sum(d[: i + 1])] = arrays[i][ind]
    return arr


@jax.jit
def jax_cartesian_product(*arrays):
    """Compute cartesian product of arrays
    with different shapes in an efficient manner.

    Args:
        arrays: each array shoud be rank 2 with shape (N_i, d).
        inds: indices for each array, should be rank 1.

    Returns:
        Cartesian product of arrays with shape (N_1, N_2, ..., N_n, n * d).
    """
    d = [*map(lambda x: x.shape[-1], arrays)]
    ls = [*map(len, arrays)]
    inds = [*map(jnp.arange, ls)]

    dtype = jnp.result_type(*arrays)
    arr = jnp.empty(ls + [sum(d)], dtype=dtype)

    for i, ind in enumerate(jnp.ix_(*inds)):
        arr = arr.at[..., sum(d[:i]) : sum(d[: i + 1]) * d].set(arrays[i][ind])
    return arr
