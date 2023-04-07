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


import collections.abc

import jax
import numpy as np


def accumulate_gradient(grad_fn, params, batch, accum_steps):
    """Accumulate gradient over multiple steps to save on memory."""
    batch_size = batch["psi_label"].shape[0]
    print(batch_size)

    if accum_steps and accum_steps > 1:
        assert (
            batch_size % accum_steps == 0
        ), f"Bad accum_steps {accum_steps} for batch size {batch_size}"
        step_size = batch_size // accum_steps

        def dynamic_slice_feat(feat_dict, i, step_size):
            def slice_fn(x):
                return jax.lax.dynamic_slice(
                    x, (i,) + (0,) * (x.ndim - 1), (step_size,) + x.shape[1:]
                )

            return jax.tree_map(slice_fn, feat_dict)

        # loss, (scalars, state) = ret
        l_and_state = grad_fn(params, dynamic_slice_feat(batch, 0, step_size))

        def acc_grad_and_loss(i, l_and_state):
            sliced_batch = dynamic_slice_feat(batch, i * step_size, step_size)
            grads_i, (scalars_i, state_i) = grad_fn(params, sliced_batch)
            grads, (scalars, state) = l_and_state
            return jax.tree_map(lambda x, y: x + y, grads, grads_i), (
                jax.tree_map(lambda x, y: x + y, scalars, scalars_i),
                state_i,
            )

        grads, (scalars, state) = jax.lax.fori_loop(
            1, accum_steps, acc_grad_and_loss, l_and_state
        )
        return jax.tree_map(
            lambda x: x / accum_steps, (grads, (scalars, state))
        )
    else:
        return grad_fn(params, batch)


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
