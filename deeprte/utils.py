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


import collections

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
