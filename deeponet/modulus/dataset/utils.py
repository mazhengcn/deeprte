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


def normalize(data):
    _min = np.min(data)
    _range = np.max(data) - _min
    return (data - np.min(data)) / _range, _min, _range
