import functools as ft
from collections.abc import Iterable

import jax
from grain import DataLoader


class MultiHostDataLoadIterator:
    """fold get_next_batch_sharded into a iterator class"""

    def __init__(self, dataloader: DataLoader, sharding: jax.P | None):
        self.dataloader = dataloader
        self.sharding = sharding if sharding else jax.P()

        if isinstance(self.dataloader, Iterable):
            self.local_iterator = iter(self.dataloader)
        else:
            raise ValueError(
                "Type error: dataloader should be either tf.data.Dataset or Iterable."
            )

    def reset(self):
        if isinstance(self.dataloader, Iterable):
            self.local_iterator = iter(self.dataloader)
        else:
            raise ValueError(
                "Type error: dataloader should be either tf.data.Dataset or grain.DataLoader."
            )

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        return jax.tree.map(
            ft.partial(jax.make_array_from_process_local_data, self.sharding),
            next(self.local_iterator),
        )
