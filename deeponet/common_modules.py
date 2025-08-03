import torch
import torch.nn as nn
from typing import Callable, Sequence


class Mlp(nn.Module):
    def __init__(
        self,
        units: Sequence[int],
        activation: Callable = nn.ReLU,
        out_act: bool = True,
        name="mlp",
        **kwargs
    ):
        super().__init__()

        num_layers = len(units) - 1
        assert num_layers >= 1, "num_layers must be greater than or equal to 1"

        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(units[i], units[i + 1]))
            self.layers.append(activation())

        self.layers.append(nn.Linear(units[-2], units[-1]))
        if out_act:
            self.layers.append(activation())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
