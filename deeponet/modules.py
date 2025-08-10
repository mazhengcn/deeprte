from functools import reduce
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from modulus.key import Key

# from common_modules import Mlp


class Arch(nn.Module):
    """
    Base class for all neural networks
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys

    def to_tensor(self, dict_vars: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Converts a dictionary of variables to a tensor

        Parameters
        ----------
        dict_vars : Dict[str, torch.Tensor]
            Dictionary of variables

        Returns
        -------
        torch.Tensor
            Tensor of variables
        """
        assert len(dict_vars) == 1, "Only one variable can be converted to tensor"
        return list(dict_vars.values())[0]


class DeepONet(Arch):
    def __init__(
        self,
        branch_net_list: List[Arch],
        trunk_net: Arch,
        output_keys: List[Key] = None,
        name="deeponet",
        **kwargs
    ) -> None:
        super().__init__(input_keys=[], output_keys=output_keys)
        assert len(output_keys) == 1, "DeepONet only supports one output variable"
        for i, model in enumerate(branch_net_list):
            self.add_module("branch" + str(i), model)
        self.trunk_net = trunk_net

        # self.output_linear = torch.nn.Linear(
        #     self.trunk_net.output_keys[0].size, output_keys[0].size, bias=False
        # )

        self.num_branches = len(branch_net_list)
        self.branch_keys = [key for b in branch_net_list for key in b.input_keys]
        self.trunk_keys = trunk_net.input_keys
        self.input_keys = self.branch_keys + self.trunk_keys

        assert self.trunk_net.output_keys[0].name == "trunk"
        # assert self.branch_net_list[0]

        self.apply(_init_weights)

    def preprocess_input(
        self, in_vars: Dict[str, torch.Tensor], input_keys: List[Key]
    ) -> Dict[str, torch.Tensor]:
        """Preprocess input variables

        Parameters
        ----------
        in_vars : Dict[str, torch.Tensor]
            Input variables

        Returns
        -------
        Dict[str, torch.Tensor]
            Preprocessed input variables
        """

        inputs = {key.name: in_vars[key.name] for key in input_keys}
        return inputs

    def forward(self, in_vars: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        branch_input = self.preprocess_input(in_vars, self.branch_keys)
        trunk_input = self.preprocess_input(in_vars, self.trunk_keys)

        branch_output = [
            self.to_tensor(getattr(self, "branch" + str(i))(branch_input))
            for i in range(self.num_branches)
        ]
        branch_output = reduce(torch.mul, branch_output)
        trunk_output = self.to_tensor(self.trunk_net(trunk_input))

        # print(branch_output.shape, trunk_output.shape)

        if branch_output.shape != trunk_output.shape:
            if self.training:
                _output = torch.einsum("ik,ijk->ij", branch_output, trunk_output)
            else:
                _output = torch.einsum("...k,...jk->...j", branch_output, trunk_output)
        else:
            _output = torch.sum(branch_output * trunk_output, axis=-1)

        # _output = torch.exp(_output)
        output = {self.output_keys[0].name: _output}

        return output


class FullyConnected(Arch):
    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        hidden_units: List[int],
        activation: nn.Module = nn.ReLU,
        out_act: bool = False,
        name="fullyconnected",
        **kwargs
    ):
        super().__init__(input_keys=input_keys, output_keys=output_keys)
        assert len(input_keys) == 1, "FullyConnected only supports one input variable"
        assert len(output_keys) == 1, "FullyConnected only supports one output variable"

        num_layers = len(hidden_units) + 1
        units = [input_keys[0].size] + hidden_units + [output_keys[0].size]

        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(units[i], units[i + 1]))
            self.layers.append(activation())

        self.layers.append(nn.Linear(units[-2], units[-1]))
        if out_act:
            self.layers.append(activation())

        # self.apply(_init_weights)

    def forward(self, in_vars: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = in_vars[self.input_keys[0].name]
        for layer in self.layers:
            x = layer(x)
        return {self.output_keys[0].name: x}


class ModifiedMlp(Arch):
    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        hidden_units: List[int],
        activation: nn.Module = nn.ReLU,
        name="modifiedmlp",
        **kwargs
    ):
        super().__init__(input_keys=input_keys, output_keys=output_keys)
        assert len(input_keys) == 1, "ModifiedMlp only supports one input variable"
        assert len(output_keys) == 1, "ModifiedMlp only supports one output variable"
        assert (
            len(set(hidden_units)) == 1
        ), "ModifiedMlp hidden layer units should be same"

        num_layers = len(hidden_units) + 1
        units = [input_keys[0].size] + hidden_units + [output_keys[0].size]

        self.u_encode = nn.Sequential(activation(), nn.Linear(units[0], units[1]))
        self.v_encode = nn.Sequential(activation(), nn.Linear(units[0], units[1]))

        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(units[i], units[i + 1]))
            self.hidden_layers.append(activation())

        self.output_layer = nn.Linear(units[-2], units[-1])

        # self.apply(_init_weights)

    def forward(self, in_vars: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = in_vars[self.input_keys[0].name]
        u = self.u_encode(x)
        v = self.v_encode(x)
        # print(u, v)
        for i, layer in enumerate(self.hidden_layers):
            if i % 2 == 0:
                x = (1 - layer(x)) * u + layer(x) * v
            else:
                x = layer(x)
        x = self.output_layer(x)
        return {self.output_keys[0].name: x}


class ResNet(Arch):
    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        hidden_units: List[int],
        activation: nn.Module = nn.ReLU,
        name="resnet",
        **kwargs
    ):
        super().__init__(input_keys=input_keys, output_keys=output_keys)
        assert len(input_keys) == 1, "FullyConnected only supports one input variable"
        assert len(output_keys) == 1, "FullyConnected only supports one output variable"
        self.assert_units(hidden_units)

        num_layers = len(hidden_units) + 1
        units = [input_keys[0].size] + hidden_units + [output_keys[0].size]

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(units[0], units[1]))
        self.layers.append(activation())

        for i in range(1, num_layers - 2, 2):
            self.layers.append(ResBlock(units[i], units[i + 1], activation))

        self.layers.append(nn.Linear(units[-2], units[-1]))

        # self.apply(_init_weights)

    def forward(self, in_vars: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = in_vars[self.input_keys[0].name]
        for layer in self.layers:
            x = layer(x)
        return {self.output_keys[0].name: x}

    def assert_units(self, units: List[int]) -> None:
        assert len(units) % 2 == 1, "ResNet hidden layer units should be odd"
        for i in range(len(units) // 2):
            assert (
                units[i * 2] == units[i * 2 + 2]
            ), "ResNet hidden layer units should be symmetric"


class ResBlock(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, activation: nn.Module = nn.ReLU
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, in_features)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x += identity
        x = self.activation(x)
        return x


@torch.no_grad()
def _init_weights(module: nn.Module):
    """weight initialization"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    elif hasattr(module, "init_weights"):
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in depth-first order.
        module.init_weights()
