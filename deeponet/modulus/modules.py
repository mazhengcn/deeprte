from functools import reduce
from typing import Dict, List, Tuple, Union

import torch
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Arch
from torch import Tensor


class MioBranchNet(Arch):
    def __init__(
        self,
        branch_net_list: List[Arch],
        output_keys: List[Key] = None,
        detach_keys: List[Key] = [],
    ) -> None:
        super().__init__(
            input_keys=[],
            output_keys=output_keys,
            detach_keys=detach_keys,
        )
        for i, model in enumerate(branch_net_list):
            self.add_module(str(i), model)
        self.num_branches = len(branch_net_list)

        # self.branch_net_list = branch_net_list
        self.input_keys = [key for b in branch_net_list for key in b.input_keys]
        self.input_key_dict = {str(var): var.size for var in self.input_keys}

        branch_index_name = [
            "branch_slice_index_" + str(i) for i in range(self.num_branches)
        ]
        for i in range(self.num_branches):
            index = self.prepare_slice_index(
                self.input_key_dict, branch_net_list[i].input_key_dict.keys()
            )
            self.register_buffer(branch_index_name[i], index, persistent=False)

    def _tensor_forward(self, x: Tensor) -> None:
        # print(x.shape, x)
        _output = [
            c._tensor_forward(
                self.slice_input(
                    x, getattr(self, "branch_slice_index_" + str(i)), dim=-1
                )
            )
            for i, c in enumerate(self.children())
        ]
        # print(_output)
        # output = torch.stack(_output)
        # output = torch.prod(output, dim=0)
        output = reduce(torch.mul, _output)

        output = self.process_output(output, self.output_scales_tensor)
        return output

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.concat_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        y = self._tensor_forward(x)
        return self.split_output(y, self.output_key_dict, dim=-1)
