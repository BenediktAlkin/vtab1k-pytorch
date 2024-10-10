import math
from typing import Mapping, Any

import torch
import torch.nn.functional as F
from torch import nn


class DoraLinear(nn.Linear):
    def __init__(
            self,
            # nn.Linear
            in_features: int,
            out_features: int,
            bias: bool = True,
            # lora
            lora_rank: int = 0,
            lora_alpha: int = 1,
            lora_init: str = "torch",
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            fan_in_fan_out: bool = False,
            **kwargs
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, **kwargs)
        # lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_init = lora_init

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if lora_rank > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((lora_rank, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, lora_rank)))
            self.scaling = self.lora_alpha / self.lora_rank
            self.m = nn.Parameter(self.weight.new_zeros((1, in_features)))
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    @classmethod
    def convert(
            cls,
            module,
            lora_rank=0,
            lora_init="torch",
            names_to_exclude=None,
            name=None,
    ):
        module_output = module
        if type(module) == nn.Linear:
            if names_to_exclude is None or name not in names_to_exclude:
                module_output = cls(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    lora_rank=lora_rank,
                    lora_init=lora_init,
                )
                module_output.weight = module.weight
                module_output.bias = module.bias
                module_output.m.data.copy_(module.weight.norm(p=2, dim=0, keepdim=True))
        for child_name, child in module.named_children():
            module_output.add_module(
                child_name,
                cls.convert(
                    module=child,
                    lora_rank=lora_rank,
                    lora_init=lora_init,
                    names_to_exclude=names_to_exclude,
                    name=child_name if name is None else f"{name}.{child_name}",
                ),
            )
        del module
        return module_output

    def reset_parameters(self):
        super().reset_parameters()
        self.reset_parameters_lora()

    def reset_parameters_lora(self):
        # requires to check hasattr because reset_parameters is called after nn.Linear __init__
        # lora weights are initialized after -> reset_parameters is called again after lora weights are defined
        if not hasattr(self, "lora_A"):
            return
        nn.init.zeros_(self.lora_B)
        if self.lora_init == "torch":
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        elif self.lora_init == "normal_one_over_rank":
            nn.init.normal_(self.lora_A, mean=0, std=1 / self.lora_rank)
        elif self.lora_init == "truncnormal002":
            nn.init.trunc_normal_(self.lora_A, std=2e-2)
        elif self.lora_init == "truncnormal2e5":
            nn.init.trunc_normal_(self.lora_A, std=2e-5)
        elif self.lora_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.lora_A)
        else:
            raise NotImplementedError
        self.m.data.copy_(self.weight.norm(p=2, dim=0, keepdim=True))

    def change_lora_rank(self, new_rank):
        if self.lora_rank == new_rank:
            return
        self.lora_rank = new_rank
        self.lora_A = nn.Parameter(self.weight.new_zeros((new_rank, self.in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, new_rank)))
        self.scaling = self.lora_alpha / max(self.lora_rank, 1)
        self.m = nn.Parameter(self.weight.new_zeros((1, self.in_features)))
        self.reset_parameters_lora()

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.lora_rank > 0:
            lora = self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1) * self.scaling
            combined_weight = self.weight + lora.T
            column_norm = combined_weight.norm(p=2, dim=0, keepdim=True)
            V = combined_weight / column_norm
            new_weight = self.m * V
            return F.linear(x, T(new_weight), bias=self.bias)
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
