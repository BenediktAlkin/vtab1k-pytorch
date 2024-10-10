import math
from typing import Mapping, Any

import torch
import torch.nn.functional as F
from torch import nn


class LoraLinear(nn.Linear):
    def __init__(
            self,
            # nn.Linear
            in_features: int,
            out_features: int,
            bias: bool = True,
            # lora
            lora_rank: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            lora_init: str = "torch",
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            post_layer_norm: bool = False,
            pre_batch_norm: bool = False,
            **kwargs
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, **kwargs)
        # lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.lora_init = lora_init
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        self.fan_in_fan_out = fan_in_fan_out
        self.post_layer_norm = post_layer_norm
        self.pre_batch_norm = pre_batch_norm
        # Actual trainable parameters
        if lora_rank > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((lora_rank, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, lora_rank)))
            self.scaling = self.lora_alpha / self.lora_rank
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        if self.post_layer_norm:
            self.post_ln = nn.LayerNorm(out_features)
            self.merge_weights = False
        if self.pre_batch_norm:
            self.pre_bn = nn.BatchNorm1d(in_features, affine=False)
            self.merge_weights = False

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
                module_output = LoraLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    lora_rank=lora_rank,
                    lora_init=lora_init,
                )
                module_output.weight = module.weight
                module_output.bias = module.bias
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

    def change_lora_rank(self, new_rank):
        if self.lora_rank == new_rank:
            return
        self.lora_rank = new_rank
        self.lora_A = nn.Parameter(self.weight.new_zeros((new_rank, self.in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, new_rank)))
        self.scaling = self.lora_alpha / max(self.lora_rank, 1)
        self.reset_parameters_lora()

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.lora_rank > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.lora_rank > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.lora_rank > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.pre_batch_norm:
                x = self.pre_bn(x)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            if self.post_layer_norm:
                result = self.post_ln(result)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
