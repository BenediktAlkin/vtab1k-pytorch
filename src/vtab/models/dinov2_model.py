import os

import torch
from torch import nn

from vtab.adapters.split_qkv import SplitQKV
from vtab.adapters.split_swigluffn import SplitSwigluffn


class DINOv2Model(nn.Module):
    def __init__(
            self,
            identifier,
            input_shape,
            num_outputs,
            head_init="truncnormal",
            split_qkv=True,
            split_swigluffn=True,
    ):
        # DINOv2
        # dinov2_vits14
        # dinov2_vitb14
        # dinov2_vitl14
        # dinov2_vitg14
        # DINOv2 with registers
        # dinov2_vits14_reg
        # dinov2_vitb14_reg
        # dinov2_vitl14_reg
        # dinov2_vitg14_reg
        super().__init__()
        self.identifier = identifier
        assert input_shape == (3, 224, 224)
        if "g14" in identifier:
            # disable xformers because it fuses calls to nn.Linear by using the weights directly
            # which circumevents LoraLinear or EvaLinear
            os.environ["XFORMERS_DISABLED"] = "true"
        model = torch.hub.load("facebookresearch/dinov2", identifier)
        if split_qkv:
            model = SplitQKV.convert(model)
        if "g14" in identifier and split_swigluffn:
            model = SplitSwigluffn.convert(model)

        self.model = model
        self.head = nn.Linear(self.model.embed_dim, num_outputs)
        if head_init == "none":
            pass
        elif head_init == "truncnormal":
            nn.init.trunc_normal_(self.head.weight, std=2e-5)
            nn.init.zeros_(self.head.bias)
        else:
            raise NotImplementedError

    def __str__(self):
        return self.identifier

    def forward(self, x):
        return self.head(self.model(x))

    @property
    def dim(self):
        return self.model.dim
