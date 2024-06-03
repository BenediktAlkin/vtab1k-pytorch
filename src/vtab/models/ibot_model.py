import torch
from torch import nn

from vtab.adapters.split_qkv import SplitQKV


class IBotModel(nn.Module):
    def __init__(self, identifier, input_shape, num_outputs, split_qkv=False):
        # "in1k_ibot_l16"
        super().__init__()
        assert input_shape == (3, 224, 224)
        model = torch.hub.load("BenediktAlkin/torchhub-ssl", identifier)
        if split_qkv:
            model = SplitQKV.convert(model)
        self.model = model
        self.head = nn.Linear(self.model.dim, num_outputs)
        nn.init.trunc_normal_(self.head.weight, std=2e-5)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.model(x)[:, 0])

    @property
    def dim(self):
        return self.model.dim
